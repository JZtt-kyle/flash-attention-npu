"""Mixed band native adapter; all three cache lifecycle segments include conversion."""

import torch
from tests.mla_mixed_profile import CannMixedCall
from tests.mla_window_tnd_profile import accuracy as tnd_accuracy
from tests.test_flash_attn_npu_v3_mla_window_tnd_public import effective
from tests.test_flash_attn_npu_v3_mla_window_attention import normalized_window


def window_for(case):
    s, args = case["spec"], case["args"]
    return normalized_window(
        dict(
            shape=(len(s["q"]), args["max_seqlen_q"], s["heads"], s["capacity"]),
            window=s["window"],
            causal=s.get("causal", False),
        )
    )


class WindowMixedCall(CannMixedCall):
    def __init__(self, case):
        args = case["args"]
        inputs = {name: args[name] for name in case["raw"]}
        qs, ks = effective(case["spec"])
        assert case["spec"]["uq"] is None
        super().__init__(
            inputs,
            qs,
            ks,
            args["softmax_scale"] if args["softmax_scale"] is not None else 576**-0.5,
        )
        window = window_for(case)
        if window.kind:
            self.native.kwargs.update(
                sparse_mode=3 if window.kind == 1 else 4,
                atten_mask=torch.ones(
                    (2048, 2048), dtype=torch.bool, device=inputs["q"].device
                ).triu(1),
                pre_tokens=2**31 - 1 if window.kind == 1 else window.left,
                next_tokens=0 if window.kind == 1 else window.right,
            )

    def description(self):
        return {
            name: self.native.kwargs[name] for name in ("sparse_mode", "pre_tokens", "next_tokens")
        }


def mapped(case):
    return dict(case, args=dict(case["args"], max_seqlen_k=case["spec"]["capacity"]))


def accuracy(case, actual, source):
    return tnd_accuracy(mapped(case), actual, source)


if __name__ == "__main__":
    import sys
    from tests.mla_causal_mixed_profile import profile
    from tests.mla_window_mixed_cases import PERFORMANCE_CASES
    from tests.test_flash_attn_npu_v3_mla_window_mixed_public import prepare, call, metadata

    profile(
        *sys.argv[1:],
        specs=PERFORMANCE_CASES,
        prepare_case=prepare,
        call_forward=call,
        generate_metadata=metadata,
        native_class=WindowMixedCall,
        check_accuracy=accuracy,
        mask="window",
        metadata_name="ComputeMlaWindowMixedMetadata",
    )
