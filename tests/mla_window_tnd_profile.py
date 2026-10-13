"""TND band native adapter; shared varlen timing retains packing and restoration."""

import torch
from tests.common.compare import assert_fa_close
from tests.mla_source_compare import assert_mla_source_close
from tests.mla_varlen_used_profile import CannTndUsedCall
from tests.test_flash_attn_npu_v3_mla_window_attention import normalized_window


def window_for(case):
    s, args = case["spec"], case["args"]
    return normalized_window(
        dict(
            shape=(len(s["q"]), args["max_seqlen_q"], s["heads"], args["max_seqlen_k"]),
            window=s["window"],
            causal=s.get("causal", False),
        )
    )


class WindowTndCall(CannTndUsedCall):
    def __init__(self, case):
        args = case["args"]
        inputs = {name: args[name] for name in case["raw"]}
        super().__init__(
            inputs,
            case["spec"],
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


def accuracy(case, actual, source):
    from tests.test_flash_attn_npu_v3_mla_window_tnd_public import effective, reference_pairs

    window = window_for(case)
    uq, uk = effective(case["spec"])
    references = reference_pairs(case, window)
    actual, source = ([t.cpu() for t in pair] for pair in (actual, source))
    qs = ks = 0
    reports = []
    for b, (q, k) in enumerate(zip(uq, uk)):
        if q:
            _, refs = references[b]
            if k == 0:
                refs[1].fill_(-torch.inf)
                refs[3].fill_(-torch.inf)
            got = [actual[0][qs : qs + q].unsqueeze(0), actual[1][:, qs : qs + q].unsqueeze(0)]
            src = [source[0][qs : qs + q].unsqueeze(0), source[1][:, qs : qs + q].unsqueeze(0)]
            masked = (
                q
                if k == 0
                else (
                    0
                    if window.kind == 0
                    else max(0, min(q, q - k - (window.right if window.kind == 2 else 0)))
                )
            )
            assert (got[0][:, :masked] == 0).all() and got[1][:, :, :masked].isneginf().all()
            assert (src[0][:, :masked] == 0).all() and src[1][:, :, :masked].isposinf().all()
            src[1][:, :, :masked] = -torch.inf
            comparison = {}
            for i, name in enumerate(("out", "lse")):
                if torch.equal(got[i], src[i]):
                    comparison[name] = assert_mla_source_close(
                        got[i], src[i], refs[i], refs[i + 2], name=name
                    )
                else:
                    assert_fa_close(got[i], refs[i], refs[i + 2], name=name)
                    assert_fa_close(src[i], refs[i], refs[i + 2], name="native-" + name)
                    comparison[name] = dict(source="different", math="passed")
            reports.append(comparison)
        qs += case["spec"]["q"][b]
        ks += case["spec"]["kv"][b]
    return reports


if __name__ == "__main__":
    import sys
    from tests.mla_causal_tnd_profile import profile
    from tests.mla_window_tnd_cases import PERFORMANCE_CASES
    from tests.test_flash_attn_npu_v3_mla_window_tnd_public import prepare, call

    profile(
        *sys.argv[1:],
        specs=PERFORMANCE_CASES,
        prepare_case=prepare,
        call_forward=call,
        native_class=WindowTndCall,
        check_accuracy=accuracy,
        mask="window",
        metadata_name="ComputeMlaWindowTndMetadata",
    )
