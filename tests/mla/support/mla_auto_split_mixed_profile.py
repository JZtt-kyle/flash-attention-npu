"""Native automatic mixed split public R=1/4 timing with native packing/restoration included."""

import torch
from tests.common.compare import assert_fa_close
from tests.mla.e2e.auto_split_mixed.test_public import (
    prepare,
    call,
    reference,
    metadata,
)
from tests.mla.e2e.causal_mixed.test_flash_attn_npu_v3_mla_causal_mixed_public import effective
from tests.mla.support.mla_window_mixed_profile import WindowMixedCall
from tests.mla.support.mla_auto_split_mixed_cases import PERFORMANCE_CASES


def accuracy(case, actual, source):
    refs = reference(case)
    # Native packing can choose different base tiles. Both paths are checked
    # against the same original mathematical reference, independently.
    qs, _ = effective(case["spec"])
    for label, result in (("automatic", actual), ("native", source)):
        result = [t.cpu() for t in result]
        start = 0
        for capacity, q in zip(case["spec"]["q"], qs):
            for i, name in enumerate(("out", "lse")):
                idx = (
                    (slice(start, start + q),) if i == 0 else (slice(None), slice(start, start + q))
                )
                value = result[i][idx].clone()
                if label == "native" and i == 1:
                    value[value.isposinf()] = -torch.inf
                assert_fa_close(
                    value,
                    refs[name + "_high"][idx],
                    refs[name + "_low"][idx],
                    name=label + "-mixed-" + name,
                )
            start += capacity
    return {"automatic_math": "passed", "native_math": "passed"}


if __name__ == "__main__":
    import sys
    from tests.mla.support.mla_causal_mixed_profile import profile

    profile(
        *sys.argv[1:],
        specs=PERFORMANCE_CASES,
        prepare_case=prepare,
        call_forward=call,
        generate_metadata=lambda args: metadata(dict(args=args)),
        native_class=WindowMixedCall,
        check_accuracy=accuracy,
        mask="window",
        metadata_name="ComputeMlaAutoSplitMixedMetadata",
        production_name=(
            "mla_auto_split_mixed_init_",
            "mla_auto_split_mixed_partial_",
            "mla_auto_split_mixed_merge_",
        ),
        automatic_split=True,
    )
