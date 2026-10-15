"""Explicit TND split public R=1/4 timing with native packing/restoration included."""

import torch
from tests.common.compare import assert_fa_close
from tests.mla.support.public_cases.split_tnd_public import (
    prepare,
    call,
    reference,
    compare,
    effective,
)
from tests.mla.support.mla_window_tnd_profile import WindowTndCall
from tests.mla.support.mla_split_tnd_cases import PERFORMANCE_CASES


def accuracy(case, actual, source):
    refs = reference(case)
    compare(case, actual, refs)
    # Native automatic partition may differ from explicit split. Both are
    # measured against the same original mathematical reference, independently.
    source = [t.cpu() for t in source]
    qs, ks = effective(case["spec"])
    start = 0
    for capacity, q, k in zip(case["spec"]["q"], qs, ks):
        for i, name in enumerate(("out", "lse")):
            idx = (slice(start, start + q),) if i == 0 else (slice(None), slice(start, start + q))
            actual_ref = source[i][idx].clone()
            if i == 1:
                actual_ref[actual_ref.isposinf()] = -torch.inf
            assert_fa_close(
                actual_ref,
                refs[name + "_high"][idx],
                refs[name + "_low"][idx],
                name="native-tnd-" + name,
            )
        start += capacity
    return {"explicit_math": "passed", "native_math": "passed"}


if __name__ == "__main__":
    import sys
    from tests.mla.support.mla_causal_tnd_profile import profile

    profile(
        *sys.argv[1:],
        specs=PERFORMANCE_CASES,
        prepare_case=prepare,
        call_forward=call,
        native_class=WindowTndCall,
        check_accuracy=accuracy,
        mask="window",
        metadata_name="ComputeMlaSplitTndMetadata",
        production_name=("mla_split_tnd_partial_", "mla_split_tnd_merge_"),
    )
