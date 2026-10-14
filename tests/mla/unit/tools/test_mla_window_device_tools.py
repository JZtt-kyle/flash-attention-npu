"""Fixed window selection and unchanged independent math mapping, without NPU."""

import pytest
import torch

from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_ref import mla_mask, normalize_fa3_window
from tests.mla.support.mla_window_attention_cases import (
    CHECK_CASES,
    COST_CASES,
    PERFORMANCE_CASES,
    BOUNDARY_CASES,
    EMPTY_CASES,
)


def test_window_partitions_and_timing_inputs_are_complete():
    check = CHECK_CASES + COST_CASES
    rest = [s for s in PERFORMANCE_CASES if s not in check] + BOUNDARY_CASES + EMPTY_CASES
    assert all(s not in rest for s in check)
    assert all(s in check + rest for s in PERFORMANCE_CASES)
    assert len(check) * 2 <= 100 and (len(rest) + 1) * 2 <= 100
    assert runner.SUITES["window-check"][-1] == "representative or cost_boundary"
    assert runner.SUITES["window-attention"][-1] == "not representative and not cost_boundary"
    assert (
        runner.SUITES["regression-dense"]
        and runner.SUITES["regression-nonpaged"]
        and runner.SUITES["regression-paged"]
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ((1, 1), (1, 1)),
        ((32, 6), (-1, -1)),
        ((31, 5), (31, 5)),
        ((-1, 1), (-1, 1)),
        ((1, -1), (1, -1)),
    ],
)
def test_call_bounds_and_used_mask_remain_independent(raw, expected):
    # The call allocates Sq=7, Sk=33; this sample only has three KV tokens.
    bounds = normalize_fa3_window(7, 33, window_size=raw)
    assert bounds == expected
    left, right = bounds
    mask = mla_mask(7, 3, window_size=bounds)
    manual = torch.tensor(
        [
            [
                not ((left < 0 or j >= i - 4 - left) and (right < 0 or j <= i - 4 + right))
                for j in range(3)
            ]
            for i in range(7)
        ]
    )
    assert torch.equal(mask, manual)
