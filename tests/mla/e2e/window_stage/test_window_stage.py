"""M09 combinations across all five public window layouts, no quota sampling."""

import pytest
from tests.mla.e2e.test_regression import prepare, compare, invoke, metadata

LAYOUTS = ("window_bsnd", "window_tnd", "window_mixed", "window_paged", "window_paged_tnd")
# Each named tuple is an indivisible mask/length/stride combination.
CASES = [
    ((0, 0), False, "lengths"),
    ((-1, 2), False, "lengths"),
    ((2, -1), False, "strided_reuse"),
    ((2, 5), True, "strided_reuse"),
    ((99999, 99999), False, "lengths"),
    ((2, 1), False, "empty"),
]


@pytest.mark.parametrize("capability", LAYOUTS)
@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
@pytest.mark.parametrize("window,causal,scenario", CASES)
def test_window_combinations(capability, dtype, window, causal, scenario):
    case = prepare(capability, dtype, scenario)
    case["options"].update(window_size=window, causal=causal)
    compare(case, invoke(case))
    if case["layout"] != "tnd":
        compare(case, invoke(case, record=metadata(case)))
