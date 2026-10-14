"""910B installed-CANN checks; these do not exercise the FA3 MLA implementation."""

import pytest
import torch

from tests.mla.support.mla_cann_baseline import CASES, DTYPES, check_case, require_910b


@pytest.fixture(scope="module")
def device():
    # An unavailable/wrong device fails explicitly; no all-skipped acceptance.
    require_910b("npu:0")
    return "npu:0"


@pytest.mark.parametrize("dtype", DTYPES.values(), ids=DTYPES.keys())
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_cann_dense_mla(case, dtype, device):
    call, _ = check_case(case, dtype, device)
    # Reusing prepared inputs must be repeatable and leave all components unchanged.
    inputs = (call.query, call.key, call.value, call.kwargs["query_rope"], call.kwargs["key_rope"])
    before = [(tensor, tensor.clone()) for tensor in inputs if tensor is not None]
    first = tuple(tensor.cpu() for tensor in call())
    second = tuple(tensor.cpu() for tensor in call())
    for actual, expected in zip(second, first):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in before:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
