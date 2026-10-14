"""Source precedence accepts numeric limits, never contract or source errors."""

import pytest
import torch

from tests.mla.support.mla_source_compare import assert_mla_source_close


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_source_and_math_pass(dtype):
    value = torch.tensor([0.0, 1.0, -torch.inf, torch.inf], dtype=dtype)
    assert assert_mla_source_close(value, value, value, value) == dict(
        source="equal", math="passed"
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_source_precision_limit_is_recorded(dtype):
    ref = torch.tensor([18.765625], dtype=dtype)
    source = torch.nextafter(ref, torch.full_like(ref, torch.inf))
    result = assert_mla_source_close(source, source, ref, ref)
    assert result["source"] == "equal" and result["math"] == "source_precision_limit"
    assert "max|actual-ref|" in result["error"] and "source-out" in result["source_error"]


def test_source_mismatch_fails_even_when_math_passes():
    ref = torch.tensor([1.0])
    with pytest.raises(AssertionError, match="source mismatch"):
        assert_mla_source_close(ref, ref + 1, ref, ref)


@pytest.mark.parametrize("field", (0, 1, 2, 3))
@pytest.mark.parametrize("invalid", (torch.nan, torch.inf, -torch.inf))
def test_nonfinite_mismatch_is_never_a_source_precision_limit(field, invalid):
    values = [torch.tensor([1.0]) for _ in range(4)]
    values[field][0] = invalid
    with pytest.raises(AssertionError):
        assert_mla_source_close(*values)


def test_identical_nan_source_is_rejected():
    value = torch.tensor([torch.nan])
    with pytest.raises(AssertionError):
        assert_mla_source_close(value, value, value, value)


def test_reference_shape_mismatch_is_rejected():
    value = torch.ones(2)
    with pytest.raises(AssertionError, match="shape"):
        assert_mla_source_close(value, value, value[:1], value)


def test_empty_output():
    value = torch.empty(0)
    assert assert_mla_source_close(value, value, value, value)["math"] == "passed"
