"""Record MLA source precision limits without changing the math comparator."""

import torch

from tests.common.compare import assert_fa_close


def assert_mla_source_close(actual, source, ref, pt, *, name="out"):
    """Require exact source equality; retain math failures as explicit evidence.

    Callers establish the fixed source identity and identical logical inputs,
    and check any FA3 empty-row adaptation before normalizing source LSE.
    Invalid references and observable-contract errors cannot use this exception.
    """
    actual, source, ref, pt = [t.detach().cpu() for t in (actual, source, ref, pt)]
    torch.testing.assert_close(actual, source, rtol=0, atol=0, msg=f"{name}: source mismatch")
    assert actual.shape == ref.shape == pt.shape, f"{name}: reference shape mismatch"
    for label, value in (("actual", actual), ("source", source), ("ref", ref), ("pt", pt)):
        assert not value.isnan().any(), f"{name}: {label} contains NaN"
        assert torch.equal(value.isinf(), actual.isinf()), f"{name}: {label} inf mask mismatch"
        assert torch.equal(value[actual.isinf()], actual[actual.isinf()]), (
            f"{name}: {label} inf sign mismatch"
        )
    try:
        assert_fa_close(actual, ref, pt, name=name)
    except AssertionError as error:
        # All structural/nonfinite checks above passed. Confirm the same
        # numerical failure on the exact source, preserving both diagnostics.
        try:
            assert_fa_close(source, ref, pt, name=f"source-{name}")
        except AssertionError as source_error:
            return dict(
                source="equal",
                math="source_precision_limit",
                error=str(error),
                source_error=str(source_error),
            )
        raise
    return dict(source="equal", math="passed")
