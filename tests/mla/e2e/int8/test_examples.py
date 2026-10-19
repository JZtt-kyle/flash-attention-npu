"""Exercise the documented INT8 workflow through both public query layouts."""

import pytest
import torch

from tests.mla.support.mla_int8_examples import run_example


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
def test_examples(layout):
    initial, scaled, replayed = run_example(layout)
    tnd = layout == "tnd"
    out_shape = (3, 4, 512) if tnd else (3, 2, 4, 512)
    lse_shape = (4, 3) if tnd else (3, 4, 2)
    for result in (initial, scaled, replayed):
        assert len(result) == 4
        out, lse, accum, logs = result
        assert out.shape == out_shape and lse.shape == lse_shape
        assert accum.shape == (26, *lse_shape, 512) and logs.shape == (26, *lse_shape)
        assert out.dtype == torch.bfloat16
        assert lse.dtype == accum.dtype == logs.dtype == torch.float32
        assert torch.isfinite(out).all().item() and torch.isfinite(accum).all().item()
        assert not torch.isnan(lse).any().item() and not torch.isnan(logs).any().item()
    # This also detects stale scale data when an existing record is reused.
    assert torch.count_nonzero(initial[0]).item() > 0
    for index in (0, 2):
        torch.testing.assert_close(scaled[index], initial[index] * 2, rtol=0, atol=0)
    for index in (1, 3):
        torch.testing.assert_close(scaled[index], initial[index], rtol=0, atol=0)
    # The last segment becomes empty on replay, including TND's tail row.
    assert torch.count_nonzero(replayed[0][-1]).item() == 0
    empty_lse = replayed[1][:, -1] if tnd else replayed[1][-1]
    assert torch.isneginf(empty_lse).all().item()
