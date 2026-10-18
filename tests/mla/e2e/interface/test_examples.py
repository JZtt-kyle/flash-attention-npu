"""Keep the user-facing examples executable for both supported dtypes."""

import pytest
import torch

from tests.mla.support.mla_examples import run_examples


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_examples(dtype):
    results = run_examples(dtype)
    for result in results:
        assert result[0].dtype == dtype
        assert result[1].dtype == torch.float32
        assert torch.isfinite(result[0]).all().item()
        assert not torch.isnan(result[1]).any().item()
