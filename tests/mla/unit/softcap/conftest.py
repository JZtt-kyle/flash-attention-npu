import os
from pathlib import Path

import pytest
import torch

from tests.mla.support.components.softcap import load


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def probes(representation, dtype, tmp_path_factory):
    torch.npu.set_device(0)
    assert "910B" in torch.npu.get_device_name(0)
    directory = Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path_factory.mktemp("softcap")))
    return load(representation, dtype, directory)
