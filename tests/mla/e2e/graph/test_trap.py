"""Release graph replay must still enforce live device values."""

import hashlib
import json
import os
from pathlib import Path
import sys
import pytest

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_trap_process import run_isolated
from tests.mla.support.public_cases.graph import LAYOUTS


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation,dtype", [("compressed", "fp16"), ("prefill", "bf16")])
def test_stale_metadata_trap(tmp_path, layout, representation, dtype):
    directory = (
        Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path))
        / "traps"
        / f"{layout}-{representation}-{dtype}"
    )
    directory.mkdir(parents=True)
    result = run_isolated(
        [
            sys.executable,
            "-m",
            "tests.mla.support.mla_graph_error_probe",
            layout,
            representation,
            dtype,
            str(directory),
        ],
        directory / "output.log",
    )
    assert result["returncode"] == 0 and not result["timed_out"], (
        directory / "output.log"
    ).read_text()
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] == "event"
    assert (
        report["extension_sha256"]
        == hashlib.sha256(Path(api.flash_attn_npu_3.__file__).read_bytes()).hexdigest()
    )
