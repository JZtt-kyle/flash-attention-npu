"""Feature versions bind even single split, empty rows and reused records."""

import hashlib
import json
import os
from pathlib import Path
import sys
import pytest
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_trap_process import run_isolated


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "representation,splits,fault",
    [
        ("compressed", 3, "old"),
        ("compressed", 0, "reverse"),
        ("compressed", 1, "version"),
        ("compressed", 3, "negative"),
        ("prefill", 1, "old"),
        ("prefill", 3, "reverse"),
        ("prefill", 0, "stale"),
        ("prefill", 1, "mask"),
        ("compressed", 0, "prefix"),
        ("prefill", 3, "prefix"),
        ("compressed", 3, "max"),
        ("prefill", 0, "max"),
    ],
)
def test_release_trap(tmp_path, representation, dtype, splits, fault):
    directory = (
        Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path))
        / "traps"
        / f"{representation}-{dtype}-{splits}-{fault}"
    )
    directory.mkdir(parents=True)
    result = run_isolated(
        [
            sys.executable,
            "-m",
            "tests.mla.support.mla_softcap_mixed_error_probe",
            representation,
            dtype,
            str(splits),
            fault,
            str(directory),
        ],
        directory / "output.log",
    )
    assert result["returncode"] == 0 and not result["timed_out"], (
        directory / "output.log"
    ).read_text()
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] == "event" and report["fault"] == fault
    assert (
        report["extension_sha256"]
        == hashlib.sha256(Path(api.flash_attn_npu_3.__file__).read_bytes()).hexdigest()
    )
