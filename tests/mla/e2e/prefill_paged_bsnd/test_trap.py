"""Untrusted v30 records and live length changes must trap on the device."""

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
    "fault",
    [
        "version",
        "reserved",
        "page-size",
        "tile",
        "used-snapshot",
        "partition-prefix",
        "aic-endpoint",
        "fd-count",
        "fd-split",
        "fd-vector",
        "page-negative",
        "page-over",
        "empty-q-page",
        "page-after-zero-q",
        "physical-empty",
        "used-over",
        "used-negative",
        "used-stale",
        "split-stale",
        "window-stale",
    ],
)
def test_release_trap(tmp_path, dtype, fault):
    directory = (
        Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path)) / "traps" / f"{dtype}-{fault}"
    )
    directory.mkdir(parents=True)
    result = run_isolated(
        [
            sys.executable,
            "-m",
            "tests.mla.support.mla_prefill_paged_bsnd_error_probe",
            dtype,
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
