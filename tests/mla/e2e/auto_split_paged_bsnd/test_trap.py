"""Release device rejection before output writes, including empty capacities."""

import hashlib
import json
import os
from pathlib import Path
import sys
from tests.mla.support.mla_trap_process import run_isolated
import pytest
from flash_attn_npu_3 import flash_attn_npu_interface as api


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "kind",
        "prefix",
        "prefix-high",
        "aic",
        "fd",
        "vector",
        "window",
        "snapshot",
        "empty",
        "stale-k",
        "negative",
        "over",
        "q-used",
        "page-negative",
        "page-over",
        "page-zero-used",
        "budget",
        "version-stale",
        "window-stale",
    ],
)
def test_release_trap(tmp_path, dtype, fault):
    directory = (
        Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path)) / "traps" / f"{dtype}-{fault}"
    )
    directory.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_auto_split_paged_bsnd_error_probe",
        dtype,
        fault,
        str(directory),
    ]
    result = run_isolated(command, directory / "output.log")
    (directory / "process.json").write_text(json.dumps(result, indent=2) + "\n")
    assert not result["timed_out"] and result["returncode"] == 0, (
        directory / "output.log"
    ).read_text()
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] == "event" and report["fault"] == fault
    assert (
        report["extension_sha256"]
        == hashlib.sha256(Path(api.flash_attn_npu_3.__file__).read_bytes()).hexdigest()
    )
