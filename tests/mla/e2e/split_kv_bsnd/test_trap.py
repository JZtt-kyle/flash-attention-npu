"""Release device rejection before output writes, including empty capacities."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
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
        "split",
        "window",
        "snapshot",
        "empty",
        "stale",
        "negative",
        "over",
        "q-used",
        "split-stale",
        "window-stale",
    ],
)
def test_release_trap(tmp_path, dtype, fault):
    directory = tmp_path / f"{dtype}-{fault}"
    directory.mkdir()
    with (directory / "output.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.mla.support.mla_split_bsnd_error_probe",
                dtype,
                fault,
                str(directory),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    assert result.returncode == 0, (directory / "output.log").read_text()
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] == "event" and report["fault"] == fault
    assert (
        report["extension_sha256"]
        == hashlib.sha256(Path(api.flash_attn_npu_3.__file__).read_bytes()).hexdigest()
    )
