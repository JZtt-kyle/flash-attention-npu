"""Actual device consumers enforce option, used/allocation and page identities."""

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
    "layout,representation,fault",
    [
        ("bsnd", "compressed", "margin"),
        ("bsnd", "prefill", "packed-field"),
        ("tnd", "compressed", "prefix"),
        ("tnd", "prefill", "prefix"),
        ("mixed", "compressed", "version"),
        ("mixed", "prefill", "stale"),
        ("paged_bsnd", "compressed", "page"),
        ("paged_bsnd", "prefill", "softcap"),
        ("paged_tnd", "compressed", "page-zero-q"),
        ("paged_tnd", "prefill", "window"),
    ],
)
def test_release_trap(tmp_path, layout, representation, dtype, fault):
    directory = (
        Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path))
        / "traps"
        / f"{layout}-{representation}-{dtype}-{fault}"
    )
    directory.mkdir(parents=True)
    result = run_isolated(
        [
            sys.executable,
            "-m",
            "tests.mla.support.mla_gqa_options_error_probe",
            layout,
            representation,
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
