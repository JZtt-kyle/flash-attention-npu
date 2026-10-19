"""Explicit INT8 TND public performance; never part of daily correctness."""

import json
import os
from pathlib import Path
import subprocess
import sys

from tests.mla.support.mla_int8_native import native_vendor
from tests.mla.support.mla_int8_tnd_profile_rows import selected
from tests.mla.support.mla_native_dispatch import parse_dispatch
from tests.mla.support.mla_validation_policy import check_public_performance


def test_public_performance():
    check = os.environ.get("MLA_INT8_TND_PROFILE_CHECK") == "1"
    root = Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile"
    root.mkdir()
    vendor = native_vendor()
    env = dict(
        os.environ,
        ASCEND_CUSTOM_OPP_PATH=str(vendor),
        MLA_CANN_VENDOR=str(vendor),
        ASCEND_GLOBAL_LOG_LEVEL="0",
        ASCEND_SLOG_PRINT_TO_STDOUT="1",
    )
    env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_int8_bsnd_profile",
        "--layout",
        "tnd",
        "--output",
        str(root),
    ]
    if check:
        command.append("--check")

    def worker(args, log, timeout=1800):
        with (root / log).open("w") as stream:
            subprocess.run(
                command + args,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )

    worker([], "dispatch.log")
    proof = parse_dispatch(
        root / "dispatch.log", "int8", len(selected(check)), vendor, representation="int8"
    )
    proof["specs"] = selected(check)
    path = root / "dispatch.json"
    path.write_text(json.dumps(proof, indent=2) + "\n")
    env.update(ASCEND_GLOBAL_LOG_LEVEL="3", ASCEND_SLOG_PRINT_TO_STDOUT="0")
    affinity = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, {min(affinity)})
        worker(["--dispatch", str(path), "--graph-kernels"], "graph-kernels.log")
        if check:
            return
        worker(["--dispatch", str(path)], "timing.log")
    finally:
        os.sched_setaffinity(0, affinity)
    reports = [
        json.loads((root / mode / "profile.json").read_text()) for mode in ("eager", "graph")
    ]
    for report in reports:
        check_public_performance(report)
