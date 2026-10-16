"""Native package/dispatch preflight and the complete public metadata timing gate."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tests.mla.support.native_package import native_vendor
from tests.mla.support.mla_native_dispatch import parse_dispatch
from tests.mla.support.mla_prefill_paged_bsnd_cases import PERFORMANCE_CASES
from tests.mla.support.mla_public_performance import summarize
from tests.mla.support.mla_validation_policy import check_public_performance


def test_native_public_profile():
    vendor = native_vendor()
    root = Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile"
    root.mkdir()
    check = os.environ.get("MLA_PREFILL_PAGED_BSND_PROFILE_CHECK") == "1"
    reports = []
    for dtype in ("fp16", "bf16"):
        directory = root / dtype
        env = dict(
            os.environ,
            ASCEND_CUSTOM_OPP_PATH=str(vendor),
            MLA_CANN_VENDOR=str(vendor),
            ASCEND_GLOBAL_LOG_LEVEL="0",
            ASCEND_SLOG_PRINT_TO_STDOUT="1",
        )
        env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        cmd = [
            sys.executable,
            "-m",
            "tests.mla.support.mla_prefill_paged_bsnd_profile",
            "--dtype",
            dtype,
            "--output",
            str(directory),
        ]
        if check:
            cmd.append("--check")
        log = root / f"{dtype}.dispatch.log"
        with log.open("w") as stream:
            subprocess.run(
                cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300
            )
        specs = PERFORMANCE_CASES[:1] if check else PERFORMANCE_CASES
        proof = parse_dispatch(log, dtype, len(specs), vendor, representation="prefill")
        assert all(
            r["native_template"] in (3, 4) and (r["native_tiling_key"] // 10**5 % 10) & 2
            for r in proof["cases"]
        )
        proof["specs"] = specs
        path = root / f"{dtype}.dispatch.json"
        path.write_text(json.dumps(proof, indent=2) + "\n")
        env.update(ASCEND_GLOBAL_LOG_LEVEL="3", ASCEND_SLOG_PRINT_TO_STDOUT="0")
        affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, {min(affinity)})
            with (root / f"{dtype}.timing.log").open("w") as stream:
                subprocess.run(
                    cmd + ["--dispatch", str(path)],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
        finally:
            os.sched_setaffinity(0, affinity)
        reports.append(json.loads((directory / "profile.json").read_text()))
    if not check:
        result = summarize(reports, len(PERFORMANCE_CASES))
        (root / "average.json").write_text(json.dumps(result, indent=2) + "\n")
        check_public_performance(result)
