"""Fixed native identity, cheap fixtures and the equal-configuration public gate."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tests.mla.support.native_package import native_vendor
from tests.mla.support.mla_native_dispatch import parse_dispatch
from tests.mla.support.mla_gqa_options_profile_rows import PERFORMANCE, summarize, selected
from tests.mla.support.mla_validation_policy import check_public_performance


def test_native_public_profile():
    vendor = native_vendor()
    root = Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile"
    root.mkdir()
    check = os.environ.get("MLA_GQA_OPTIONS_PROFILE_CHECK") == "1"
    jobs = []
    for representation in ("compressed", "prefill"):
        for dtype in ("fp16", "bf16"):
            directory = root / f"{representation}-{dtype}"
            env = dict(
                os.environ,
                ASCEND_CUSTOM_OPP_PATH=str(vendor),
                MLA_CANN_VENDOR=str(vendor),
                ASCEND_GLOBAL_LOG_LEVEL="0",
                ASCEND_SLOG_PRINT_TO_STDOUT="1",
            )
            env["LD_LIBRARY_PATH"] = (
                str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
            )
            cmd = [
                sys.executable,
                "-m",
                "tests.mla.support.mla_gqa_options_profile",
                "--representation",
                representation,
                "--dtype",
                dtype,
                "--output",
                str(directory),
            ]
            if check:
                cmd.append("--check")
            log = root / f"{representation}-{dtype}.dispatch.log"
            with log.open("w") as stream:
                subprocess.run(
                    cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600
                )
            proof = parse_dispatch(
                log, dtype, len(selected(check)), vendor, representation=representation
            )
            proof["specs"] = selected(check)
            path = root / f"{representation}-{dtype}.dispatch.json"
            path.write_text(json.dumps(proof, indent=2) + "\n")
            jobs.append((representation, dtype, directory, cmd, env, path))
    if check:
        return
    # All 80 numerical/native fixtures must succeed before any full sampling.
    reports = []
    for representation, dtype, directory, cmd, env, path in jobs:
        env.update(ASCEND_GLOBAL_LOG_LEVEL="3", ASCEND_SLOG_PRINT_TO_STDOUT="0")
        affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, {min(affinity)})
            with (root / f"{representation}-{dtype}.timing.log").open("w") as stream:
                subprocess.run(
                    cmd + ["--dispatch", str(path)],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1200,
                )
        finally:
            os.sched_setaffinity(0, affinity)
        reports.append(json.loads((directory / "profile.json").read_text()))
    result = summarize(reports, PERFORMANCE)
    (root / "average.json").write_text(json.dumps(result, indent=2) + "\n")
    check_public_performance(result)
