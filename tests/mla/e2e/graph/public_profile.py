"""Verify every native/graph fixture before separate eager and graph sampling."""

import json
import os
from pathlib import Path
import subprocess
import sys

from tests.mla.support.native_package import native_vendor
from tests.mla.support.mla_native_dispatch import parse_dispatch
from tests.mla.support.mla_graph_profile_rows import PERFORMANCE, selected, summarize
from tests.mla.support.mla_validation_policy import check_public_performance


def test_native_public_profile():
    vendor = native_vendor()
    root = Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile"
    root.mkdir()
    check = os.environ.get("MLA_GRAPH_PROFILE_CHECK") == "1"
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
            command = [
                sys.executable,
                "-m",
                "tests.mla.support.mla_graph_profile",
                "--representation",
                representation,
                "--dtype",
                dtype,
                "--output",
                str(directory),
            ]
            if check:
                command.append("--check")
            log = root / f"{representation}-{dtype}.dispatch.log"
            with log.open("w") as stream:
                subprocess.run(
                    command,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
            proof = parse_dispatch(
                log, dtype, len(selected(check)), vendor, representation=representation
            )
            proof["specs"] = selected(check)
            path = root / f"{representation}-{dtype}.dispatch.json"
            path.write_text(json.dumps(proof, indent=2) + "\n")
            jobs.append((representation, dtype, directory, command, env, path))
    if check:
        # Verify real replay profiler task names/lifecycle on tiny legacy BSND,
        # legacy TND and automatic TND inputs before the full matrix. Other native
        # layouts and dtypes were already captured and replayed above.
        representation, dtype, directory, command, env, path = jobs[0]
        env.update(ASCEND_GLOBAL_LOG_LEVEL="3", ASCEND_SLOG_PRINT_TO_STDOUT="0")
        with (root / "profiler-probe.log").open("w") as stream:
            subprocess.run(
                command + ["--dispatch", str(path), "--profile-probe"],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        return
    reports = dict(eager=[], graph=[])
    for representation, dtype, directory, command, env, path in jobs:
        env.update(ASCEND_GLOBAL_LOG_LEVEL="3", ASCEND_SLOG_PRINT_TO_STDOUT="0")
        affinity = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, {min(affinity)})
            # The first capture in this process must occur during RECORD.
            # Wall graphs are captured separately with profiling disabled.
            with (root / f"{representation}-{dtype}.graph-kernels.log").open("w") as stream:
                subprocess.run(
                    command + ["--dispatch", str(path), "--graph-kernels"],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
            with (root / f"{representation}-{dtype}.timing.log").open("w") as stream:
                subprocess.run(
                    command + ["--dispatch", str(path)],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                )
        finally:
            os.sched_setaffinity(0, affinity)
        for mode in reports:
            reports[mode].append(json.loads((directory / mode / "profile.json").read_text()))
    summaries = []
    for mode, values in reports.items():
        result = summarize(values, PERFORMANCE)
        (root / f"{mode}-average.json").write_text(json.dumps(result, indent=2) + "\n")
        summaries.append(result)
    for result in summaries:
        check_public_performance(result)
