"""Paired fixed-source PA/non-FD attention component performance, not public E2E."""

import json
import os
from pathlib import Path
import sys

import torch

from tests.mla.support.components import paged_attention as suite
from tests.mla.support.mla_bsnd_stride_profile import profile
from tests.mla.support.mla_dense_build import ROOT, sha256


def run(probes, producer, output, layout="bsnd"):
    if layout == "tnd":
        from tests.mla.support.components import tnd_paged_attention as active_suite
    else:
        assert layout == "bsnd"
        active_suite = suite
    probes, output = Path(probes).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    drivers = {
        Path(__file__).resolve(),
        Path(active_suite.__file__).resolve(),
        Path(suite.__file__).resolve(),
        ROOT / "tests/mla/support/mla_bsnd_stride_profile.py",
    }
    (output / "driver-sources.json").write_text(
        json.dumps({str(p.relative_to(ROOT)): sha256(p) for p in sorted(drivers)}, indent=2) + "\n"
    )
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    # Persist both fixed cases and source dispatch before the first sample.
    (output / "matrix.json").write_text(json.dumps(active_suite.PERFORMANCE_CASES, indent=2) + "\n")
    records = {}
    for dtype in ("fp16", "bf16"):
        path = probes / dtype / ("tnd-schedules.json" if layout == "tnd" else "schedules.json")
        record = json.loads(path.read_text())
        count = len(active_suite.PERFORMANCE_CASES)
        records[dtype] = {key: value[:count] for key, value in record.items()}
        records[dtype]["source_evidence_sha256"] = sha256(path)
    (output / "dispatch.json").write_text(json.dumps(records, indent=2) + "\n")
    scope = (
        f"Standalone {layout.upper()} PA attention versus pinned unmodified CANN PA/non-FD Kernel; "
        "identical logical inputs, M/S2 blocks, core count and endpoints. Migrated timing includes "
        "v6/current-length/full-page validation. Metadata generation and all allocation/upload are outside timing. "
        "Q and cache-row/table strides outside the CANN ABI are materialized before source timing; native "
        "independent K/V page strides are preserved. TND Q effective segments are independently packed "
        "before source timing; PA KV lengths remain per-sample. Wall includes ctypes, launch and synchronization with "
        "preallocated buffers, and is not public PA or metadata E2E. Native FD dispatch is retained separately."
    )
    results = {}
    for dtype in ("fp16", "bf16"):
        profile(
            probes / dtype,
            producer,
            output / dtype,
            suite=active_suite,
            kernel_prefix="paged",
            kernel_suffix="attention",
            scope=scope,
            seed=20261011,
        )
        report = json.loads((output / dtype / "summary.json").read_text())
        for i, case in enumerate(report["results"]):
            assert case["config"] == active_suite.PERFORMANCE_CASES[i]
            assert case["expected_metadata"] == records[dtype]["metadata"][i]
            case["native_fd"] = bool(records[dtype]["native"][i][5])
            case["wall_ratio"] = case["migrated"]["wall_us"] / case["oracle"]["wall_us"]
        results[dtype] = report["results"]
        torch.npu.empty_cache()
    (output / "comparison.json").write_text(
        json.dumps(dict(scope=scope, results=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    run(*sys.argv[1:])
