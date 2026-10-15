"""192/128 components reuse the existing isolated paired sampling protocol."""

import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
from tests.mla.support.components.prefill_attention import (
    make_case,
    wave,
    launch_case,
    prepare_merge,
    launch_merge,
)
from tests.mla.support.mla_prefill_cases import CASES, PERFORMANCE_IDS
from tests.mla.support.mla_split_kv_attention_profile import sample_components


def profile(probes, output, *, preflight=False):
    (libs, _), dtype = probes
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    configs = [s for s in CASES if s["id"] in PERFORMANCE_IDS]
    cases = [make_case(s, dtype, libs["cores"]) for s in configs]
    tasks = [wave(c, 0) for c in cases]
    assert all(len(c["tasks"]) == c["params"].cores for c in cases)
    (output / "matrix.json").write_text(json.dumps(configs, indent=2) + "\n")
    (output / "tasks.json").write_text(
        json.dumps([[r[0] for r in c["tasks"]] for c in cases]) + "\n"
    )
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    (output / "npu-before.log").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
    try:

        def call(index, name):
            case = cases[index]
            status = launch_case(libs, case, name, tasks[index], partial=case["spec"]["splits"] > 1)
            if status:
                raise RuntimeError(f"{name} prefill launch failed: {status}")
            torch.npu.synchronize()

        for i, c in enumerate(cases):
            for _ in range(5):
                for name in ("oracle", "migrated"):
                    call(i, name)
            if c["spec"]["splits"] > 1:
                for core, (_, _, visible) in enumerate(c["tasks"]):
                    count = visible.shape[0]
                    for a, b in zip(c["raw"]["migrated"], c["raw"]["oracle"]):
                        torch.testing.assert_close(
                            a.cpu()[core, :count],
                            b.cpu()[core, :count],
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )
            else:
                for a, b in zip(c["outputs"]["migrated"], c["outputs"]["oracle"]):
                    torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        # Reuse the same already-executed partial inputs. One M group is timed
        # per fixed configuration; allocation/canonicalization stay outside.
        merge_configs, merges = [], []
        for c in cases:
            if c["spec"]["splits"] > 1:
                raw = {name: [t.cpu() for t in c["raw"][name]] for name in ("oracle", "migrated")}
                merge = prepare_merge(c, raw)
                merge["groups"] = merge["groups"][:1]
                merge_configs.append(c["spec"])
                merges.append(merge)

        def call_merge(index, name, canonical=True):
            code = launch_merge(libs, merges[index], name, canonical=canonical)
            if code:
                raise RuntimeError(f"{name} merge launch failed: {code}")
            torch.npu.synchronize()

        for i, merge in enumerate(merges):
            for _ in range(5):
                for name in ("oracle", "migrated"):
                    call_merge(i, name)
                call_merge(i, "migrated", canonical=False)
            for a, b in zip(merge["outputs"]["migrated"], merge["outputs"]["oracle"]):
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        if preflight:
            report = dict(results=configs, merge=merge_configs, status="fixtures-passed")
            (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
        adaptation = []
        for i in range(len(merges)):
            samples = []
            for _ in range(5):
                for _ in range(20):
                    start = time.perf_counter_ns()
                    call_merge(i, "migrated", canonical=False)
                    samples.append((time.perf_counter_ns() - start) / 1000)
            adaptation.append(
                dict(
                    config=merge_configs[i],
                    wall_us=samples,
                    median_wall_us=statistics.median(samples),
                )
            )
        (output / "adaptation-wall.json").write_text(json.dumps(adaptation, indent=2) + "\n")
        results, paths = sample_components(call, configs, output, "prefill_attention")
        merge_dir = output / "merge"
        merge_dir.mkdir()
        merge_results, merge_paths = sample_components(
            call_merge, merge_configs, merge_dir, "split_kv_merge"
        )
        report = dict(
            scope="192/128 direct or partial component versus fixed generic CANN with identical fragments. Allocation, mask creation, input copies and reference are outside timing. Wall includes ctypes, launch and synchronize with preallocated buffers. Source output initialization is retained. No public or metadata E2E claim.",
            dtype=dtype,
            seed=20261015,
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            device=torch.npu.get_device_name(0),
            mean_kernel_ratio=statistics.mean(r["kernel_ratio"] for r in results),
            mean_wall_ratio=statistics.mean(r["wall_ratio"] for r in results),
            degraded_kernel=sum(r["kernel_ratio"] > 1 for r in results),
            results=results,
            merge=dict(
                scope="128-wide FD on one M group with identical canonical statistics; row adaptation and accumulator writes are excluded from paired timing and reported separately in adaptation-wall.json.",
                results=merge_results,
                adaptation=adaptation,
                **merge_paths,
            ),
            **paths,
        )
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "npu-after.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        return report
    finally:
        os.sched_setaffinity(0, affinity)
