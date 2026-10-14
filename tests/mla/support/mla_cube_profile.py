"""Paired Cube component diagnostics; this is not MLA/FA3 forward E2E."""

import csv
import ctypes
from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch_npu

from tests.mla.support.mla_cube_build import build_probes
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_cube import launch, make_case


SHAPES = [(2, 1, 8, 257), (1, 7, 16, 65), (1, 1, 128, 512), (2, 2, 128, 511)]


def profile(directory):
    directory = Path(directory).resolve()
    libraries = build_probes(directory)
    torch.npu.set_device(0)
    device = torch.npu.get_device_name(0)
    assert "910B" in device and torch.npu.device_count() == 1
    loaded = {}
    for name, path in libraries.items():
        lib = ctypes.CDLL(path)
        lib.launch_cube.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 6
        lib.launch_cube.restype = ctypes.c_int
        loaded[name] = lib
    cases = [make_case(*shape) for shape in SHAPES]
    torch.npu.synchronize()

    def snapshot(name):
        (directory / name).write_text(subprocess.check_output(["npu-smi", "info"], text=True))

    def call(index, name):
        launch(loaded[name], cases[index], name, min(SHAPES[index][0], 20), 1)

    snapshot("npu-before.log")
    for index in range(len(cases)):
        for _ in range(5):
            for name in loaded:
                call(index, name)
                torch.npu.synchronize()
    # Alternate the order each round; use identical completion synchronization.
    samples = []
    for index in range(len(cases)):
        for round_id in range(5):
            order = ["oracle", "migrated"] if round_id % 2 == 0 else ["migrated", "oracle"]
            for iteration in range(20):
                for name in order:
                    start = time.perf_counter_ns()
                    call(index, name)
                    torch.npu.synchronize()
                    samples.append(
                        dict(
                            case=index,
                            name=name,
                            round=round_id,
                            iteration=iteration,
                            probe_e2e_us=(time.perf_counter_ns() - start) / 1000,
                        )
                    )
    (directory / "wall-samples.json").write_text(json.dumps(samples, indent=2) + "\n")

    profiler = torch_npu.profiler
    labels = []
    trace = directory / "profiler"
    trace.mkdir(exist_ok=False)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=1, active=4 * 5 * 10 * 2, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(trace)),
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level0,
            export_type=profiler.ExportType.Text,
            data_simplification=False,
        ),
    ) as prof:
        call(0, "oracle")
        torch.npu.synchronize()
        prof.step()
        for index in range(len(cases)):
            for round_id in range(5):
                order = ["oracle", "migrated"] if round_id % 2 == 0 else ["migrated", "oracle"]
                for iteration in range(10):
                    for name in order:
                        call(index, name)
                        torch.npu.synchronize()
                        labels.append(
                            dict(case=index, name=name, round=round_id, iteration=iteration)
                        )
                        prof.step()
    (directory / "profile-labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    csvs = list(trace.rglob("kernel_details.csv"))
    assert len(csvs) == 1, csvs
    with csvs[0].open() as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: Decimal(row["Start Time(us)"].strip()))
    assert len(rows) == len(labels), (len(rows), len(labels))
    assert len({row["Device_id"] for row in rows}) == 1
    # Direct runtime launches lack framework operator parents in torch_npu's
    # flow parser. Validate task timing against raw CANN output independently.
    raw_csvs = list(trace.rglob("op_summary_*.csv"))
    assert len(raw_csvs) == 1, raw_csvs
    with raw_csvs[0].open() as f:
        raw = list(csv.DictReader(f))
    raw.sort(key=lambda row: Decimal(row["Task Start Time(us)"].strip()))
    assert len(raw) == len(rows)
    for row, source in zip(rows, raw):
        assert row["Name"] == source["Op Name"] and row["Device_id"] == source["Device_id"]
        assert Decimal(row["Start Time(us)"].strip()) == Decimal(
            source["Task Start Time(us)"].strip()
        )
        assert float(row["Duration(us)"]) == float(source["Task Duration(us)"])
    for row, label in zip(rows, labels):
        prefix = "cann_cube_probe" if label["name"] == "oracle" else "mla_cube_probe"
        assert prefix in row["Name"], (label, row["Name"])
        label["kernel_us"] = float(row["Duration(us)"])
        assert 0 < label["kernel_us"] < 1e6
    results = []
    for index, shape in enumerate(SHAPES):
        result = dict(shape=shape)
        for name in loaded:
            result[name] = dict(
                median_kernel_us=statistics.median(
                    s["kernel_us"] for s in labels if s["case"] == index and s["name"] == name
                ),
                median_probe_e2e_us=statistics.median(
                    s["probe_e2e_us"] for s in samples if s["case"] == index and s["name"] == name
                ),
            )
        result["kernel_ratio"] = (
            result["migrated"]["median_kernel_us"] / result["oracle"]["median_kernel_us"]
        )
        results.append(result)
    snapshot("npu-after.log")
    report = dict(
        scope="Preallocated Cube probe: QK + independently supplied P times V; synthetic Vector events. Not MLA forward or FA3 E2E.",
        device=device,
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        seed=20260928,
        warmup=5,
        rounds=5,
        wall_iterations=20,
        profiler_iterations=10,
        kernel_csv=str(csvs[0]),
        raw_kernel_csv=str(raw_csvs[0]),
        profiler_scope="Kernel task timing only; framework flow, block-dimension CSV and counters are not used.",
        results=results,
        kernel_samples=labels,
    )
    (directory / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    profile(os.environ["MLA_CUBE_ARTIFACT_DIR"])
