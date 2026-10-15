"""Independent FD merge component timing; preallocated probes, not public E2E."""

import csv
from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch_npu

from tests.mla.support.components import split_kv_merge as suite
from tests.mla.support.mla_split_kv_merge_cases import PERFORMANCE_CASES


def profile(probes, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    names = ("oracle", "migrated")
    (output / "matrix.json").write_text(json.dumps(PERFORMANCE_CASES, indent=2) + "\n")
    (output / "npu-before.log").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
    try:
        cases = [suite.make_case(probes, spec) for spec in PERFORMANCE_CASES]
        # One independently validated M block per fixed input; partial work,
        # row canonicalization and allocations stay outside merge timing.
        for case in cases:
            case["groups"] = case["groups"][:1]
        schedules = [
            dict(
                config=c["spec"],
                rows=c["groups"][0]["params"].rows,
                splits=c["groups"][0]["params"].splits,
            )
            for c in cases
        ]
        (output / "schedules.json").write_text(json.dumps(schedules, indent=2) + "\n")

        def call(index, name):
            assert suite.launch(probes, cases[index], name, canonical=True) == 0
            torch.npu.synchronize()

        for index, case in enumerate(cases):
            for _ in range(5):
                for name in names:
                    call(index, name)
            for a, b in zip(case["outputs"]["oracle"], case["outputs"]["migrated"]):
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        # Adapter+accumulator cost is separate; native has no FA3 accumulator.
        adaptation = []
        for case in cases:
            for _ in range(5):
                assert suite.launch(probes, case, "migrated") == 0
                torch.npu.synchronize()
            values = []
            for _ in range(5):
                for _ in range(20):
                    start = time.perf_counter_ns()
                    assert suite.launch(probes, case, "migrated") == 0
                    torch.npu.synchronize()
                    values.append((time.perf_counter_ns() - start) / 1000)
            adaptation.append(values)
        (output / "adaptation-wall.json").write_text(json.dumps(adaptation) + "\n")
        walls = []
        for index in range(len(cases)):
            for round_id in range(5):
                for iteration in range(20):
                    for name in names if round_id % 2 == 0 else names[::-1]:
                        start = time.perf_counter_ns()
                        call(index, name)
                        walls.append(
                            dict(
                                case=index,
                                name=name,
                                round=round_id,
                                iteration=iteration,
                                wall_us=(time.perf_counter_ns() - start) / 1000,
                            )
                        )
        (output / "wall.json").write_text(json.dumps(walls, indent=2) + "\n")
        labels = []
        profiler = torch_npu.profiler
        trace = output / "profiler"
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            schedule=profiler.schedule(wait=0, warmup=1, active=len(cases) * 5 * 10 * 2, repeat=1),
            on_trace_ready=profiler.tensorboard_trace_handler(str(trace)),
            experimental_config=profiler._ExperimentalConfig(
                profiler_level=profiler.ProfilerLevel.Level0,
                export_type=profiler.ExportType.Text,
                data_simplification=False,
            ),
        ) as prof:
            call(0, "oracle")
            prof.step()
            for index in range(len(cases)):
                for round_id in range(5):
                    for iteration in range(10):
                        for name in names if round_id % 2 == 0 else names[::-1]:
                            call(index, name)
                            labels.append(
                                dict(case=index, name=name, round=round_id, iteration=iteration)
                            )
                            prof.step()
        (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
        csvs, raws = list(trace.rglob("kernel_details.csv")), list(trace.rglob("op_summary_*.csv"))
        assert len(csvs) == len(raws) == 1
        with csvs[0].open() as stream:
            rows = sorted(
                csv.DictReader(stream), key=lambda r: Decimal(r["Start Time(us)"].strip())
            )
        with raws[0].open() as stream:
            raw = sorted(
                csv.DictReader(stream), key=lambda r: Decimal(r["Task Start Time(us)"].strip())
            )
        assert len(rows) == len(raw) == len(labels)
        for row, source, label in zip(rows, raw, labels):
            assert row["Name"] == source["Op Name"] and row["Device_id"] == source["Device_id"]
            assert Decimal(row["Start Time(us)"].strip()) == Decimal(
                source["Task Start Time(us)"].strip()
            )
            assert float(row["Duration(us)"]) == float(source["Task Duration(us)"])
            prefix = "cann" if label["name"] == "oracle" else "mla"
            assert prefix + "_split_kv_merge" in row["Name"]
            label.update(kernel_us=float(row["Duration(us)"]), kernel_name=row["Name"])
        (output / "kernel-samples.json").write_text(json.dumps(labels, indent=2) + "\n")
        results = []
        for index, spec in enumerate(PERFORMANCE_CASES):
            result = dict(config=spec)
            for name in names:
                result[name] = dict(
                    kernel_us=statistics.median(
                        s["kernel_us"] for s in labels if s["case"] == index and s["name"] == name
                    ),
                    wall_us=statistics.median(
                        s["wall_us"] for s in walls if s["case"] == index and s["name"] == name
                    ),
                )
            result["kernel_ratio"] = result["migrated"]["kernel_us"] / result["oracle"]["kernel_us"]
            result["wall_ratio"] = result["migrated"]["wall_us"] / result["oracle"]["wall_us"]
            results.append(result)
        report = dict(
            scope="Independent FD merge on the same canonical FP32 O/sum/max, one M block per configuration. Partial attention, row canonicalization, allocation and H2D are outside timing; wall includes ctypes, launch and completion. Separate adaptation-wall samples include migrated raw-row adaptation and FA3 accumulator stores. No public E2E claim.",
            dtype=str(probes["dtype"]),
            seed=20260929,
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            device=torch.npu.get_device_name(0),
            kernel_csv=str(csvs[0]),
            raw_csv=str(raws[0]),
            results=results,
        )
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "npu-after.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        return report
    finally:
        os.sched_setaffinity(0, affinity)
