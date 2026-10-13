"""Same-schedule window component timing; preallocated probes, not public E2E."""

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

from tests import test_flash_attn_npu_v3_mla_window_attention as suite


def profile(probes, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    names = ("oracle", "migrated")
    (output / "matrix.json").write_text(json.dumps(suite.PERFORMANCE_CASES, indent=2) + "\n")
    (output / "npu-before.log").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
    try:
        cases = [suite.make_case(probes, spec) for spec in suite.PERFORMANCE_CASES]
        schedules = [
            dict(
                tile=[c["schedule"].m_base, c["schedule"].s2_base],
                cores=c["schedule"].cores,
                endpoints=list(zip(c["schedule"].b_end, c["schedule"].m_end, c["schedule"].s_end)),
                lengths=c["lengths"],
                scale=c["scale"],
                fd=False,
                source_mask=dict(
                    shape=[2048, 2048],
                    sparse_mode={0: 0, 1: 3, 2: 4}[c["windows"]["oracle"].kind],
                    pre=c["windows"]["oracle"].left,
                    next=c["windows"]["oracle"].right,
                ),
            )
            for c in cases
        ]
        (output / "schedules.json").write_text(json.dumps(schedules, indent=2) + "\n")

        def call(index, name):
            assert suite.launch(probes, cases[index], name) == 0
            torch.npu.synchronize()

        for index, case in enumerate(cases):
            for _ in range(5):
                for name in names:
                    call(index, name)
            # The same fixed cases have separate dual-math acceptance. Verify
            # source equality again on the actual allocations used for timing.
            torch.testing.assert_close(
                *[case["outputs"][n][0].cpu() for n in names], rtol=0, atol=0
            )
            for b, length in enumerate(case["lengths"]):
                empty = suite.empty_rows(case, length)
                torch.testing.assert_close(
                    *[case["outputs"][n][1].cpu()[b, :, empty:] for n in names], rtol=0, atol=0
                )
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
            assert prefix + "_window_attention" in row["Name"]
            label.update(kernel_us=float(row["Duration(us)"]), kernel_name=row["Name"])
        (output / "kernel-samples.json").write_text(json.dumps(labels, indent=2) + "\n")
        results = []
        for index, spec in enumerate(suite.PERFORMANCE_CASES):
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
            scope="Independent window/non-FD Kernel versus unmodified fixed CANN with identical schedules and logical inputs. Mask preparation, host schedule, allocation and H2D are outside timing. Wall includes ctypes, launch and synchronization using preallocated outputs/workspace; not public or metadata E2E.",
            dtype=str(probes[0]["dtype"]),
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
