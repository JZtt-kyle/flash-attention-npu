"""Profile mixed-layout attention against fixed CANN with preallocated buffers.

Run as python3 -m tests.mla.support.mla_tnd_bsnd_profile PROBES PRODUCER_MANIFEST OUTPUT.
This is a non-FD component comparison, not public metadata E2E.
"""

import argparse
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

from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.components.tnd_bsnd_attention import (
    PERFORMANCE_CASES,
    check_case,
    configure_producer,
    launch,
    load_attention,
    make_case,
)
from tests.mla.support.components.metadata import load_probe


def profile(probe_dir, producer_manifest, output):
    probes = load_attention(probe_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(producer_manifest).read_text())
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    assert sha256(Path(manifest["library"])) == manifest["library_sha256"]
    with load_probe(manifest["library"], output / "producer") as producer:
        configure_producer(producer)
        cases = [make_case(probes, producer, spec) for spec in PERFORMANCE_CASES]
        accuracy = [check_case(probes, case) for case in cases]
        names = ("oracle", "migrated")

        def call(index, name):
            launch(probes, cases[index], name)
            torch.npu.synchronize()

        (output / "npu-before.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        for index in range(len(cases)):
            for _ in range(5):
                for name in names:
                    call(index, name)
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
        trace = output / "profiler"
        trace.mkdir(exist_ok=False)
        profiler = torch_npu.profiler
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
        csvs = list(trace.rglob("kernel_details.csv"))
        raws = list(trace.rglob("op_summary_*.csv"))
        assert len(csvs) == len(raws) == 1
        with csvs[0].open() as f:
            rows = sorted(csv.DictReader(f), key=lambda row: Decimal(row["Start Time(us)"].strip()))
        with raws[0].open() as f:
            raw = sorted(
                csv.DictReader(f), key=lambda row: Decimal(row["Task Start Time(us)"].strip())
            )
        assert len(rows) == len(raw) == len(labels)
        for row, source, label in zip(rows, raw, labels):
            assert row["Name"] == source["Op Name"] and row["Device_id"] == source["Device_id"]
            assert Decimal(row["Start Time(us)"].strip()) == Decimal(
                source["Task Start Time(us)"].strip()
            )
            assert float(row["Duration(us)"]) == float(source["Task Duration(us)"])
            assert ("cann_mixed" if label["name"] == "oracle" else "mla_mixed") in row["Name"]
            label.update(kernel_us=float(row["Duration(us)"]), kernel_name=row["Name"])
        (output / "kernel-samples.json").write_text(json.dumps(labels, indent=2) + "\n")
        results = []
        for index, record in enumerate(accuracy):
            result = dict(record)
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
            results.append(result)
        (output / "npu-after.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        report = dict(
            scope="Standalone v4-validated TND Q with physical BSND KV attention versus pinned CANN non-FD Kernel, identical schedules and preallocated buffers; not public metadata E2E.",
            empty_lse="FA3 -inf versus source +inf checked separately",
            manifest=str(Path(probe_dir) / "build-manifest.json"),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
            kernel_csv=str(csvs[0]),
            raw_csv=str(raws[0]),
            results=results,
        )
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print([(r["spec"], round(r["kernel_ratio"], 3)) for r in results], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir")
    parser.add_argument("producer_manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    profile(args.probe_dir, args.producer_manifest, args.output)
