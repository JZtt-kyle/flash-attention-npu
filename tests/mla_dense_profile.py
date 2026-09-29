"""Dense Kernel diagnostics against a pinned source probe and native CANN.

The probes reuse their schedule, outputs and workspace. Native CANN includes
its public dispatch and output allocation. Neither path is FA3 metadata E2E.
"""

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

from tests.mla_cann_baseline import CannCall, check_vendor_configuration, environment
from tests.mla_dense_build import ROOT, build_probes
from tests.test_flash_attn_npu_v3_mla_dense import check_case, launch, load_probes, make_case


CASES = [
    ((2, 1, 8, 257), None),
    ((2, 1, 8, 257), 0.23),
    ((1, 7, 16, 65), None),
    ((1, 1, 128, 512), None),
    ((2, 2, 128, 511), None),
]
NAMES = ("oracle", "migrated", "native_cann")


def profile(directory, vendor):
    vendor = check_vendor_configuration(vendor)
    directory = Path(directory).resolve()
    probes = load_probes(build_probes(directory), directory)
    device = torch.npu.get_device_name(0)
    assert torch.npu.device_count() == 1
    cases = [make_case(probes, *shape, scale=scale) for shape, scale in CASES]
    native = [CannCall(case["device"], scale=case["scale"]) for case in cases]
    for case, call in zip(cases, native):
        check_case(probes, case)
        result = call()
        torch.npu.synchronize()
        for actual, expected in zip(result, case["outputs"]["migrated"]):
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    source = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    (directory / "native-environment.json").write_text(
        json.dumps(environment(source, vendor), indent=2) + "\n"
    )

    def snapshot(name):
        (directory / name).write_text(subprocess.check_output(["npu-smi", "info"], text=True))

    def call(index, name):
        if name == "native_cann":
            return native[index]()
        launch(probes[0][name], cases[index], name)
        return cases[index]["outputs"][name]

    def order(round_id):
        return NAMES if round_id % 2 == 0 else NAMES[::-1]

    snapshot("npu-before.log")
    for index in range(len(cases)):
        for _ in range(5):
            for name in NAMES:
                result = call(index, name)
                torch.npu.synchronize()
    samples = []
    for index in range(len(cases)):
        for round_id in range(5):
            for iteration in range(20):
                for name in order(round_id):
                    start = time.perf_counter_ns()
                    result = call(index, name)
                    torch.npu.synchronize()
                    samples.append(
                        dict(
                            case=index,
                            name=name,
                            round=round_id,
                            iteration=iteration,
                            call_e2e_us=(time.perf_counter_ns() - start) / 1000,
                        )
                    )
    (directory / "wall-samples.json").write_text(json.dumps(samples, indent=2) + "\n")

    profiler = torch_npu.profiler
    labels = []
    trace = directory / "profiler"
    trace.mkdir(exist_ok=False)
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(
            wait=0, warmup=1, active=len(cases) * 5 * 10 * len(NAMES), repeat=1
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(str(trace)),
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level0,
            export_type=profiler.ExportType.Text,
            data_simplification=False,
        ),
    ) as prof:
        result = call(0, "oracle")
        torch.npu.synchronize()
        prof.step()
        for index in range(len(cases)):
            for round_id in range(5):
                for iteration in range(10):
                    for name in order(round_id):
                        result = call(index, name)
                        torch.npu.synchronize()
                        labels.append(
                            dict(case=index, name=name, round=round_id, iteration=iteration)
                        )
                        prof.step()
    # Hold the last result through synchronization without introducing a kernel.
    del result
    (directory / "profile-labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    csvs = list(trace.rglob("kernel_details.csv"))
    raw_csvs = list(trace.rglob("op_summary_*.csv"))
    assert len(csvs) == len(raw_csvs) == 1, (csvs, raw_csvs)
    with csvs[0].open() as f:
        rows = list(csv.DictReader(f))
    with raw_csvs[0].open() as f:
        raw = list(csv.DictReader(f))
    rows.sort(key=lambda row: Decimal(row["Start Time(us)"].strip()))
    raw.sort(key=lambda row: Decimal(row["Task Start Time(us)"].strip()))
    assert len(rows) == len(raw) == len(labels), (len(rows), len(raw), len(labels))
    assert len({row["Device_id"] for row in rows}) == 1
    prefixes = dict(
        oracle="cann_dense_probe",
        migrated="mla_dense_probe",
        native_cann="FusedInferAttentionScore",
    )
    for row, source_row, label in zip(rows, raw, labels):
        assert row["Name"] == source_row["Op Name"]
        assert row["Device_id"] == source_row["Device_id"]
        assert Decimal(row["Start Time(us)"].strip()) == Decimal(
            source_row["Task Start Time(us)"].strip()
        )
        assert float(row["Duration(us)"]) == float(source_row["Task Duration(us)"])
        assert prefixes[label["name"]] in row["Name"], (label, row["Name"])
        label["kernel_us"] = float(row["Duration(us)"])
        label["kernel_name"] = row["Name"]
        assert 0 < label["kernel_us"] < 1e6
    results = []
    for index, case in enumerate(cases):
        result = dict(
            shape=case["shape"], scale=case["scale"], native_call=native[index].description()
        )
        for name in NAMES:
            kernels = [s for s in labels if s["case"] == index and s["name"] == name]
            walls = [s for s in samples if s["case"] == index and s["name"] == name]
            result[name] = dict(
                median_kernel_us=statistics.median(s["kernel_us"] for s in kernels),
                median_call_e2e_us=statistics.median(s["call_e2e_us"] for s in walls),
                round_mean_kernel_us=[
                    statistics.mean(s["kernel_us"] for s in kernels if s["round"] == r)
                    for r in range(5)
                ],
                round_mean_call_e2e_us=[
                    statistics.mean(s["call_e2e_us"] for s in walls if s["round"] == r)
                    for r in range(5)
                ],
            )
        result["kernel_ratio_to_source_probe"] = (
            result["migrated"]["median_kernel_us"] / result["oracle"]["median_kernel_us"]
        )
        result["kernel_ratio_to_native_cann"] = (
            result["migrated"]["median_kernel_us"] / result["native_cann"]["median_kernel_us"]
        )
        results.append(result)
    snapshot("npu-after.log")
    report = dict(
        scope="Standalone dense Kernel diagnostics; preallocated probes versus native CANN including its output allocation and public dispatch. Not FA3 or metadata E2E.",
        device=device,
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        seed=20260929,
        warmup=5,
        rounds=5,
        wall_iterations=20,
        profiler_iterations=10,
        kernel_csv=str(csvs[0]),
        raw_kernel_csv=str(raw_csvs[0]),
        profiler_scope="Kernel task timing only, cross-checked against raw CANN; framework flow and block-dimension CSV are not used.",
        results=results,
        kernel_samples=labels,
    )
    (directory / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    profile(os.environ["MLA_DENSE_ARTIFACT_DIR"], os.environ["MLA_CANN_VENDOR"])
