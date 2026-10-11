"""Fixed paired Copy component matrix; no attention or public PA E2E claim."""

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

from tests.mla_paged_copy_build import load_probes, sha256
from tests.test_flash_attn_npu_v3_mla_paged_copy import check_case, launch, make_case


# Fixed before sampling. Query layout describes the workload, not an input to
# Copy. Repeat cache tiles for one/three Q blocks, with short/long and zero KV.
PERFORMANCE_CASES = [
    dict(page=p, query_layout=layout, lengths=lengths, query_blocks=qblocks, profile=profile)
    for p in (16, 112, 128, 256, 528, 1024)
    for layout, lengths, qblocks, profile in (
        ("BSND", [257, 0], [1, 1], "contiguous"),
        ("TND", [8193, 1025, 0], [3, 1, 0], "independent"),
    )
]


def tasks_for(config):
    page = config["page"]
    s2 = 512 // page * page if page < 512 and 512 % page else 512
    tasks = []
    for b, (length, queries) in enumerate(zip(config["lengths"], config["query_blocks"])):
        for _ in range(queries):
            for start in range(0, length, s2):
                end = min(length, start + s2)
                for r in range(start, end, 128):
                    rows = min(128, end - r)
                    tasks.extend([[b, r, rows, (rows + 15) // 16 * 16, mode] for mode in (0, 1)])
                for r in range(start, end, 256):
                    rows = min(256, end - r)
                    tasks.extend(
                        [[b, r, rows, (rows + 15) // 16 * 16, mode] for mode in range(2, 6)]
                    )
    for b, length in enumerate(config["lengths"]):
        if length == 0:
            tasks.append([b, 0, 0, 16, 0])
    return tasks


def profile(probe_dir, directory):
    probe_dir, directory = Path(probe_dir).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    (directory / "matrix.json").write_text(json.dumps(PERFORMANCE_CASES, indent=2) + "\n")
    (directory / "npu-before.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )
    names = ("oracle", "migrated")
    reports = {}
    for dtype_name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        output = directory / dtype_name
        output.mkdir()
        libraries = load_probes(probe_dir, dtype_name)
        cases = []
        for index, config in enumerate(PERFORMANCE_CASES):
            tasks = tasks_for(config)
            case = make_case(dtype, config["page"], config["profile"], tasks, diagnostic=False)
            # Check first/last tiles of every mode before sampling. The separate
            # suite checks all 64 page sizes and the full boundary matrix.
            selected = []
            for mode in range(6):
                candidates = [t for t in tasks if t[-1] == mode]
                selected += candidates[:1] + candidates[-1:]
            check = make_case(dtype, config["page"], config["profile"], selected)
            check_case((libraries, output, dtype), check, label=f"case-{index}")
            del check
            cases.append(case)
        torch.npu.synchronize()

        def call(index, name):
            launch(libraries, cases[index], name, dump=False, rounds=1)
            torch.npu.synchronize()

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
        profiler = torch_npu.profiler
        labels = []
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
        with csvs[0].open() as f:
            rows = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Start Time(us)"].strip()))
        with raws[0].open() as f:
            raw = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
        assert len(rows) == len(raw) == len(labels)
        for row, original, label in zip(rows, raw, labels):
            prefix = "cann_paged_copy" if label["name"] == "oracle" else "mla_paged_copy"
            assert prefix in row["Name"] and row["Name"] == original["Op Name"]
            assert row["Device_id"] == original["Device_id"]
            assert Decimal(row["Start Time(us)"].strip()) == Decimal(
                original["Task Start Time(us)"].strip()
            )
            assert float(row["Duration(us)"]) == float(original["Task Duration(us)"])
            label["kernel_us"] = float(row["Duration(us)"])
            assert 0 < label["kernel_us"] < 1e6
        results = []
        for index, config in enumerate(PERFORMANCE_CASES):
            item = dict(config=config, tasks=len(cases[index]["tasks"]))
            for name in names:
                item[name] = dict(
                    kernel_us=statistics.median(
                        x["kernel_us"] for x in labels if x["case"] == index and x["name"] == name
                    ),
                    wall_us=statistics.median(
                        x["wall_us"] for x in walls if x["case"] == index and x["name"] == name
                    ),
                )
            item["kernel_ratio"] = item["migrated"]["kernel_us"] / item["oracle"]["kernel_us"]
            item["wall_ratio"] = item["migrated"]["wall_us"] / item["oracle"]["wall_us"]
            results.append(item)
        report = dict(
            results=results, kernel_samples=labels, kernel_csv=str(csvs[0]), raw_csv=str(raws[0])
        )
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        reports[dtype_name] = results
        cases.clear()
        torch.npu.empty_cache()
    (directory / "npu-after.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )
    report = dict(
        scope="Copy component: task/page reads, GM-to-L1, a 512-byte MTE1 sink per tile and two-slot events. Excludes diagnostic dump/initialization. No metadata, attention or public E2E.",
        source_scope="Pinned unmodified Copy headers; same physical views, test-only stride injection. Not native FIA host dispatch.",
        warmup=5,
        rounds=5,
        wall_samples_per_round=20,
        kernel_samples_per_round=10,
        affinity=list(os.sched_getaffinity(0)),
        device=torch.npu.get_device_name(0),
        loaded_manifest_sha256=sha256(probe_dir / "build-manifest.json"),
        results=reports,
    )
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({dtype: [r["kernel_ratio"] for r in data] for dtype, data in reports.items()}))


if __name__ == "__main__":
    profile(os.environ["MLA_PAGED_COPY_DIR"], os.environ["MLA_PAGED_COPY_PROFILE_DIR"])
