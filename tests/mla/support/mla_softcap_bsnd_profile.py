"""Public capped BSND timing; uncapped execution is an overhead reference only."""

import csv
from decimal import Decimal
import json
import os
import statistics
import subprocess
import time

import torch

from tests.mla.support.public_cases.softcap_bsnd import make_case, metadata, invoke, compare
from tests.mla.support.mla_prefill_bsnd_profile_rows import partition_kernels
from tests.mla.support.mla_split_bsnd_profile_rows import pair_split_kernels


# Fixed before sampling: each split mode and mask, for both representations/dtypes.
PERFORMANCE = [
    dict(id="single-none", splits=1, mask="none", shape=(2, 17, 8, 1057)),
    dict(id="explicit-window", splits=3, mask="window", shape=(2, 17, 8, 1057)),
    dict(id="auto-causal", splits=0, mask="causal", shape=(2, 17, 8, 1057)),
]
MODES = ["generate"] + [
    f"{entry}_{r}" for entry in ("implicit", "generate", "reuse", "func") for r in (1, 4)
]


def kernel_samples(call, representation, dtype, spec, directory):
    import torch_npu

    profiler = torch_npu.profiler
    labels = []
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=1, active=50, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory / "profiler")),
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level0,
            export_type=profiler.ExportType.Text,
            data_simplification=False,
        ),
    ) as prof:
        result = call()
        torch.npu.synchronize()
        prof.step()
        for round_id in range(5):
            for iteration in range(10):
                result = call()
                torch.npu.synchronize()
                labels.append(dict(case=0, name="fa3", round=round_id, iteration=iteration))
                prof.step()
    del result
    csvs = list(directory.rglob("kernel_details.csv"))
    raws = list(directory.rglob("op_summary_*.csv"))
    assert len(csvs) == len(raws) == 1
    with csvs[0].open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: Decimal(r["Start Time(us)"]))
    with raws[0].open() as stream:
        raw = sorted(csv.DictReader(stream), key=lambda r: Decimal(r["Task Start Time(us)"]))
    assert len(rows) == len(raw)
    for row, source in zip(rows, raw):
        assert row["Name"] == source["Op Name"] and row["Device_id"] == source["Device_id"]
        assert Decimal(row["Start Time(us)"]) == Decimal(source["Task Start Time(us)"])
        assert float(row["Duration(us)"]) == float(source["Task Duration(us)"])
    if representation == "prefill":
        samples = partition_kernels(rows, labels, [spec], dtype)
    else:
        paired = pair_split_kernels(rows, labels, dtype, automatic=spec["splits"] == 0)
        samples = [
            dict(
                **label,
                kernel_us=float(row["Duration(us)"]),
                adaptation_us=0,
                tasks=row["Split Segments"],
            )
            for label, row in zip(labels, paired)
        ]
    (directory / "kernel-samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    return samples


def profile(representation, dtype, directory, *, check=False):
    directory.mkdir(parents=True, exist_ok=False)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    (directory / "npu-before.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )
    rows = []
    try:
        for spec in PERFORMANCE:
            c = make_case(representation, dtype, **spec)
            record = metadata(c)
            compare(c, invoke(c, record), record)
            uncapped = dict(c, options=dict(c["options"], softcap=0))
            records = {15: record, 0: metadata(uncapped)}

            def call(mode, cap):
                case = c if cap else uncapped
                if mode == "generate":
                    return metadata(case)
                entry, r = mode.rsplit("_", 1)
                rec = (
                    metadata(case)
                    if entry == "generate"
                    else records[cap]
                    if entry == "reuse"
                    else None
                )
                return [
                    invoke(case, rec, entry="func" if entry == "func" else "cache")
                    for _ in range(int(r))
                ]

            for _ in range(5):
                for cap in (0, 15):
                    for mode in MODES:
                        result = call(mode, cap)
                        torch.npu.synchronize()
            subdir = directory / spec["id"]
            subdir.mkdir()
            if check:
                # Preflight exercises every fixed input and profiler phase parser.
                kernels = kernel_samples(
                    lambda: call("reuse_1", 15), representation, dtype, spec, subdir
                )
                rows.append(dict(spec=spec, kernel_samples=len(kernels)))
                continue
            samples = []
            for round_id in range(5):
                for iteration in range(20):
                    for cap in (0, 15) if round_id % 2 == 0 else (15, 0):
                        for mode in MODES if round_id % 2 == 0 else MODES[::-1]:
                            start = time.perf_counter_ns()
                            result = call(mode, cap)
                            torch.npu.synchronize()
                            samples.append(
                                dict(
                                    cap=cap,
                                    mode=mode,
                                    round=round_id,
                                    iteration=iteration,
                                    e2e_us=(time.perf_counter_ns() - start) / 1000,
                                )
                            )
            del result
            (subdir / "wall-samples.json").write_text(json.dumps(samples, indent=2) + "\n")
            kernels = kernel_samples(
                lambda: call("reuse_1", 15), representation, dtype, spec, subdir
            )
            segments = {}
            for mode in MODES:
                values = {
                    str(cap): statistics.median(
                        x["e2e_us"] for x in samples if x["cap"] == cap and x["mode"] == mode
                    )
                    for cap in (0, 15)
                }
                values["overhead_ratio"] = values["15"] / values["0"]
                segments[mode] = values
            rows.append(
                dict(
                    spec=spec,
                    segments=segments,
                    kernel_us=statistics.median(x["kernel_us"] for x in kernels),
                    adaptation_us=statistics.median(x["adaptation_us"] for x in kernels),
                )
            )
        report = dict(
            scope="Complete public BSND calls and explicit-reuse kernel time. cap=0 is a migrated overhead reference with different semantics, never an equivalent CANN softcap baseline or acceptance ratio.",
            status="fixtures-passed" if check else "sampled",
            representation=representation,
            dtype=dtype,
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            device=torch.npu.get_device_name(0),
            results=rows,
        )
        (directory / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
        (directory / "npu-after.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        return report
    finally:
        os.sched_setaffinity(0, affinity)
