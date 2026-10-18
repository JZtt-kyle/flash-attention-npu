"""Public capped mixed timing; uncapped execution is an overhead reference only."""

import csv
from decimal import Decimal
import json
import os
import statistics
import subprocess
import time

import torch

from tests.mla.support.public_cases.softcap_mixed import make_case, metadata, invoke, compare
from tests.mla.support.mla_prefill_bsnd_profile_rows import partition_kernels
from tests.mla.support.mla_split_tnd_profile_rows import production_sample


# Fixed before sampling: each split mode and mask, for both representations/dtypes.
PERFORMANCE = [
    dict(
        id="compressed-single-none",
        representation="compressed",
        splits=1,
        mask="none",
        q=(17, 3),
        k=(1057, 513),
    ),
    dict(
        id="compressed-explicit-window",
        representation="compressed",
        splits=3,
        mask="window",
        q=(17, 3),
        k=(1057, 513),
        strided=True,
        used_k=(1025, 7),
    ),
    dict(
        id="compressed-auto-causal",
        representation="compressed",
        splits=0,
        mask="causal",
        q=(1, 1),
        k=(16384, 513),
    ),
    dict(
        id="prefill-single-causal",
        representation="prefill",
        splits=1,
        mask="causal",
        q=(17, 3),
        k=(1057, 513),
    ),
    dict(
        id="prefill-auto-window",
        representation="prefill",
        splits=0,
        mask="window",
        q=(17, 3),
        k=(1057, 513),
        strided=True,
        used_k=(1025, 7),
    ),
]
MODES = ["metadata"] + [
    f"{entry}_{r}" for entry in ("implicit", "generate", "reuse") for r in (1, 4)
]


def partition_rows(rows, raw, labels, representation, dtype, spec, *, layout="mixed"):
    if representation == "prefill":
        count = 3 if spec["splits"] == 1 else 4
        if len(rows) != len(labels) * count:
            raise ValueError("Unmatched mixed prefill metadata/initialization/numerical phases")
        result = []
        for i, label in enumerate(labels):
            group = rows[i * count : (i + 1) * count]
            producer, first = group[:2]
            if "ComputeMlaPrefillBsndMetadata" not in producer["Name"]:
                raise ValueError("Missing mixed prefill metadata phase")
            if producer["Device_id"] != first["Device_id"] or Decimal(
                producer["Start Time(us)"]
            ) + Decimal(producer["Duration(us)"]) > Decimal(first["Start Time(us)"]):
                raise ValueError("mixed prefill metadata overlaps or changes device")
            sample = partition_kernels(group[1:], [label], [spec], dtype)[0]
            sample["metadata_us"] = float(producer["Duration(us)"])
            sample["metadata_task"] = producer
            result.append(sample)
        return result
    automatic = spec["splits"] == 0
    prefix = f"mla_{'auto_' if automatic else ''}split_{layout}_"
    stages = [prefix + s + "_" for s in (["init"] if automatic else []) + ["partial", "merge"]]
    count = len(stages) + 1
    if len(raw) != len(labels) * count:
        raise ValueError("Unmatched mixed metadata/partial/merge phases")
    result = []
    for i, label in enumerate(labels):
        group = raw[i * count : (i + 1) * count]
        merged = production_sample(
            group,
            ("ComputeMlaAutoSplit" if automatic else "ComputeMlaSplit")
            + "".join(word.capitalize() for word in layout.split("_"))
            + "Metadata",
            stages,
        )
        if not all(stage + dtype in row["Op Name"] for stage, row in zip(stages, group[1:])):
            raise ValueError("Wrong mixed dtype")
        result.append(
            dict(
                **label,
                kernel_us=float(merged["Task Duration(us)"]),
                adaptation_us=0,
                metadata_us=float(group[0]["Task Duration(us)"]),
                tasks=group,
            )
        )
    return result


def kernel_samples(call, representation, dtype, spec, directory, *, partition=partition_rows):
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
    samples = partition(rows, raw, labels, representation, dtype, spec)
    (directory / "kernel-samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    return samples


def profile(
    representation,
    dtype,
    directory,
    *,
    check=False,
    cases=PERFORMANCE,
    make_case=make_case,
    metadata=metadata,
    invoke=invoke,
    compare=compare,
    partition=partition_rows,
    layout="mixed cache",
):
    """Share the fixed wall/profiler protocol with the BSND PA adapter."""
    directory.mkdir(parents=True, exist_ok=False)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    (directory / "npu-before.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )
    rows = []
    try:
        for spec in cases:
            if spec["representation"] != representation:
                continue
            c = make_case(dtype=dtype, **spec)
            record = metadata(c)
            compare(c, invoke(c, record), record)
            uncapped = dict(c, options=dict(c["options"], softcap=0))
            uncapped_record = metadata(uncapped)
            compare(uncapped, invoke(uncapped, entry="public"), metadata(uncapped))

            def call(mode, cap):
                case = c if cap else uncapped
                if mode == "metadata":
                    return metadata(case)
                entry, r = mode.rsplit("_", 1)
                selected = (
                    metadata(case)
                    if entry == "generate"
                    else (record if cap else uncapped_record)
                    if entry == "reuse"
                    else None
                )
                return [invoke(case, selected, entry="public") for _ in range(int(r))]

            for _ in range(5):
                for cap in (0, 15):
                    for mode in MODES:
                        if mode == "metadata" and cap == 0:
                            continue
                        result = call(mode, cap)
                        torch.npu.synchronize()
            subdir = directory / spec["id"]
            subdir.mkdir()
            if check:
                # Preflight exercises every fixed input and profiler phase parser.
                kernels = kernel_samples(
                    lambda: call("implicit_1", 15),
                    representation,
                    dtype,
                    spec,
                    subdir,
                    partition=partition,
                )
                rows.append(dict(spec=spec, kernel_samples=len(kernels)))
                continue
            samples = []
            for round_id in range(5):
                for iteration in range(20):
                    for cap in (0, 15) if round_id % 2 == 0 else (15, 0):
                        for mode in MODES if round_id % 2 == 0 else MODES[::-1]:
                            if mode == "metadata" and cap == 0:
                                continue
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
                lambda: call("implicit_1", 15),
                representation,
                dtype,
                spec,
                subdir,
                partition=partition,
            )
            segments = {}
            for mode in MODES:
                values = {
                    str(cap): statistics.median(
                        x["e2e_us"] for x in samples if x["cap"] == cap and x["mode"] == mode
                    )
                    for cap in ((15,) if mode == "metadata" else (0, 15))
                }
                if mode != "metadata":
                    values["overhead_ratio"] = values["15"] / values["0"]
                segments[mode] = values
            rows.append(
                dict(
                    spec=spec,
                    segments=segments,
                    kernel_us=statistics.median(x["kernel_us"] for x in kernels),
                    adaptation_us=statistics.median(x["adaptation_us"] for x in kernels),
                    metadata_us=statistics.median(x["metadata_us"] for x in kernels),
                )
            )
        report = dict(
            scope=f"Complete public {layout} implicit/generate/reuse calls at R=1/4; profiler reports implicit metadata and kernels separately. cap=0 is a migrated overhead reference with different semantics, never an equivalent CANN softcap baseline or acceptance ratio.",
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
