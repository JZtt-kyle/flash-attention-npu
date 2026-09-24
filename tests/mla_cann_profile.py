"""Strict parsing of the serialized dense MLA profiler experiment.

Each synchronized call ends with one FIA kernel, preceded by its Slice copies.
No transfers, references, or other device work run inside the active region.
The torch_npu 2.7.1 CSV Step Id can lag the device calls, so attribution uses
chronological kernel order and validates every call's complete kernel sequence.
"""

from collections import Counter
import csv
from decimal import Decimal
import math
from pathlib import Path
import statistics


def parse_kernel_details(path, *, conversion_counts, iterations, rounds):
    """Reject missing/extra calls and unknown kernels instead of undercounting."""
    if not conversion_counts or min(iterations, rounds) <= 0:
        raise ValueError("At least one case and positive iteration/round counts are required")
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Name", "Device_id", "Start Time(us)", "Duration(us)"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Profiler CSV is missing kernel timing columns")
        rows = list(reader)
    if not rows or len({row["Device_id"] for row in rows}) != 1:
        raise ValueError("Profiler must contain kernels from exactly one device")
    for row in rows:
        start = Decimal(row["Start Time(us)"].strip())
        duration = float(row["Duration(us)"])
        if not start.is_finite() or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Profiler contains invalid kernel timing")
    rows.sort(key=lambda row: Decimal(row["Start Time(us)"].strip()))
    calls, pending = [], []
    for row in rows:
        name = row["Name"]
        if name.endswith("_Slice") and name.startswith("aclnnFusedInferAttentionScore"):
            pending.append(row)
        elif name.endswith("_FusedInferAttentionScore") and name.startswith(
            "aclnnFusedInferAttentionScore"
        ):
            calls.append((row, pending))
            pending = []
        else:
            raise ValueError(f"Unexpected kernel in serialized FIA capture: {name}")
    count = iterations * rounds
    if pending or len(calls) != len(conversion_counts) * count:
        raise ValueError("Profiler kernel count does not match the requested calls")
    results = []
    for case_index, expected_copies in enumerate(conversion_counts):
        samples, names = [], Counter()
        for kernel, copies in calls[case_index * count : (case_index + 1) * count]:
            if len(copies) != expected_copies:
                raise ValueError("Profiler conversion count does not match the input views")
            kernel_us = float(kernel["Duration(us)"])
            copy_us = [float(row["Duration(us)"]) for row in copies]
            samples.append(dict(fia_kernel_us=kernel_us, conversion_kernel_us=copy_us))
            names.update(row["Name"] for row in [*copies, kernel])
        round_means = [
            dict(
                fia_kernel_us=statistics.mean(row["fia_kernel_us"] for row in group),
                conversion_kernel_us=statistics.mean(
                    sum(row["conversion_kernel_us"]) for row in group
                ),
            )
            for offset in range(0, count, iterations)
            for group in [samples[offset : offset + iterations]]
        ]
        results.append(
            dict(
                samples=samples,
                round_means=round_means,
                median_fia_kernel_us=statistics.median(row["fia_kernel_us"] for row in samples),
                median_conversion_kernel_us=statistics.median(
                    sum(row["conversion_kernel_us"]) for row in samples
                ),
                median_device_kernel_sum_us=statistics.median(
                    row["fia_kernel_us"] + sum(row["conversion_kernel_us"]) for row in samples
                ),
                kernel_counts=dict(names),
                device_id=rows[0]["Device_id"],
            )
        )
    return results


def profile_calls(calls, directory, *, iterations, rounds):
    """Capture only prepared calls; ordinary wall timing must run separately."""
    import torch
    import torch_npu

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    count = iterations * rounds
    conversion_counts = [
        sum(
            tensor is not None and not tensor.is_contiguous()
            for tensor in (
                call.query,
                call.key,
                call.value,
                call.kwargs["query_rope"],
                call.kwargs["key_rope"],
            )
        )
        for call in calls
    ]
    p = torch_npu.profiler
    with p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        schedule=p.schedule(wait=0, warmup=1, active=len(calls) * count, repeat=1),
        on_trace_ready=p.tensorboard_trace_handler(str(directory)),
        experimental_config=p._ExperimentalConfig(
            profiler_level=p.ProfilerLevel.Level0,
            export_type=p.ExportType.Text,
            data_simplification=False,
        ),
    ) as prof:
        calls[0]()
        torch.npu.synchronize()
        prof.step()
        for call in calls:
            for _ in range(count):
                call()
                torch.npu.synchronize()
                prof.step()
    csvs = list(directory.rglob("kernel_details.csv"))
    if len(csvs) != 1:
        raise RuntimeError("Profiler must export exactly one kernel_details.csv")
    return csvs[0], parse_kernel_details(
        csvs[0], conversion_counts=conversion_counts, iterations=iterations, rounds=rounds
    )
