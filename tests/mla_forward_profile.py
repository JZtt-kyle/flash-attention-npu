"""Public FA3 dense E2E diagnostics, including implicit MLA metadata generation."""

import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.common.mla_ref import mla_reference_pair
from tests.mla_cann_baseline import (
    CannCall,
    Case,
    check_vendor_configuration,
    environment,
    make_inputs,
)
from tests.mla_profile_cases import CASES, KV_CASES, LONG_KV_CASES, S1G_CASES


def profile(directory, vendor, dtype_name="fp16", metadata_reuse=False, cases=CASES):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    timed_cpu_threads = torch.get_num_threads()
    reference_cpu_threads = int(os.environ.get("MLA_FORWARD_REFERENCE_THREADS", timed_cpu_threads))
    assert reference_cpu_threads > 0
    vendor = check_vendor_configuration(vendor)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    inputs, native, lengths, metadata = [], [], [], []

    def generate(index):
        q, k = inputs[index]["q"], inputs[index]["k"]
        return api.get_scheduler_metadata(
            q.shape[0],
            q.shape[1],
            k.shape[1],
            q.shape[2],
            1,
            64,
            lengths[index],
            qkv_dtype=dtype,
            headdim_v=512,
            num_splits=1,
        )

    def cached(index, record):
        tensors = inputs[index]
        return api.flash_attn_with_kvcache(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            qv=tensors["qv"],
            cache_seqlens=lengths[index],
            scheduler_metadata=record,
            softmax_scale=cases[index][1],
            num_splits=1,
            return_softmax_lse=True,
        )

    accuracy = []
    for shape, scale in cases:
        batch, sq, heads, sk = shape
        cpu = make_inputs(
            Case("public", "compressed", batch, sq, sk, heads, 1), dtype, seed=20260929
        )
        device = {name: tensor.npu() for name, tensor in cpu.items()}
        inputs.append(device)
        native.append(CannCall(device, scale=scale))
        if metadata_reuse:
            lengths.append(torch.full((batch,), sk, dtype=torch.int32, device=device["q"].device))
            metadata.append(generate(len(inputs) - 1))
            actual = cached(len(inputs) - 1, metadata[-1])
        else:
            actual = api.flash_attn_func(
                **device, softmax_scale=scale, num_splits=1, return_attn_probs=True
            )
        source = native[-1]()
        # The long-KV CPU oracle is outside every timed region. Restore the
        # caller's thread count even on failure, before any warmup or sampling.
        try:
            torch.set_num_threads(reference_cpu_threads)
            refs = mla_reference_pair(**cpu, softmax_scale=scale)
        finally:
            torch.set_num_threads(timed_cpu_threads)
        for i, name in enumerate(("out", "lse")):
            if sk <= 512:
                torch.testing.assert_close(actual[i].cpu(), source[i].cpu(), rtol=0, atol=0)
            assert_fa_close(actual[i].cpu(), refs[i], refs[i + 2], name=name)
            assert_fa_close(source[i].cpu(), refs[i], refs[i + 2], name=name)
        accuracy.append(
            dict(
                shape=shape,
                scale=scale,
                source_equal=all(torch.equal(actual[i].cpu(), source[i].cpu()) for i in range(2)),
                fa3_max_error=[float((actual[i].cpu() - refs[i]).abs().max()) for i in range(2)],
                cann_max_error=[float((source[i].cpu() - refs[i]).abs().max()) for i in range(2)],
            )
        )
    (directory / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    source_root = Path(
        os.environ.get(
            "MLA_CANN_SOURCE_ROOT", Path(__file__).resolve().parents[2] / "ops-transformer"
        )
    )
    env = environment(source_root, vendor)
    extension = Path(api.flash_attn_npu_3.__file__).resolve()
    env["fa3_extension"] = dict(
        path=str(extension), sha256=hashlib.sha256(extension.read_bytes()).hexdigest()
    )
    (directory / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    def snapshot(label):
        (directory / label).write_text(subprocess.check_output(["npu-smi", "info"], text=True))

    def call(index, name):
        if name == "native_cann":
            return native[index]()
        if metadata_reuse:
            if name == "generate":
                return generate(index)
            if name == "native_cann_4":
                return [native[index]() for _ in range(4)]
            if name == "implicit":
                return cached(index, None)
            if name in ("generate_1", "generate_4"):
                record = generate(index)
                return [cached(index, record) for _ in range(1 if name == "generate_1" else 4)]
            return cached(index, metadata[index])
        return api.flash_attn_func(
            **inputs[index], softmax_scale=cases[index][1], num_splits=1, return_attn_probs=True
        )

    names = ("fa3", "native_cann")
    wall_names = (
        names + ("generate", "implicit", "generate_1", "generate_4", "native_cann_4")
        if metadata_reuse
        else names
    )
    snapshot("npu-before.log")
    assert torch.get_num_threads() == timed_cpu_threads
    for index in range(len(cases)):
        for _ in range(5):
            for name in wall_names:
                result = call(index, name)
                torch.npu.synchronize()
    wall = []
    for index in range(len(cases)):
        for r in range(5):
            for i in range(20):
                for name in wall_names if r % 2 == 0 else wall_names[::-1]:
                    start = time.perf_counter_ns()
                    result = call(index, name)
                    torch.npu.synchronize()
                    wall.append(
                        dict(
                            case=index,
                            name=name,
                            round=r,
                            iteration=i,
                            e2e_us=(time.perf_counter_ns() - start) / 1000,
                        )
                    )
    (directory / "wall-samples.json").write_text(json.dumps(wall, indent=2) + "\n")

    labels = []
    profiler = torch_npu.profiler
    trace = directory / "profiler"
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
        result = call(0, "fa3")
        torch.npu.synchronize()
        prof.step()
        for index in range(len(cases)):
            for r in range(5):
                for i in range(10):
                    for name in names if r % 2 == 0 else names[::-1]:
                        result = call(index, name)
                        torch.npu.synchronize()
                        labels.append(dict(case=index, name=name, round=r, iteration=i))
                        prof.step()
    del result
    (directory / "profile-labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    csvs, raw_csvs = list(trace.rglob("kernel_details.csv")), list(trace.rglob("op_summary_*.csv"))
    assert len(csvs) == len(raw_csvs) == 1
    with csvs[0].open() as f:
        rows = sorted(csv.DictReader(f), key=lambda row: Decimal(row["Start Time(us)"].strip()))
    with raw_csvs[0].open() as f:
        raw = sorted(csv.DictReader(f), key=lambda row: Decimal(row["Task Start Time(us)"].strip()))
    assert len(rows) == len(raw), (len(rows), len(raw))
    for row, source in zip(rows, raw):
        assert row["Name"] == source["Op Name"] and row["Device_id"] == source["Device_id"]
        assert Decimal(row["Start Time(us)"].strip()) == Decimal(
            source["Task Start Time(us)"].strip()
        )
        assert float(row["Duration(us)"]) == float(source["Task Duration(us)"])
    # Some profiler versions include AICPU tasks here, others export only
    # attention kernels. Never fold producer duration into the attention
    # comparison, and explicitly report whether it was captured.
    metadata_rows = [row for row in rows if "ComputeMlaDenseMetadata" in row["Name"]]
    if metadata_reuse:
        assert not metadata_rows, "explicit reuse unexpectedly regenerated metadata"
    rows = [row for row in rows if "ComputeMlaDenseMetadata" not in row["Name"]]
    assert len(rows) == len(labels), (len(rows), len(labels))
    fa3_count = sum(label["name"] == "fa3" for label in labels)
    assert len(metadata_rows) in (0, fa3_count), (len(metadata_rows), fa3_count)
    producers = iter(metadata_rows)
    previous_end = Decimal(0)
    for row, label in zip(rows, labels):
        expected_name = (
            f"mla_dense_{dtype_name}" if label["name"] == "fa3" else "FusedInferAttentionScore"
        )
        assert expected_name in row["Name"]
        label.update(kernel_us=float(row["Duration(us)"]), kernel_name=row["Name"])
        if metadata_rows and label["name"] == "fa3":
            producer = next(producers)
            begin = Decimal(producer["Start Time(us)"].strip())
            end = begin + Decimal(producer["Duration(us)"].strip())
            assert previous_end <= begin < end <= Decimal(row["Start Time(us)"].strip())
            label["metadata_kernel_us"] = float(producer["Duration(us)"])
        previous_end = Decimal(row["Start Time(us)"].strip()) + Decimal(row["Duration(us)"].strip())
    results = []
    for index, (shape, scale) in enumerate(cases):
        item = dict(shape=shape, scale=scale, native_call=native[index].description())
        for name in wall_names:
            kernels = [x for x in labels if x["case"] == index and x["name"] == name]
            walls = [x for x in wall if x["case"] == index and x["name"] == name]
            item[name] = dict(
                median_e2e_us=statistics.median(x["e2e_us"] for x in walls),
                round_mean_e2e_us=[
                    statistics.mean(x["e2e_us"] for x in walls if x["round"] == r) for r in range(5)
                ],
            )
            if kernels:
                item[name].update(
                    median_kernel_us=statistics.median(x["kernel_us"] for x in kernels),
                    round_mean_kernel_us=[
                        statistics.mean(x["kernel_us"] for x in kernels if x["round"] == r)
                        for r in range(5)
                    ],
                )
        item["e2e_ratio"] = item["fa3"]["median_e2e_us"] / item["native_cann"]["median_e2e_us"]
        if metadata_reuse:
            item["generate_1_ratio"] = (
                item["generate_1"]["median_e2e_us"] / item["native_cann"]["median_e2e_us"]
            )
            item["generate_4_ratio"] = (
                item["generate_4"]["median_e2e_us"] / item["native_cann_4"]["median_e2e_us"]
            )
        results.append(item)
    snapshot("npu-after.log")
    report = dict(
        dtype=dtype_name,
        scope=(
            "Public read-only dense kvcache metadata reuse vs fixed native CANN"
            if metadata_reuse
            else "Public flash_attn_func with implicit AICPU metadata generation vs fixed native CANN; diagnostics, not explicit metadata reuse acceptance"
        ),
        metadata_lifecycle=(
            dict(
                reuse_forward="precomputed once before measurement",
                complete_segments=[1, 4],
                complete_segment="one public metadata generation plus R forwards; CANN performs the same R forwards; R fixed before collection",
            )
            if metadata_reuse
            else "implicit generation for each forward"
        ),
        pid=os.getpid(),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        seed=20260929,
        warmup=5,
        rounds=5,
        wall_iterations=20,
        profiler_iterations=10,
        reference_cpu_threads=reference_cpu_threads,
        timed_cpu_threads=timed_cpu_threads,
        timing="public call through device completion, including validation, scheduling, output/workspace allocation and launch; input H2D excluded on both paths",
        kernel_csv=str(csvs[0]),
        raw_kernel_csv=str(raw_csvs[0]),
        metadata_kernel_rows=len(metadata_rows),
        results=results,
        kernel_samples=labels,
    )
    (directory / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    profile(
        os.environ["MLA_FORWARD_ARTIFACT_DIR"],
        os.environ["MLA_CANN_VENDOR"],
        os.environ.get("MLA_FORWARD_DTYPE", "fp16"),
        os.environ.get("MLA_FORWARD_METADATA_REUSE") == "1",
        CASES
        + (S1G_CASES if os.environ.get("MLA_FORWARD_S1G_MULTIBLOCK") == "1" else [])
        + (KV_CASES if os.environ.get("MLA_FORWARD_KV_MULTIBLOCK") == "1" else [])
        + (LONG_KV_CASES if os.environ.get("MLA_FORWARD_LONG_KV") == "1" else []),
    )
