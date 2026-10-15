"""Pinned native/public BSND prefill, unchanged synchronized wall timing boundary."""

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
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.prefill_bsnd import make_case, metadata, invoke, compare
from tests.mla.support.mla_window_public_profile import WindowCannCall
from tests.mla.support.mla_prefill_bsnd_cases import PERFORMANCE_CASES
from tests.mla.support.mla_native_dispatch import sha256, e2e_applicable
from tests.mla.support.mla_public_performance import SEGMENTS, FUNC_SEGMENTS
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_prefill_bsnd_profile_rows import partition_kernels


def sample_kernels(call, specs, directory, dtype):
    import torch_npu

    profiler = torch_npu.profiler
    labels = []
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(wait=0, warmup=1, active=len(specs) * 5 * 10 * 2, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(str(directory / "profiler")),
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level0,
            export_type=profiler.ExportType.Text,
            data_simplification=False,
        ),
    ) as prof:
        result = call(0, "fa3")
        torch.npu.synchronize()
        prof.step()
        for index in range(len(specs)):
            for round_id in range(5):
                for iteration in range(10):
                    names = ("fa3", "native_cann")
                    for name in names if round_id % 2 == 0 else names[::-1]:
                        result = call(index, name)
                        torch.npu.synchronize()
                        labels.append(
                            dict(case=index, name=name, round=round_id, iteration=iteration)
                        )
                        prof.step()
    del result
    paths = list(directory.rglob("kernel_details.csv"))
    raw_paths = list(directory.rglob("op_summary_*.csv"))
    assert len(paths) == len(raw_paths) == 1
    with paths[0].open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Start Time(us)"]))
    with raw_paths[0].open() as f:
        raw = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Task Start Time(us)"]))
    assert len(rows) == len(raw)
    for row, source in zip(rows, raw):
        assert row["Name"] == source["Op Name"] and row["Device_id"] == source["Device_id"]
        assert Decimal(row["Start Time(us)"]) == Decimal(source["Task Start Time(us)"])
        assert float(row["Duration(us)"]) == float(source["Task Duration(us)"])
    samples = partition_kernels(rows, labels, specs, dtype)
    (directory / "kernel-samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    return samples


def run(dtype, directory, dispatch=None, check=False):
    directory.mkdir(parents=True, exist_ok=True)
    specs = PERFORMANCE_CASES[:1] if check else PERFORMANCE_CASES
    cases = [make_case(dtype, **s) for s in specs]
    calls = [
        WindowCannCall(
            c["t"],
            scale=c["options"]["softmax_scale"],
            kv_lengths=s.get("lengths"),
            spec=dict(window=c["options"]["window_size"], causal=c["options"]["causal"]),
        )
        for s, c in zip(specs, cases)
    ]
    if dispatch is None:
        accuracy = []
        for index, (c, call) in enumerate(zip(cases, calls)):
            record = metadata(c)
            result = invoke(c, record)
            refs = compare(c, result, record)
            print(f"MLA_STRIDE_NATIVE_BEGIN {dtype} {index}", flush=True)
            native = call()
            torch.npu.synchronize()
            print(f"MLA_STRIDE_NATIVE_END {dtype} {index}", flush=True)
            for i, name in enumerate(("out", "lse")):
                value = native[i].cpu()
                if i == 1:
                    value[value.isposinf()] = -torch.inf
                assert_fa_close(
                    value, refs[name + "_high"], refs[name + "_low"], name="native-prefill-" + name
                )
            accuracy.append(
                dict(spec=specs[index], record=record.cpu().tolist(), native=call.description())
            )
        (directory / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
        return
    proof = json.loads(dispatch.read_text())
    assert proof["dtype"] == dtype and len(proof["cases"]) == len(specs)
    assert all(sha256(p) == v for p, v in proof["artifacts"].items())
    assert json.loads(json.dumps(specs)) == proof["specs"]
    assert os.environ.get("ASCEND_GLOBAL_LOG_LEVEL") == "3"
    records = [metadata(c) for c in cases]

    def call(i, name):
        c = cases[i]
        if name.startswith("native_cann"):
            return [calls[i]() for _ in range(4 if name.endswith("_4") else 1)]
        if name.startswith("func"):
            return [
                api.flash_attn_func(**c["t"], return_attn_probs=True, **c["options"])
                for _ in range(4 if name.endswith("_4") else 1)
            ]
        if name == "generate":
            return metadata(c)
        rec = (
            metadata(c)
            if name.startswith("generate_")
            else None
            if name.startswith("implicit")
            else records[i]
        )
        return [invoke(c, rec) for _ in range(4 if name.endswith("_4") else 1)]

    def modes(i):
        return [
            "fa3",
            "native_cann",
            "generate",
            "generate_1",
            "generate_4",
            "native_cann_4",
            "reuse_4",
            "implicit",
            "implicit_4",
        ] + ([] if "lengths" in specs[i] else ["func", "func_4"])

    (directory / "npu-before.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )
    samples = []
    for i in range(len(cases)):
        for _ in range(5):
            for name in modes(i):
                result = call(i, name)
                torch.npu.synchronize()
        for round_ in range(5):
            for repeat in range(20):
                for name in modes(i) if round_ % 2 == 0 else modes(i)[::-1]:
                    start = time.perf_counter_ns()
                    result = call(i, name)
                    torch.npu.synchronize()
                    samples.append(
                        dict(
                            case=i,
                            name=name,
                            round=round_,
                            iteration=repeat,
                            e2e_us=(time.perf_counter_ns() - start) / 1000,
                        )
                    )
    (directory / "wall-samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    kernels = sample_kernels(call, specs, directory, dtype)
    rows = []
    for i, c in enumerate(cases):
        native = proof["cases"][i]
        row = dict(
            shape=c["shape"],
            scale=c["options"]["softmax_scale"],
            lengths=specs[i].get("lengths"),
            stride_profile=None,
            window=c["options"]["window_size"],
            causal=c["options"]["causal"],
            num_splits=c["options"]["num_splits"],
            native_template=native["native_template"],
            native_fd=native["native_fd"],
            native_call=calls[i].description(),
        )
        row["e2e_applicable"] = e2e_applicable(
            row["native_template"],
            row["native_fd"],
            representation="prefill",
            include_native_fd=row["num_splits"] == 0,
        )
        row["kernel_us"] = {
            name: statistics.median(
                x["kernel_us"] for x in kernels if x["case"] == i and x["name"] == name
            )
            for name in ("fa3", "native_cann")
        }
        row["adaptation_us"] = {
            name: statistics.median(
                x["adaptation_us"] for x in kernels if x["case"] == i and x["name"] == name
            )
            for name in ("fa3", "native_cann")
        }
        for name in modes(i):
            values = [x for x in samples if x["case"] == i and x["name"] == name]
            row[name] = dict(
                median_e2e_us=statistics.median(x["e2e_us"] for x in values),
                round_mean_e2e_us=[
                    statistics.mean(x["e2e_us"] for x in values if x["round"] == r)
                    for r in range(5)
                ],
            )
        for name, (a, b) in (SEGMENTS | (FUNC_SEGMENTS if row["lengths"] is None else {})).items():
            row[name] = row[a]["median_e2e_us"] / row[b]["median_e2e_us"]
        rows.append(row)
    (directory / "profile.json").write_text(
        json.dumps(
            dict(
                dtype=dtype,
                representation="prefill",
                causal=False,
                window=True,
                warmup=5,
                rounds=5,
                wall_iterations=20,
                profiler_iterations=10,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                device=torch.npu.get_device_name(0),
                metadata_lifecycle=dict(complete_segments=[1, 4]),
                results=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    (directory / "npu-after.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dispatch", type=Path)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    run(a.dtype, a.output, a.dispatch, a.check)
