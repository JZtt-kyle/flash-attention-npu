"""Mixed public metadata E2E, including native CANN's necessary KV packing."""

import argparse
import csv
from decimal import Decimal
import itertools
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
from tests.mla_cann_baseline import check_vendor_configuration, environment, file_info
from tests.mla_dense_build import ROOT, sha256
from tests.mla_varlen_profile import CannTndCall
from tests.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.test_flash_attn_npu_v3_mla_mixed_public import metadata, public
from tests.test_flash_attn_npu_v3_mla_tnd_bsnd_attention import references
from tests.test_flash_attn_npu_v3_mla_tnd_schedule import build_oracle, evaluate, tnd_case


# Fixed before sampling. Retain the twelve mixed component geometries, with
# Hq capped at the pinned native host limit of 128 and real platform cores.
PERFORMANCE_CASES = [
    ([3, 0, 1], [17, 0, 33], 64, 128),
    ([1, 7, 2], [1, 513, 31], 1025, 16),
    ([0, 3, 1, 0, 7], [33, 511, 0, 513, 1025], 2048, 128),
    ([1, 2, 3], [16384, 8193, 17], 16384, 8),
    ([1, 1, 1], [17, 129, 257], 513, 128),
    ([2] * 10, [511] + [0] * 9, 511, 128),
    ([3, 0, 1, 7, 0], [17, 65, 513, 0, 1], 1025, 128),
    ([3, 0] * 21, [0, 513] * 21, 513, 128),
    ([257, 1, 0, 3], [33, 513, 17, 1], 513, 1),
    ([0, 1, 0, 0], [0, 0, 0, 1], 17, 1),
    ([3, 0, 1], [0, 0, 0], 64, 128),
    ([2] * 10, [511] * 10, 511, 128),
]


class CannMixedCall:
    """Pinned native TND call; physical BSND KV is gathered on every call.

    FIA GetKvLayout requires matching Q/KV for continuous storage. Preparation
    only creates immutable views/indices and host length lists. The two device
    gathers, allocation and native call are all within E2E. No source host or
    kernel is modified and no independent probe substitutes for native FIA.
    """

    def __init__(self, inputs, qs, ks, scale):
        self.inputs = inputs
        capacity = inputs["k"].shape[1]
        positions = [b * capacity + j for b, length in enumerate(ks) for j in range(length)]
        self.index = torch.tensor(positions, dtype=torch.int32, device=inputs["q"].device)
        self.batch_index = torch.tensor(
            [b for b, length in enumerate(ks) for _ in range(length)],
            dtype=torch.int64,
            device=inputs["q"].device,
        )
        self.token_index = torch.tensor(
            [j for length in ks for j in range(length)],
            dtype=torch.int64,
            device=inputs["q"].device,
        )
        self.cache = {}
        for name in ("k", "v"):
            try:
                self.cache[name] = inputs[name].view(-1, 1, inputs[name].shape[-1])
            except RuntimeError:
                # Only create zero-copy views before timing. Arbitrary BSND
                # strides need a two-axis gather on each native invocation.
                self.cache[name] = None
        self.native = CannTndCall(inputs, qs, ks, scale)

    def pack(self):
        return {
            n: tensor.index_select(0, self.index)
            if tensor is not None
            else self.inputs[n][self.batch_index, self.token_index]
            for n, tensor in self.cache.items()
        }

    def __call__(self):
        packed = self.pack()
        self.native.inputs = dict(self.inputs, **packed)
        self.native.kwargs["key_rope"] = packed["k"]
        return self.native()


def prepare(dtype, platform, output):
    oracle_dir = output / "source-schedule"
    oracle_dir.mkdir()
    oracle = build_oracle(oracle_dir)
    shapes = [
        tnd_case(
            q,
            k,
            heads=h,
            aic=platform["aic"],
            ratio=2,
            libapi=platform["libapi"],
            **dict(zip(("l1", "ub", "l0a", "l0b", "l0c"), platform["memory"])),
        )
        for q, k, _, h in PERFORMANCE_CASES
    ]
    native_schedules = evaluate(oracle, shapes, native=True)
    (output / "dispatch.json").write_text(
        json.dumps(dict(cases=PERFORMANCE_CASES, inputs=shapes, native=native_schedules), indent=2)
        + "\n"
    )
    args, natives, records, accuracy = [], [], [], []
    for index, (qs, ks, capacity, heads) in enumerate(PERFORMANCE_CASES):
        gen = torch.Generator().manual_seed(20261010)
        cpu = {
            n: torch.randn(total, h, d, generator=gen).to(dtype)
            for n, total, h, d in (
                ("q", sum(qs), heads, 64),
                ("qv", sum(qs), heads, 512),
                ("k", len(qs) * capacity, 1, 64),
                ("v", len(qs) * capacity, 1, 512),
            )
        }
        # Detect padding reads in both native packing and the mixed kernel.
        for b, length in enumerate(ks):
            for n in ("k", "v"):
                cpu[n][b * capacity + length : (b + 1) * capacity] = torch.nan
        device = {n: t.npu() for n, t in cpu.items()}
        for n in ("k", "v"):
            device[n] = device[n].view(len(qs), capacity, 1, device[n].shape[-1])
        item = dict(
            **device,
            cu_seqlens_q=torch.tensor(
                list(itertools.accumulate(qs, initial=0)), dtype=torch.int32, device="npu"
            ),
            cache_seqlens=torch.tensor(ks, dtype=torch.int32, device="npu"),
            max_seqlen_q=max(qs) + 3,
        )
        args.append(item)
        record = metadata(item)
        records.append(record)
        natives.append(CannMixedCall(device, qs, ks, 576**-0.5))
        actual = public(dict(item, scheduler_metadata=record))
        source = natives[-1]()
        refs = references(dict(spec=(qs, ks, capacity, heads, None), inputs=cpu, scale=576**-0.5))
        nonempty = torch.tensor([k > 0 for q, k in zip(qs, ks) for _ in range(q)])
        for i in range(2):
            a, s = actual[i].cpu(), source[i].cpu()
            assert_fa_close(a, refs[i], refs[i + 2], name="mixed-fa3")
            valid = nonempty if i == 0 else (slice(None), nonempty)
            if nonempty.any():
                assert_fa_close(s[valid], refs[i][valid], refs[i + 2][valid], name="mixed-native")
            if i == 0:
                assert (s[~nonempty] == 0).all()
            else:
                assert s[:, ~nonempty].isposinf().all()
        accuracy.append(dict(case=index, native_fd=bool(native_schedules[index][5]), passed=True))
    (output / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    return args, natives, records, native_schedules


def prepare_stride(platform, output, probe_dir, producer_manifest):
    from tests import test_flash_attn_npu_v3_mla_mixed_stride_production as cases_api
    from tests.test_flash_attn_npu_v3_mla_tnd_stride import configure_producer, load_attention
    from tests.test_flash_attn_npu_v3_mla_used_attention import (
        check_math,
        effective,
        references,
        valid_outputs,
    )

    probes = load_attention(probe_dir)
    manifest = json.loads(Path(producer_manifest).read_text())
    assert sha256(Path(manifest["library"])) == manifest["library_sha256"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    configs = cases_api.PUBLIC_CASES
    oracle_dir = output / "source-schedule"
    oracle_dir.mkdir()
    oracle = build_oracle(oracle_dir)
    shapes = [
        tnd_case(
            *effective(c["item"]),
            heads=c["item"]["heads"],
            aic=platform["aic"],
            ratio=2,
            libapi=platform["libapi"],
            **dict(zip(("l1", "ub", "l0a", "l0b", "l0c"), platform["memory"])),
        )
        for c in configs
    ]
    schedules = evaluate(oracle, shapes, native=True)
    (output / "dispatch.json").write_text(
        json.dumps(dict(cases=configs, inputs=shapes, native=schedules), indent=2) + "\n"
    )
    args, natives, records, accuracy, specs = [], [], [], [], []
    with load_probe(manifest["library"], output / "producer") as producer:
        configure_producer(producer)
        for index, config in enumerate(configs):
            case = cases_api.production_case(probes, producer, config)
            item = case["item"]
            qs, ks = effective(item)
            specs.append((qs, ks, item["sk"], item["heads"]))
            arg = cases_api.public_args(cases_api.forward_args(case))
            args.append(arg)
            record = metadata(arg)
            records.append(record)
            native = CannMixedCall(case["devices"]["migrated"], qs, ks, case["scale"])
            natives.append(native)
            assert all(native.inputs[n] is arg[n] for n in ("q", "qv", "k", "v"))
            cases_api.launch(probes, case, "oracle")
            errors = cases_api.check_result(
                case, public(dict(arg, scheduler_metadata=record)), source=True
            )
            source = valid_outputs(case, [t.cpu() for t in native()])
            refs = references(case)
            nonempty = torch.tensor(
                [k > 0 for q, k in zip(qs, ks) for _ in range(q)], dtype=torch.bool
            )
            assert (source[0][~nonempty] == 0).all() and source[1][:, ~nonempty].isposinf().all()
            native_errors = check_math(
                [source[0][nonempty], source[1][:, nonempty]],
                [r[nonempty] if i % 2 == 0 else r[:, nonempty] for i, r in enumerate(refs)],
            )
            accuracy.append(
                dict(
                    case=index,
                    errors=errors,
                    native_errors=native_errors,
                    native_fd=bool(schedules[index][5]),
                )
            )
    (output / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    return args, natives, records, schedules, specs


def profile(
    output,
    vendor,
    platform_path,
    dtype_name,
    dispatch_only=False,
    probe_dir=None,
    producer_manifest=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    vendor = check_vendor_configuration(vendor)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    platform = json.loads(Path(platform_path).read_text())
    torch.set_num_threads(8)
    if probe_dir is None:
        args, natives, records, schedules = prepare(dtype, platform, output)
        specs = PERFORMANCE_CASES
    else:
        assert producer_manifest is not None
        args, natives, records, schedules, specs = prepare_stride(
            platform, output, probe_dir, producer_manifest
        )
        assert all(a["q"].dtype == dtype for a in args)
    if dispatch_only:
        for index, native in enumerate(natives):
            os.write(1, f"MLA_MIXED_NATIVE_BEGIN {dtype_name} {index}\n".encode())
            native()
            torch.npu.synchronize()
            os.write(1, f"MLA_MIXED_NATIVE_END {dtype_name} {index}\n".encode())
        return
    env = environment(Path(__file__).resolve().parents[2] / "ops-transformer", vendor)
    env.update(
        fa3_extension=file_info(api.flash_attn_npu_3.__file__), profile_source=file_info(__file__)
    )
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    torch.set_num_threads(1)

    def call(index, name, repeats=1):
        if name == "generation":
            return metadata(args[index])
        if name == "conversion":
            return natives[index].pack()
        record = metadata(args[index]) if name == "generated" else records[index]
        if name == "cann":
            function = natives[index]
        else:
            kwargs = dict(args[index], scheduler_metadata=None if name == "implicit" else record)

            def function():
                return public(kwargs)

        return [function() for _ in range(repeats)]

    modes = ("reuse", "generated", "implicit", "cann")
    configurations = [(name, r) for r in (1, 4) for name in modes] + [
        ("generation", 1),
        ("conversion", 1),
    ]

    def snapshot(name):
        (output / name).write_text(subprocess.check_output(["npu-smi", "info"], text=True))

    snapshot("npu-before.log")
    for index in range(len(args)):
        for _ in range(5):
            for name, repeats in configurations:
                result = call(index, name, repeats)
                torch.npu.synchronize()
    wall = []
    for index in range(len(args)):
        for round_id in range(5):
            for iteration in range(20):
                for name, repeats in configurations if round_id % 2 == 0 else configurations[::-1]:
                    start = time.perf_counter_ns()
                    result = call(index, name, repeats)
                    torch.npu.synchronize()
                    wall.append(
                        dict(
                            case=index,
                            name=name,
                            repeats=repeats,
                            round=round_id,
                            iteration=iteration,
                            e2e_us=(time.perf_counter_ns() - start) / 1000,
                        )
                    )
    (output / "wall.json").write_text(json.dumps(wall, indent=2) + "\n")
    labels = []
    p = torch_npu.profiler
    with p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        schedule=p.schedule(wait=0, warmup=1, active=len(args) * 5 * 10 * 3, repeat=1),
        on_trace_ready=p.tensorboard_trace_handler(str(output / "profiler")),
        experimental_config=p._ExperimentalConfig(
            profiler_level=p.ProfilerLevel.Level0,
            export_type=p.ExportType.Text,
            data_simplification=False,
        ),
    ) as prof:
        result = call(0, "reuse")
        torch.npu.synchronize()
        prof.step()
        for index in range(len(args)):
            for round_id in range(5):
                for iteration in range(10):
                    for name in (
                        ("reuse", "generation", "cann")
                        if round_id % 2 == 0
                        else ("cann", "generation", "reuse")
                    ):
                        result = call(index, name)
                        torch.npu.synchronize()
                        labels.append(
                            dict(
                                case=index,
                                name=name,
                                round=round_id,
                                iteration=iteration,
                                step=len(labels) + 1,
                            )
                        )
                        prof.step()
    del result
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    csvs, raws = list(output.rglob("kernel_details.csv")), list(output.rglob("op_summary_*.csv"))
    assert len(csvs) == len(raws) == 1
    with csvs[0].open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Start Time(us)"].strip()))
    with raws[0].open() as f:
        raw = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
    assert len(rows) == len(raw)
    for r, s in zip(rows, raw):
        assert r["Name"] == s["Op Name"] and r["Device_id"] == s["Device_id"]
        assert Decimal(r["Start Time(us)"].strip()) == Decimal(s["Task Start Time(us)"].strip())
        assert float(r["Duration(us)"]) == float(s["Task Duration(us)"])
    cursor = 0
    native_patterns = {}
    for label in labels:
        # torch_npu may leave Step Id empty for an external launcher after an
        # AICPU step. Every call above completes before the next starts, so
        # chronological order is unambiguous; verify the full kernel sequence.
        if label["name"] == "cann":
            index = label["case"]
            if index not in native_patterns:
                assert label["round"] == label["iteration"] == 0
                end = next(
                    i
                    for i in range(cursor, len(rows))
                    if "mla_tnd_bsnd_" in rows[i]["Name"] or "ComputeMla" in rows[i]["Name"]
                )
                native_patterns[index] = [r["Name"] for r in rows[cursor:end]]
            pattern = native_patterns[index]
            count = len(pattern)
        else:
            count = 1
        samples = rows[cursor : cursor + count]
        assert len(samples) == count, label
        cursor += count
        expected = {
            "reuse": "mla_tnd_bsnd_" + dtype_name,
            "generation": "ComputeMlaTndBsndMetadataFromPrefix",
            "cann": "FusedInferAttentionScore",
        }[label["name"]]
        # The pinned native global-empty branch has its own FIA kernel/key.
        if label["name"] == "cann":
            assert [r["Name"] for r in samples] == pattern
            attention = [r for r in samples if r["Name"].endswith("_FusedInferAttentionScore")]
            assert len(attention) == 1, (label, samples)
            kernel = attention[0]
        else:
            assert expected in samples[0]["Name"], (label, samples)
            kernel = samples[0]
        assert all(r["Step Id"] in ("", str(label["step"])) for r in samples), (label, samples)
        label.update(
            kernel_us=float(kernel["Duration(us)"]),
            kernel_names=[kernel["Name"]],
            other_kernel_us=sum(float(r["Duration(us)"]) for r in samples if r is not kernel),
            reported_step_ids=[r["Step Id"] for r in samples],
            tasks=samples,
        )
    assert cursor == len(rows), (cursor, len(rows))
    summary = []
    for index, spec in enumerate(specs):
        row = dict(
            case=index,
            spec=spec,
            native_fd=bool(schedules[index][5]),
            native_empty=sum(spec[1]) == 0,
        )
        row["input_views"] = {
            n: dict(
                shape=list(t.shape),
                stride=list(t.stride()),
                offset=t.storage_offset(),
                storage_bytes=t.untyped_storage().size(),
                data_ptr=t.data_ptr(),
            )
            for n, t in natives[index].inputs.items()
        }
        assert all(natives[index].inputs[n] is args[index][n] for n in ("q", "qv", "k", "v"))
        for repeats in (1, 4):
            values = {
                name: statistics.median(
                    r["e2e_us"]
                    for r in wall
                    if r["case"] == index and r["name"] == name and r["repeats"] == repeats
                )
                for name in modes
            }
            row["r" + str(repeats)] = dict(
                **values,
                ratios={name: values[name] / values["cann"] for name in modes if name != "cann"},
            )
        row["diagnostic_us"] = {
            name: statistics.median(
                r["e2e_us"] for r in wall if r["case"] == index and r["name"] == name
            )
            for name in ("generation", "conversion")
        }
        row["kernel_us"] = {
            name: statistics.median(
                r["kernel_us"] for r in labels if r["case"] == index and r["name"] == name
            )
            for name in ("reuse", "generation", "cann")
        }
        row["native_conversion_device_us"] = statistics.median(
            r["other_kernel_us"] for r in labels if r["case"] == index and r["name"] == "cann"
        )
        summary.append(row)
    snapshot("npu-after.log")
    report = dict(
        scope="Public mixed FA3 versus pinned native TND with per-call physical BSND KV packing. R=1/4 includes O/LSE completion; generated includes one public metadata generation. Native FD and empty paths identified separately.",
        dtype=dtype_name,
        seed=20261010,
        stride_probes=probe_dir,
        stride_producer=producer_manifest,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        warmup=5,
        rounds=5,
        wall_iterations=20,
        profiler_iterations=10,
        kernel_csv=str(csvs[0]),
        raw_csv=str(raws[0]),
        kernel_samples=labels,
        results=summary,
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("vendor")
    parser.add_argument("platform")
    parser.add_argument("dtype", choices=("fp16", "bf16"))
    parser.add_argument("--dispatch-only", action="store_true")
    parser.add_argument("--stride-probes")
    parser.add_argument("--stride-producer")
    a = parser.parse_args()
    profile(
        a.output, a.vendor, a.platform, a.dtype, a.dispatch_only, a.stride_probes, a.stride_producer
    )
