"""Public varlen used E2E, including native packing and output restoration."""

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
from tests.mla_cann_baseline import check_vendor_configuration, environment, file_info
from tests.mla_varlen_profile import CannTndCall
from tests import test_flash_attn_npu_v3_mla_varlen_used as used_cases
from tests.test_flash_attn_npu_v3_mla_used_attention import (
    configure_producer,
    effective,
    load_attention,
    valid_outputs,
    references,
    check_math,
)
from tests.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.test_flash_attn_npu_v3_mla_tnd_schedule import build_oracle, evaluate, tnd_case


class CannTndUsedCall:
    """Test adapter to pinned FIA TND; no production fallback.

    Native TND uses the same cumulative array for offsets and lengths. Gather
    only axes with real allocation gaps, and restore Q output locations on
    every native call. Index/H2D preparation is outside both timed paths.
    """

    def __init__(self, inputs, item, scale):
        self.inputs = inputs
        qs, ks = effective(item)
        self.indices = {}
        for axis, capacities, lengths in (("q", item["q"], qs), ("k", item["kv"], ks)):
            starts = list(itertools.accumulate(capacities, initial=0))[:-1]
            positions = [p + j for p, n in zip(starts, lengths) for j in range(n)]
            self.indices[axis] = (
                torch.tensor(positions, dtype=torch.int64, device=inputs["q"].device)
                if lengths != capacities
                else None
            )
        self.native = CannTndCall(inputs, qs, ks, scale)
        self.compact_result = None

    def pack(self):
        return {
            name: tensor
            if self.indices["q" if name in ("q", "qv") else "k"] is None
            else tensor.index_select(0, self.indices["q" if name in ("q", "qv") else "k"])
            for name, tensor in self.inputs.items()
        }

    def restore(self, result):
        index = self.indices["q"]
        if index is None:
            return result
        out = torch.empty_like(self.inputs["qv"])
        # Scatter the native token-major LSE, then expose a zero-copy [H,T]
        # view; scattering dimension 1 would introduce avoidable transposes.
        lse = torch.empty(out.shape[:2], dtype=torch.float32, device=out.device)
        out.index_copy_(0, index, result[0])
        lse.index_copy_(0, index, result[1].T)
        return out, lse.T

    def conversion(self):
        packed = self.pack()
        restored = self.restore(self.compact_result)
        return packed, restored

    def __call__(self):
        packed = self.pack()
        self.native.inputs = packed
        self.native.kwargs.update(query_rope=packed["q"], key_rope=packed["k"])
        self.compact_result = self.native()
        return self.restore(self.compact_result)


def prepare(dtype_name, platform, output, probes, producer, cases_api):
    configs = getattr(cases_api, "PUBLIC_CASES", cases_api.PERFORMANCE_CASES)
    items = [c.get("item", c) for c in configs]
    oracle_dir = output / "source-schedule"
    oracle_dir.mkdir()
    oracle = build_oracle(oracle_dir)
    shapes = [
        tnd_case(
            *effective(c),
            heads=c["heads"],
            aic=platform["aic"],
            ratio=2,
            libapi=platform["libapi"],
            **dict(zip(("l1", "ub", "l0a", "l0b", "l0c"), platform["memory"])),
        )
        for c in items
    ]
    native_schedules = evaluate(oracle, shapes, native=True)
    (output / "dispatch.json").write_text(
        json.dumps(dict(cases=configs, inputs=shapes, native=native_schedules), indent=2) + "\n"
    )
    inputs, natives, cases, accuracy = [], [], [], []
    for index, spec in enumerate(configs):
        case = cases_api.production_case(probes, producer, spec)
        cases.append(case)
        args = dict(cases_api.forward_args(case), num_splits=1, return_attn_probs=True)
        inputs.append(args)
        native = CannTndUsedCall(case["devices"]["migrated"], items[index], case["scale"])
        natives.append(native)
        actual = api.flash_attn_varlen_func(**args)
        cases_api.launch(probes, case, "oracle")
        errors = cases_api.check_result(case, actual, source=True)
        source = valid_outputs(case, [t.cpu() for t in native()])
        refs = references(case)
        uq, uk = effective(items[index])
        nonempty = torch.tensor([k > 0 for q, k in zip(uq, uk) for _ in range(q)], dtype=torch.bool)
        assert (source[0][~nonempty] == 0).all() and source[1][:, ~nonempty].isposinf().all()
        native_errors = check_math(
            [source[0][nonempty], source[1][:, nonempty]],
            [r[nonempty] if i % 2 == 0 else r[:, nonempty] for i, r in enumerate(refs)],
        )
        accuracy.append(
            dict(
                case=index,
                native_fd=bool(native_schedules[index][5]),
                errors=errors,
                native_errors=native_errors,
            )
        )
    (output / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    return inputs, natives, cases, native_schedules


def profile(
    output,
    vendor,
    platform_path,
    dtype_name,
    probes,
    producer,
    dispatch_only=False,
    cases_api=used_cases,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    vendor = check_vendor_configuration(vendor)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    platform = json.loads(Path(platform_path).read_text())
    inputs, native, cases, native_schedules = prepare(
        dtype_name, platform, output, probes, producer, cases_api
    )
    if dispatch_only:
        for index, function in enumerate(native):
            os.write(1, f"MLA_VARLEN_USED_NATIVE_BEGIN {dtype_name} {index}\n".encode())
            function()
            torch.npu.synchronize()
            os.write(1, f"MLA_VARLEN_USED_NATIVE_END {dtype_name} {index}\n".encode())
        return
    env = environment(Path(__file__).resolve().parents[2] / "ops-transformer", vendor)
    env["fa3_extension"] = file_info(api.flash_attn_npu_3.__file__)
    env["profile_source"] = file_info(__file__)
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    torch.set_num_threads(1)

    def call(index, name, repeats=1):
        if name == "conversion":
            return native[index].conversion()
        function = (
            native[index] if name == "cann" else lambda: api.flash_attn_varlen_func(**inputs[index])
        )
        return [function() for _ in range(repeats)]

    def snapshot(name):
        (output / name).write_text(subprocess.check_output(["npu-smi", "info"], text=True))

    snapshot("npu-before.log")
    for index in range(len(inputs)):
        for _ in range(5):
            for name in ("fa3", "cann"):
                result = call(index, name, 4)
                torch.npu.synchronize()
    wall = []
    for index in range(len(inputs)):
        for round_id in range(5):
            for iteration in range(20):
                for repeats in (1, 4):
                    for name in ("fa3", "cann") if round_id % 2 == 0 else ("cann", "fa3"):
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
    conversions = []
    for index in range(len(inputs)):
        for _ in range(5):
            result = call(index, "conversion")
            torch.npu.synchronize()
        for round_id in range(5):
            for iteration in range(20):
                start = time.perf_counter_ns()
                result = call(index, "conversion")
                torch.npu.synchronize()
                conversions.append(
                    dict(
                        case=index,
                        round=round_id,
                        iteration=iteration,
                        wall_us=(time.perf_counter_ns() - start) / 1000,
                    )
                )
    (output / "conversion-wall.json").write_text(json.dumps(conversions, indent=2) + "\n")
    (output / "wall.json").write_text(json.dumps(wall, indent=2) + "\n")
    labels = []
    p = torch_npu.profiler
    with p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        schedule=p.schedule(wait=0, warmup=1, active=len(inputs) * 5 * 10 * 2, repeat=1),
        on_trace_ready=p.tensorboard_trace_handler(str(output / "profiler")),
        experimental_config=p._ExperimentalConfig(
            profiler_level=p.ProfilerLevel.Level0,
            export_type=p.ExportType.Text,
            data_simplification=False,
        ),
    ) as prof:
        result = call(0, "fa3")
        torch.npu.synchronize()
        prof.step()
        for index in range(len(inputs)):
            for round_id in range(5):
                for iteration in range(10):
                    for name in ("fa3", "cann") if round_id % 2 == 0 else ("cann", "fa3"):
                        result = call(index, name)
                        torch.npu.synchronize()
                        labels.append(
                            dict(case=index, name=name, round=round_id, iteration=iteration)
                        )
                        prof.step()
    del result
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    csvs = list(output.rglob("kernel_details.csv"))
    raws = list(output.rglob("op_summary_*.csv"))
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
    pos = 0
    native_patterns = {}
    for label in labels:
        begin = pos
        if label["name"] == "fa3":
            version = cases[label["case"]].get("config", {}).get("version", 5)
            producer_name = "ComputeMlaTndMetadata" if version == 3 else "ComputeMlaUsedMetadata"
            assert producer_name in raw[pos]["Op Name"]
            label["generation_us"] = float(raw[pos]["Task Duration(us)"])
            pos += 1
            kernel_name = "mla_tnd_" if version == 3 else "mla_tnd_used_"
            assert kernel_name + dtype_name in raw[pos]["Op Name"]
            label.update(
                kernel_us=float(raw[pos]["Task Duration(us)"]),
                kernel_name=raw[pos]["Op Name"],
                conversion_us=0,
            )
            pos += 1
        else:
            # The first native sample of every case is between FA3 calls.
            # Derive its complete task pattern from these generation anchors;
            # torch operators can emit multiple tasks. Verify every repetition,
            # including adjacent native calls at alternating-round boundaries.
            index = label["case"]
            if index not in native_patterns:
                assert label["round"] == label["iteration"] == 0
                end = next(i for i in range(pos, len(raw)) if "ComputeMla" in raw[i]["Op Name"])
                native_patterns[index] = [r["Op Name"] for r in raw[pos:end]]
            pattern = native_patterns[index]
            group = raw[pos : pos + len(pattern)]
            assert [r["Op Name"] for r in group] == pattern
            # Layout conversions also carry the enclosing FIA API name.
            # Identify the final attention task, not its API-name prefix.
            kernels = [r for r in group if r["Op Name"].endswith("_FusedInferAttentionScore")]
            assert len(kernels) == 1, group
            assert all("ComputeMla" not in r["Op Name"] for r in group)
            label.update(
                kernel_us=float(kernels[0]["Task Duration(us)"]),
                kernel_name=kernels[0]["Op Name"],
                conversion_us=sum(
                    float(r["Task Duration(us)"]) for r in group if r is not kernels[0]
                ),
            )
            pos += len(pattern)
        label["tasks"] = raw[begin:pos]
    assert pos == len(raw), (pos, len(raw))
    producers = [r for r in raw if "ComputeMla" in r["Op Name"]]
    assert len(producers) == len(labels) // 2
    summary = []
    for index, case in enumerate(cases):
        spec = case.get("config", case["item"])
        row = dict(case=index, spec=spec, native_fd=bool(native_schedules[index][5]))
        row["input_views"] = {
            name: dict(
                shape=list(t.shape),
                stride=list(t.stride()),
                offset=t.storage_offset(),
                storage_bytes=t.untyped_storage().size(),
                data_ptr=t.data_ptr(),
            )
            for name, t in case["devices"]["migrated"].items()
        }
        assert all(
            native[index].inputs[name] is t for name, t in case["devices"]["migrated"].items()
        )
        for repeats in (1, 4):
            values = {
                name: statistics.median(
                    r["e2e_us"]
                    for r in wall
                    if r["case"] == index and r["name"] == name and r["repeats"] == repeats
                )
                for name in ("fa3", "cann")
            }
            row["r" + str(repeats)] = dict(**values, ratio=values["fa3"] / values["cann"])
        row["kernel_us"] = {
            name: statistics.median(
                r["kernel_us"] for r in labels if r["case"] == index and r["name"] == name
            )
            for name in ("fa3", "cann")
        }
        row["generation_us"] = statistics.median(
            r["generation_us"] for r in labels if r["case"] == index and r["name"] == "fa3"
        )
        row["conversion_kernel_us"] = statistics.median(
            r["conversion_us"] for r in labels if r["case"] == index and r["name"] == "cann"
        )
        row["conversion_wall_us"] = statistics.median(
            r["wall_us"] for r in conversions if r["case"] == index
        )
        summary.append(row)
    snapshot("npu-after.log")
    report = dict(
        scope="Public varlen through allocated O/LSE completion; each of R=1/4 calls includes v3/v5 generation. Both sides receive the same physical views; native layout conversions, gathers and output restoration are included. Native FD cases reported separately; conversion wall measures the explicit packing/restoration adapter, while conversion kernels include all native non-attention tasks.",
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        warmup=5,
        rounds=5,
        wall_iterations=20,
        profiler_iterations=10,
        kernel_csv=str(csvs[0]),
        raw_csv=str(raws[0]),
        metadata_kernel_rows=len(producers),
        metadata_kernel_samples=producers,
        kernel_samples=labels,
        native_task_patterns=native_patterns,
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
    parser.add_argument("probe_dir")
    parser.add_argument("producer_manifest")
    parser.add_argument("--dispatch-only", action="store_true")
    parser.add_argument("--stride", action="store_true")
    args = parser.parse_args()
    cases_api = used_cases
    if args.stride:
        from tests import test_flash_attn_npu_v3_mla_tnd_stride_production as cases_api
        from tests.test_flash_attn_npu_v3_mla_tnd_stride import configure_producer, load_attention
    probes = load_attention(args.probe_dir)
    manifest = json.loads(Path(args.producer_manifest).read_text())
    from tests.mla_dense_build import ROOT, sha256

    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    assert sha256(Path(manifest["library"])) == manifest["library_sha256"]
    with load_probe(
        manifest["library"], Path(args.output).parent / (Path(args.output).name + "-producer")
    ) as producer:
        configure_producer(producer)
        profile(
            args.output,
            args.vendor,
            args.platform,
            args.dtype,
            probes,
            producer,
            args.dispatch_only,
            cases_api,
        )
