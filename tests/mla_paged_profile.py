"""Public BSND PA metadata E2E against the pinned native CANN PA dispatch."""

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

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla_cann_baseline import check_vendor_configuration, environment, file_info
from tests.mla_dense_build import ROOT, sha256
from tests.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.test_flash_attn_npu_v3_mla_paged_public import (
    PERFORMANCE_CASES,
    public,
    generate as metadata,
    make_case,
    check_case,
    production,
)
from tests.test_flash_attn_npu_v3_mla_paged_attention import (
    configure_producer,
    load_attention,
    valid_outputs,
    references,
    check_math,
)
from tests.test_flash_attn_npu_v3_mla_paged_schedule import build_oracle, execute


class CannPagedCall:
    """Same physical inputs, with necessary native layout copies inside each call.

    Pinned arch22 accepts BSH/BSND Q and BnBsH cache with independent page
    strides. Preserve these strides when supported; pack other views per call.
    Lengths are submitted as the native API's host list, never read from device.
    """

    def __init__(self, args, lengths, scale):
        self.inputs = {n: args[n] for n in ("q", "qv", "k", "v", "page_table")}
        q = args["q"]
        self.kwargs = dict(
            actual_seq_kvlen=list(lengths),
            num_query_heads=q.shape[2],
            num_key_value_heads=1,
            softmax_scale=scale,
            input_layout="BSND",
            pre_tokens=2147483647,
            next_tokens=2147483647,
            block_size=args["k"].shape[1],
            return_softmax_lse=True,
        )

    def pack(self):
        packed = {n: self.inputs[n].contiguous() for n in ("q", "qv", "page_table")}
        for name in ("k", "v"):
            cache = self.inputs[name].squeeze(2)
            # CANN only accepts independent dim-0 padding, not arbitrary row
            # stride/broadcast. This decision uses shape/stride only.
            packed[name] = (
                cache
                if (
                    cache.stride(1) == cache.shape[2]
                    and cache.stride(0) >= cache.shape[1] * cache.shape[2]
                )
                else cache.contiguous()
            )
        return packed

    def __call__(self):
        p = self.pack()
        q = p["q"]
        out = torch.empty((*q.shape[:-1], 512), device=q.device, dtype=q.dtype)
        lse = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1], 1), device=q.device, dtype=torch.float32
        )
        out, lse = torch.ops.npu.npu_fused_infer_attention_score_v2.out(
            p["qv"],
            p["v"],
            p["v"],
            query_rope=p["q"],
            key_rope=p["k"],
            block_table=p["page_table"],
            **self.kwargs,
            out=[out, lse],
        )
        return out, lse.squeeze(-1)


def prepare(output, probe_dir, producer_manifest, dispatch_only=False):
    probes = load_attention(probe_dir)
    manifest = json.loads(Path(producer_manifest).read_text())
    assert sha256(Path(manifest["library"])) == manifest["library_sha256"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    oracle_dir = output / "source-schedule"
    oracle_dir.mkdir()
    oracle = build_oracle(oracle_dir, used=True, paged=True)
    args, natives, records, accuracy = [], [], [], []
    with load_probe(manifest["library"], output / "producer") as producer:
        configure_producer(producer)
        cases = [make_case(probes, producer, spec) for spec in PERFORMANCE_CASES]
        items = [case["item"] for case in cases]
        schedules = execute(oracle, items, native=True)
        source = execute(oracle, items)
        # Freeze the complete matrix and native dispatch before any timings.
        (output / "dispatch.json").write_text(
            json.dumps(
                dict(
                    cases=PERFORMANCE_CASES,
                    inputs=items,
                    native=schedules,
                    source=source,
                ),
                indent=2,
            )
            + "\n"
        )
        for index, case in enumerate(cases):
            arg = production.forward_args(case)
            assert arg.pop("seqused_q") is None
            args.append(arg)
            records.append(metadata(arg))
            natives.append(CannPagedCall(arg, case["config"]["lengths"], case["scale"]))
            if dispatch_only:
                continue  # Full source/dual-math checks precede the timed run.
            errors = check_case(probes, case, metadata=records[-1])
            native_result = natives[-1]()
            native = valid_outputs(case, [t.cpu() for t in native_result])
            refs = references(case)
            nonempty = torch.tensor(
                [
                    k > 0
                    for q, k in zip(case["config"]["q"], case["config"]["lengths"])
                    for _ in range(q)
                ],
                dtype=torch.bool,
            )
            assert (native[0][~nonempty] == 0).all() and native[1][:, ~nonempty].isposinf().all()
            native_errors = check_math(
                [native[0][nonempty], native[1][:, nonempty]],
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
            torch.save(
                dict(
                    host=case["host"],
                    pages=case["table_host"],
                    native=[t.cpu() for t in native_result],
                    source=[t.cpu() for t in case["outputs"]["oracle"]],
                    actual=[t.cpu() for t in public(dict(arg, scheduler_metadata=records[-1]))[:2]],
                ),
                output / ("accuracy-" + str(index) + ".pt"),
            )
    (output / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    return args, natives, records, schedules, PERFORMANCE_CASES


def profile(
    output,
    vendor,
    dtype_name,
    dispatch_only=False,
    probe_dir=None,
    producer_manifest=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    vendor = check_vendor_configuration(vendor)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    torch.set_num_threads(8)
    args, natives, records, schedules, specs = prepare(
        output, probe_dir, producer_manifest, dispatch_only
    )
    assert all(
        a["q"].dtype == {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name] for a in args
    )
    if dispatch_only:
        for index, native in enumerate(natives):
            os.write(1, f"MLA_PAGED_NATIVE_BEGIN {dtype_name} {index}\n".encode())
            native()
            torch.npu.synchronize()
            os.write(1, f"MLA_PAGED_NATIVE_END {dtype_name} {index}\n".encode())
        return
    env = environment(Path(__file__).resolve().parents[2] / "ops-transformer", vendor)
    env.update(
        fa3_extension=file_info(api.flash_attn_npu_3.__file__), profile_source=file_info(__file__)
    )
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

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
                    if "mla_paged_bsnd_" in rows[i]["Name"] or "ComputeMla" in rows[i]["Name"]
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
            "reuse": "mla_paged_bsnd_" + dtype_name,
            "generation": "ComputeMlaPagedMetadata",
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
            native_empty=sum(spec["lengths"]) == 0,
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
        scope="Public BSND PA FA3 versus pinned native PA; required query/cache/table conversions included per call. R=1/4 includes O/LSE completion; generated includes one public metadata generation. Native FD and empty paths identified separately.",
        dtype=dtype_name,
        seed=20261011,
        probes=probe_dir,
        producer=producer_manifest,
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
    parser.add_argument("probes")
    parser.add_argument("producer")
    parser.add_argument("dtype", choices=("fp16", "bf16"))
    parser.add_argument("--dispatch-only", action="store_true")
    a = parser.parse_args()
    profile(a.output, a.vendor, a.dtype, a.dispatch_only, a.probes, a.producer)
