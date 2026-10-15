"""Public varlen E2E: every forward includes its private v3 generation."""

from tests.mla.paths import ROOT

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
from tests.mla.support.mla_cann_baseline import check_vendor_configuration, environment, file_info
from tests.mla.support.components.tnd_attention import references
from tests.mla.support.components.tnd_schedule import build_oracle, evaluate, tnd_case


# Fixed before sampling. Native host Hq<=128; larger-head component cases
# remain in the independent TND suite. Keep long-KV/zero-cost cases regardless
# of their native split policy or measured speed.
PERFORMANCE_CASES = [
    ([3, 0, 1], [17, 0, 33], 128),
    ([1, 7, 2], [1, 513, 31], 16),
    ([0, 3, 1, 0, 7], [33, 511, 0, 513, 1025], 128),
    ([1, 2, 3], [16384, 8193, 17], 8),
    ([2] * 10, [511] + [0] * 9, 128),
    ([3, 0, 1, 7, 0], [17, 65, 513, 0, 1], 128),
    ([3, 0] * 21, [0, 513] * 21, 128),
    ([257, 1, 0, 3], [33, 513, 17, 1], 1),
    ([0, 1, 0, 0], [0, 0, 0, 1], 1),
    ([1, 1, 1], [17, 129, 257], 128),
]


class CannTndCall:
    """Only a test adapter to pinned native FIA, never a production fallback."""

    def __init__(self, inputs, qs, ks, scale):
        self.inputs = inputs
        q = inputs["q"]
        self.kwargs = dict(
            query_rope=q,
            key_rope=inputs["k"],
            actual_seq_qlen=list(itertools.accumulate(qs)),
            actual_seq_kvlen=list(itertools.accumulate(ks)),
            num_query_heads=q.shape[1],
            num_key_value_heads=1,
            softmax_scale=scale,
            input_layout="TND",
            sparse_mode=0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            return_softmax_lse=True,
        )

    def __call__(self):
        q, v = self.inputs["qv"], self.inputs["v"]
        out = torch.empty_like(q)
        lse = torch.empty((*q.shape[:2], 1), dtype=torch.float32, device=q.device)
        out, lse = torch.ops.npu.npu_fused_infer_attention_score_v2.out(
            q, v, v, **self.kwargs, out=[out, lse]
        )
        assert out.shape == q.shape and lse.shape == (*q.shape[:2], 1)
        return out, lse.squeeze(-1).T


def profile(output, vendor, platform_path, dtype_name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    vendor = check_vendor_configuration(vendor)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    platform = json.loads(Path(platform_path).read_text())
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
        for q, k, h in PERFORMANCE_CASES
    ]
    native_schedules = evaluate(oracle, shapes, native=True)
    (output / "dispatch.json").write_text(
        json.dumps(dict(cases=PERFORMANCE_CASES, inputs=shapes, native=native_schedules), indent=2)
        + "\n"
    )
    torch.set_num_threads(8)
    inputs, native, accuracy = [], [], []
    for index, (qs, ks, heads) in enumerate(PERFORMANCE_CASES):
        gen = torch.Generator().manual_seed(20261010)
        cpu = {
            n: torch.randn(total, h, d, generator=gen).to(dtype)
            for n, total, h, d in (
                ("q", sum(qs), heads, 64),
                ("qv", sum(qs), heads, 512),
                ("k", sum(ks), 1, 64),
                ("v", sum(ks), 1, 512),
            )
        }
        device = {n: t.npu() for n, t in cpu.items()}
        args = dict(
            **device,
            cu_seqlens_q=torch.tensor(
                list(itertools.accumulate(qs, initial=0)), dtype=torch.int32, device="npu"
            ),
            cu_seqlens_k=torch.tensor(
                list(itertools.accumulate(ks, initial=0)), dtype=torch.int32, device="npu"
            ),
            max_seqlen_q=max(qs),
            max_seqlen_k=max(ks),
            num_splits=1,
            return_attn_probs=True,
        )
        inputs.append(args)
        native.append(CannTndCall(device, qs, ks, 576**-0.5))
        actual = api.flash_attn_varlen_func(**args)
        source = native[-1]()
        refs = references(dict(spec=(qs, ks, heads, None), inputs=cpu, scale=576**-0.5))
        nonempty = torch.cat([torch.full((q,), k > 0) for q, k in zip(qs, ks)])
        for i in range(2):
            a, s = actual[i].cpu(), source[i].cpu()
            assert_fa_close(a, refs[i], refs[i + 2], name="varlen-fa3")
            valid = nonempty if i == 0 else (slice(None), nonempty)
            if nonempty.any():
                assert_fa_close(s[valid], refs[i][valid], refs[i + 2][valid], name="varlen-cann")
            if i == 0:
                assert (s[~nonempty] == 0).all()
            else:
                assert s[:, ~nonempty].isposinf().all()
        accuracy.append(dict(case=index, native_fd=bool(native_schedules[index][5]), passed=True))
    (output / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    env = environment(ROOT.parent / "ops-transformer", vendor)
    env["fa3_extension"] = file_info(api.flash_attn_npu_3.__file__)
    env["profile_source"] = file_info(__file__)
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    torch.set_num_threads(1)

    def call(index, name, repeats=1):
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
    producers = [r for r in rows if "ComputeMlaTndMetadata" in r["Name"]]
    kernels = [r for r in rows if "ComputeMlaTndMetadata" not in r["Name"]]
    assert len(kernels) == len(labels)
    assert len(producers) in (0, len(labels) // 2)
    for r, label in zip(kernels, labels):
        assert (
            "mla_tnd_" + dtype_name if label["name"] == "fa3" else "FusedInferAttentionScore"
        ) in r["Name"]
        label.update(kernel_us=float(r["Duration(us)"]), kernel_name=r["Name"])
    summary = []
    for index, spec in enumerate(PERFORMANCE_CASES):
        row = dict(case=index, spec=spec, native_fd=bool(native_schedules[index][5]))
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
        summary.append(row)
    snapshot("npu-after.log")
    report = dict(
        scope="Public varlen through O/LSE completion; each of R=1/4 calls includes v3 generation; native FD cases reported separately.",
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
    args = parser.parse_args()
    profile(args.output, args.vendor, args.platform, args.dtype)
