"""INT8 public eager/graph sampling using the established wall/kernel drivers."""

import argparse
import csv
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch_npu
from torch.utils._pytree import tree_flatten

from tests.mla.support.mla_int8_cases import NativeCall, SOURCE_CASES
from tests.mla.support.mla_int8_bsnd_profile_rows import (
    selected,
    modes,
    summarize,
    partition_kernels,
)
from tests.mla.support.mla_graph_profile import sample_wall, profiler_replay
from tests.mla.support.mla_graph_profile_rows import select_model_rows
from tests.mla.support.mla_prefill_bsnd_profile import sample_kernels
from tests.mla.support.mla_native_dispatch import sha256, e2e_applicable
from tests.mla.support.public_cases.graph import capture
from tests.mla.support.public_cases.int8_bsnd import make_case, metadata, invoke


class PublicNativeCall:
    def __init__(self, case):
        self.native = NativeCall(case["spec"], case["raw"])
        self.inputs = case["tensors"]

    def __call__(self):
        n = self.native
        # Match the public allocation boundary; fixed ND->NZ source packing is
        # preparation only, explicitly recorded below. No target cache packing.
        # Preserve the public device views until the timed invocation. Only
        # source-only NZ cache packing is allowed outside this boundary.
        qv = self.inputs["qv"].contiguous()
        options = dict(
            n.kwargs,
            query_rope=self.inputs["q"].contiguous(),
            dequant_scale_query=self.inputs["qs"].contiguous(),
            dequant_scale_key=self.inputs["ks"],
            dequant_scale_value=self.inputs["vs"],
            block_table=self.inputs["table"].contiguous(),
        )
        out = torch.empty_like(qv, dtype=torch.bfloat16)
        lse = torch.empty(1, dtype=torch.float32, device=out.device)
        return torch.ops.npu.npu_fused_infer_attention_score_v2.out(
            qv, n.key, n.key, **options, out=[out, lse]
        )[0]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(directory, check=False, dispatch=None, graph_kernels=False):
    directory.mkdir(parents=True, exist_ok=True)
    specs = selected(check)
    cases = [make_case(SOURCE_CASES[s["index"]], s["view"]) for s in specs]
    natives = [PublicNativeCall(c) for c in cases]
    records = [metadata(c) for c in cases]

    def eager(i, name):
        repeat = 4 if name.endswith("_4") else 1
        if name.startswith("native_cann"):
            return [natives[i]() for _ in range(repeat)]
        if name == "metadata":
            return metadata(cases[i])
        record = (
            metadata(cases[i])
            if name.startswith("generate")
            else None
            if name.startswith("implicit")
            else records[i]
        )
        return [invoke(cases[i], record, lse=False) for _ in range(repeat)]

    if dispatch is None:
        evidence = []
        for i, c in enumerate(cases):
            actual = invoke(c, records[i], lse=False).cpu()
            print(f"MLA_STRIDE_NATIVE_BEGIN int8 {i}", flush=True)
            source = natives[i]().cpu()
            print(f"MLA_STRIDE_NATIVE_END int8 {i}", flush=True)
            torch.testing.assert_close(actual, source, rtol=0, atol=0)
            for name in modes(specs[i]):
                expected = eager(i, name)
                graph, output = capture(lambda: eager(i, name))
                graph.replay()
                torch.npu.synchronize()
                left, ls = tree_flatten(output)
                right, rs = tree_flatten(expected)
                assert ls == rs
                for a, b in zip(left, right):
                    torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
                graph.reset()
            torch.save(dict(actual=actual, source=source), directory / f"source-{i}.pt")
            evidence.append(
                dict(
                    spec=specs[i],
                    source_pack_seconds=natives[i].native.pack_seconds,
                    source_elementwise=True,
                    captured_modes=modes(specs[i]),
                )
            )
        save(directory / "preflight.json", evidence)
        return

    proof = json.loads(dispatch.read_text())
    assert proof["specs"] == list(specs) and proof["dtype"] == proof["representation"] == "int8"
    assert all(sha256(p) == h for p, h in proof["artifacts"].items())
    assert all(
        e2e_applicable(
            x["native_template"], x["native_fd"], representation="int8", include_native_fd=True
        )
        for x in proof["cases"]
    )
    assert os.environ["ASCEND_GLOBAL_LOG_LEVEL"] == "3"
    if graph_kernels:
        for i in range(len(specs)):
            for name in ("fa3", "native_cann"):
                eager(i, name)
        torch.npu.synchronize()
        target = directory / "graph"
        target.mkdir()

        def partition(rows, labels, specs, dtype):
            paths = list(target.rglob("op_summary_*.csv"))
            assert len(paths) == 1
            with paths[0].open() as f:
                raw = sorted(csv.DictReader(f), key=lambda x: Decimal(x["Task Start Time(us)"]))
            return partition_kernels(select_model_rows(rows, raw), labels, specs, dtype)

        sample_kernels(
            profiler_replay(eager, specs),
            specs,
            target,
            "int8",
            partition=partition,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        )
        return

    graphs, outputs, captures = {}, {}, []
    for i, spec in enumerate(specs):
        for name in modes(spec):
            start = time.perf_counter_ns()
            graph, output = capture(lambda: eager(i, name))
            graphs[i, name], outputs[i, name] = graph, output
            captures.append(
                dict(case=i, name=name, prepare_capture_us=(time.perf_counter_ns() - start) / 1000)
            )
    save(directory / "capture-times.json", captures)

    def call(i, name, mode):
        if mode == "eager":
            return eager(i, name)
        graphs[i, name].replay()
        return outputs[i, name]

    (directory / "npu-before.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )
    samples = sample_wall(call, specs)
    save(directory / "wall-samples.json", samples)
    reports = summarize(samples, specs)
    for mode, report in reports.items():
        target = directory / mode
        target.mkdir(exist_ok=True)
        if mode == "eager":
            sample_kernels(eager, specs, target, "int8", partition=partition_kernels)
        assert (target / "kernel-samples.json").is_file()
        report.update(
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            device=torch.npu.get_device_name(0),
            kernel_samples=str(target / "kernel-samples.json"),
            profiler_iterations=10,
        )
        save(target / "profile.json", report)
    (directory / "npu-after.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dispatch", type=Path)
    parser.add_argument("--graph-kernels", action="store_true")
    args = parser.parse_args()
    run(args.output, args.check, args.dispatch, args.graph_kernels)
