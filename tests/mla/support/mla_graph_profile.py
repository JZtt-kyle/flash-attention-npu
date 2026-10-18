"""Separate graph replay and eager public timing with the same native inputs/R."""

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
from torch.utils._pytree import tree_flatten
from tests.common.compare import assert_fa_close
from tests.mla.support.public_cases.graph import make_case, metadata, invoke, compare, capture
from tests.mla.support.mla_gqa_options_profile import native_call
from tests.mla.support.mla_graph_profile_rows import (
    selected,
    modes,
    partition_kernels,
    select_model_rows,
)
from tests.mla.support.mla_native_dispatch import sha256, e2e_applicable
from tests.mla.support.mla_prefill_bsnd_profile import sample_kernels


def profiler_replay(eager, specs):
    # sample_kernels first calls us during WARMUP, then during RECORD. CANN
    # needs capture during RECORD to collect model/kernel associations.
    graphs, outputs = {}, {}
    warmup = True

    def call(i, name):
        nonlocal warmup
        if warmup:
            warmup = False
            return eager(i, name)
        if not graphs:
            for index in range(len(specs)):
                for phase in ("fa3", "native_cann"):
                    graph, result = capture(lambda: eager(index, phase), warmup=False)
                    graphs[index, phase], outputs[index, phase] = graph, result
        graphs[i, name].replay()
        return outputs[i, name]

    return call


def partition_graph(rows, labels, specs, dtype, representation, directory):
    # The existing sampler verifies exported rows against this raw source.
    # Its Model ID also separates capture preparation and replay bookkeeping
    # from the model's actual tasks; all excluded rows remain in the trace.
    paths = list(directory.rglob("op_summary_*.csv"))
    assert len(paths) == 1
    with paths[0].open() as stream:
        raw = sorted(csv.DictReader(stream), key=lambda r: Decimal(r["Task Start Time(us)"]))
    return partition_kernels(select_model_rows(rows, raw), labels, specs, dtype, representation)


def sample_wall(call, specs):
    samples = []
    for i, spec in enumerate(specs):
        names = modes(spec)
        for _ in range(5):
            for mode in ("eager", "graph"):
                for name in names:
                    result = call(i, name, mode)
                    torch.npu.synchronize()
        for round_ in range(5):
            for iteration in range(20):
                for mode in ("eager", "graph") if round_ % 2 == 0 else ("graph", "eager"):
                    for name in names if round_ % 2 == 0 else names[::-1]:
                        # Replacing an eager output can free its tensors. That
                        # preceding invocation must not be charged to this one;
                        # captured graph outputs remain owned by their graphs.
                        result = None
                        begin = time.perf_counter_ns()
                        result = call(i, name, mode)
                        torch.npu.synchronize()
                        samples.append(
                            dict(
                                case=i,
                                name=name,
                                mode=mode,
                                round=round_,
                                iteration=iteration,
                                e2e_us=(time.perf_counter_ns() - begin) / 1000,
                            )
                        )
    del result
    return samples


def run(
    representation,
    dtype,
    directory,
    dispatch=None,
    check=False,
    profile_probe=False,
    graph_kernels=False,
):
    if (profile_probe or graph_kernels) and dispatch is None:
        raise ValueError("Graph sampling requires verified native dispatch")
    directory.mkdir(parents=True, exist_ok=True)
    specs = selected(check)
    cases = [make_case(representation=representation, dtype=dtype, **s) for s in specs]
    native = [native_call(c) for c in cases]
    records = [metadata(c) for c in cases]

    def eager(i, name):
        c = cases[i]
        r = 4 if name.endswith("_4") else 1
        if name.startswith("native_cann"):
            return [native[i]() for _ in range(r)]
        if name == "metadata":
            return metadata(c)
        record = (
            metadata(c)
            if name.startswith("generate")
            else None
            if name.startswith("implicit")
            else records[i]
        )
        return [invoke(c, record) for _ in range(r)]

    if dispatch is None:
        accuracy = []
        for i, c in enumerate(cases):
            ours = invoke(c, records[i])
            refs = compare(c, ours)
            print(f"MLA_STRIDE_NATIVE_BEGIN {dtype} {i}", flush=True)
            source = native[i]()
            torch.npu.synchronize()
            print(f"MLA_STRIDE_NATIVE_END {dtype} {i}", flush=True)
            for j, name in enumerate(("out", "lse")):
                value = source[j].cpu()
                if j == 1:
                    value[value.isposinf()] = -torch.inf
                assert_fa_close(
                    value, refs[name + "_high"], refs[name + "_low"], name="native-graph-" + name
                )
            # Every public/native segment is captured and actually replayed in
            # preflight before any full sampling, including metadata generation.
            for name in modes(specs[i]):
                expected = eager(i, name)
                graph, actual = capture(lambda: eager(i, name))
                graph.replay()
                torch.npu.synchronize()
                left, left_spec = tree_flatten(actual)
                right, right_spec = tree_flatten(expected)
                assert left_spec == right_spec
                for a, b in zip(left, right):
                    torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
                graph.reset()
            torch.save(
                dict(actual=[x.cpu() for x in ours], source=[x.cpu() for x in source], refs=refs),
                directory / f"source-{i}.pt",
            )
            accuracy.append(dict(spec=specs[i], native=native[i].description()))
        (directory / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
        return

    proof = json.loads(dispatch.read_text())
    assert proof["dtype"] == dtype and proof["representation"] == representation
    assert proof["specs"] == json.loads(json.dumps(specs))
    assert all(sha256(p) == digest for p, digest in proof["artifacts"].items())
    assert os.environ["ASCEND_GLOBAL_LOG_LEVEL"] == "3"
    if profile_probe or graph_kernels:
        # On this CANN runtime, capturing any graph before profiling prevents
        # later captures from reporting their inner tasks, even on a new stream.
        # The driver runs this branch in a fresh process, before wall captures.
        # Keep the same real model-task validation and sampling protocol.
        indices = [0, 2, 3] if profile_probe else range(len(specs))
        # CANN also collects initialization tasks during profiler WARMUP. An
        # AICPU bootstrap stream can later be reused by a captured model, losing
        # that model's association in the exporter. Initialize all families
        # before entering the profiler, without capturing any graph.
        for index in indices:
            for phase in ("fa3", "native_cann"):
                eager(index, phase)
        torch.npu.synchronize()
        target = directory / ("profiler-probe" if profile_probe else "graph")
        target.mkdir()
        probe_specs = [specs[i] for i in indices]
        sample_kernels(
            profiler_replay(lambda i, name: eager(indices[i], name), probe_specs),
            probe_specs,
            target,
            dtype,
            partition=lambda rows, labels, s, dt: partition_graph(
                rows, labels, s, dt, representation, target
            ),
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        )
        return

    graphs, outputs, capture_times = {}, {}, []
    for i, spec in enumerate(specs):
        for name in modes(spec):
            begin = time.perf_counter_ns()
            graph, result = capture(lambda: eager(i, name))
            torch.npu.synchronize()
            capture_times.append(
                dict(case=i, name=name, prepare_capture_us=(time.perf_counter_ns() - begin) / 1000)
            )
            graphs[i, name], outputs[i, name] = graph, result

    def call(i, name, mode):
        if mode == "eager":
            return eager(i, name)
        graphs[i, name].replay()
        return outputs[i, name]

    (directory / "capture-times.json").write_text(json.dumps(capture_times, indent=2) + "\n")
    (directory / "npu-before.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )
    samples = sample_wall(call, specs)
    (directory / "wall-samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    for mode in ("eager", "graph"):
        target = directory / mode
        if mode == "graph":
            # This attempt's graph worker has already validated these samples
            # against its raw op_summary and model identities. No old attempt's
            # directory is accepted: the driver creates a new result root.
            kernels = json.loads((target / "kernel-samples.json").read_text())
        else:
            target.mkdir()
            kernels = sample_kernels(
                eager,
                specs,
                target,
                dtype,
                partition=lambda rows, labels, s, dt: partition_kernels(
                    rows, labels, s, dt, representation
                ),
            )
        rows = []
        for i, spec in enumerate(specs):
            source = proof["cases"][i]
            row = dict(
                spec=spec,
                native_template=source["native_template"],
                native_fd=source["native_fd"],
                native_call=native[i].description(),
            )
            row["e2e_applicable"] = e2e_applicable(
                row["native_template"],
                row["native_fd"],
                representation=representation,
                include_native_fd=spec["splits"] == 0,
            )
            for name in modes(spec):
                values = [
                    x for x in samples if x["case"] == i and x["name"] == name and x["mode"] == mode
                ]
                row[name] = dict(
                    median_e2e_us=statistics.median(x["e2e_us"] for x in values),
                    round_mean_e2e_us=[
                        statistics.fmean(x["e2e_us"] for x in values if x["round"] == r)
                        for r in range(5)
                    ],
                )
            for metric in ("kernel_us", "adaptation_us"):
                row[metric] = {
                    name: statistics.median(
                        x[metric] for x in kernels if x["case"] == i and x["name"] == name
                    )
                    for name in ("fa3", "native_cann")
                }
            rows.append(row)
        report = dict(
            mode=mode,
            dtype=dtype,
            representation=representation,
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            device=torch.npu.get_device_name(0),
            metadata_lifecycle=dict(complete_segments=[1, 4]),
            results=rows,
        )
        (target / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "npu-after.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path)
    parser.add_argument("--check", action="store_true")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--profile-probe", action="store_true")
    sampling.add_argument("--graph-kernels", action="store_true")
    args = parser.parse_args()
    run(
        args.representation,
        args.dtype,
        args.output,
        args.dispatch,
        args.check,
        args.profile_probe,
        args.graph_kernels,
    )
