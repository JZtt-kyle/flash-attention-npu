"""GQA public calls versus the pinned native package, with unchanged timing boundaries."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
from tests.common.compare import assert_fa_close
from tests.mla.support.public_cases.gqa_options import make_case, metadata, invoke, compare
from tests.mla.support.mla_gqa_options_profile_rows import selected, modes, partition_kernels
from tests.mla.support.mla_native_dispatch import sha256, e2e_applicable
from tests.mla.support.mla_prefill_bsnd_profile import sample_kernels
from tests.mla.support.mla_window_public_profile import WindowCannCall
from tests.mla.support.mla_window_tnd_profile import WindowTndCall
from tests.mla.support.mla_window_mixed_profile import WindowMixedCall
from tests.mla.support.mla_window_paged_profile import WindowPagedCall
from tests.mla.support.mla_prefill_tnd_profile import PrefillTndCall
from tests.mla.support.mla_prefill_mixed_profile import PrefillMixedCall
from tests.mla.support.mla_prefill_paged_native import PrefillPagedCall


def native_call(c):
    layout, o = c["layout"], c["options"]
    if layout == "bsnd":
        return WindowCannCall(
            c["t"],
            scale=o["softmax_scale"],
            kv_lengths=c["klens"],
            spec=dict(window=o["window_size"], causal=o["causal"]),
        )
    if c["representation"] == "prefill":
        return (
            PrefillPagedCall
            if c.get("page")
            else PrefillTndCall
            if layout == "tnd"
            else PrefillMixedCall
        )(c)
    if c.get("page"):
        return WindowPagedCall(
            dict(
                args=dict(**c["t"], page_table=c["page_table"]),
                scale=o["softmax_scale"],
                mask=torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu(1),
                config=dict(
                    q=c.get("q", c["qlens"]),
                    lengths=c["klens"],
                    capacity=c["shape"][3],
                    causal=o["causal"],
                    window=o["window_size"],
                ),
            )
        )
    a = dict(c["t"])
    a.update(c["lengths"])
    a.update(softmax_scale=o["softmax_scale"], max_seqlen_k=c["shape"][3])
    s = dict(
        q=c["q"],
        kv=c["k"],
        uq=None,
        uk=c["klens"],
        heads=c["shape"][2],
        capacity=c["shape"][3],
        window=o["window_size"],
        causal=o["causal"],
    )
    return (WindowTndCall if layout == "tnd" else WindowMixedCall)(
        dict(args=a, spec=s, raw=c["raw"])
    )


def run(representation, dtype, directory, dispatch=None, check=False):
    directory.mkdir(parents=True, exist_ok=True)
    specs = selected(check)
    cases = [make_case(representation=representation, dtype=dtype, **s) for s in specs]
    calls = [native_call(c) for c in cases]
    records = [metadata(c, geometry=c["layout"] == "tnd") for c in cases]
    if dispatch is None:
        accuracy = []
        for i, (c, record, native) in enumerate(zip(cases, records, calls)):
            ours = invoke(c)
            refs = compare(c, ours, record)
            if c["options"]["softcap"]:
                uncapped = dict(c, options=dict(c["options"], softcap=0))
                refs = compare(
                    uncapped, invoke(uncapped), metadata(uncapped, geometry=c["layout"] == "tnd")
                )
            print(f"MLA_STRIDE_NATIVE_BEGIN {dtype} {i}", flush=True)
            source = native()
            torch.npu.synchronize()
            print(f"MLA_STRIDE_NATIVE_END {dtype} {i}", flush=True)
            for j, name in enumerate(("out", "lse")):
                value = source[j].cpu()
                if j == 1:
                    value[value.isposinf()] = -torch.inf
                assert_fa_close(
                    value, refs[name + "_high"], refs[name + "_low"], name="native-gqa-" + name
                )
            torch.save(
                dict(actual=[x.cpu() for x in ours], source=[x.cpu() for x in source], refs=refs),
                directory / f"source-{i}.pt",
            )
            accuracy.append(
                dict(spec=specs[i], record=record.cpu().tolist(), native=native.description())
            )
        (directory / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
        return
    proof = json.loads(dispatch.read_text())
    assert proof["dtype"] == dtype and proof["representation"] == representation
    assert proof["specs"] == json.loads(json.dumps(specs))
    assert all(sha256(p) == digest for p, digest in proof["artifacts"].items())
    assert os.environ["ASCEND_GLOBAL_LOG_LEVEL"] == "3"

    def call(i, name):
        c = cases[i]
        r = 4 if name.endswith("_4") else 1
        if name.startswith("native_cann"):
            return [calls[i]() for _ in range(r)]
        if name == "metadata":
            return metadata(c)
        record = (
            metadata(c)
            if name.startswith("generate")
            else None
            if name.startswith("implicit") or c["layout"] == "tnd"
            else records[i]
        )
        return [invoke(c, record) for _ in range(r)]

    (directory / "npu-before.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )
    samples = []
    for i, spec in enumerate(specs):
        names = modes(spec)
        for _ in range(5):
            for name in names:
                result = call(i, name)
                torch.npu.synchronize()
        for round_ in range(5):
            for iteration in range(20):
                for name in names if round_ % 2 == 0 else names[::-1]:
                    start = time.perf_counter_ns()
                    result = call(i, name)
                    torch.npu.synchronize()
                    samples.append(
                        dict(
                            case=i,
                            name=name,
                            round=round_,
                            iteration=iteration,
                            e2e_us=(time.perf_counter_ns() - start) / 1000,
                        )
                    )
    del result
    (directory / "wall-samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    kernels = sample_kernels(call, specs, directory, dtype, partition=partition_kernels)
    rows = []
    for i, spec in enumerate(specs):
        native = proof["cases"][i]
        row = dict(
            spec=spec,
            native_template=native["native_template"],
            native_fd=native["native_fd"],
            native_call=calls[i].description(),
        )
        row["e2e_applicable"] = not spec.get("cap", 0) and e2e_applicable(
            row["native_template"],
            row["native_fd"],
            representation=representation,
            include_native_fd=spec["splits"] == 0,
        )
        for name in modes(spec):
            values = [x for x in samples if x["case"] == i and x["name"] == name]
            row[name] = dict(
                median_e2e_us=statistics.median(x["e2e_us"] for x in values),
                round_mean_e2e_us=[
                    statistics.fmean(x["e2e_us"] for x in values if x["round"] == r)
                    for r in range(5)
                ],
            )
        row["kernel_us"] = {
            n: statistics.median(
                x["kernel_us"] for x in kernels if x["case"] == i and x["name"] == n
            )
            for n in ("fa3", "native_cann")
        }
        row["adaptation_us"] = {
            n: statistics.median(
                x["adaptation_us"] for x in kernels if x["case"] == i and x["name"] == n
            )
            for n in ("fa3", "native_cann")
        }
        rows.append(row)
    report = dict(
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
    (directory / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    (directory / "npu-after.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--representation", required=True)
    p.add_argument("--dtype", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dispatch", type=Path)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    run(a.representation, a.dtype, a.output, a.dispatch, a.check)
