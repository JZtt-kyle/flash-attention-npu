"""Same-fragment component timing; nonzero softcap has no equivalent CANN baseline."""

import json
import os
import subprocess
import time

import torch

from tests.mla.support.mla_softcap_cases import cases, PERFORMANCE
from tests.mla.support.components import softcap
from tests.mla.support.components import prefill_attention as prefill
from tests.mla.support.components import split_kv_attention as partial
from tests.mla.support.components import split_kv_merge as merge
from tests.mla.support.mla_split_kv_attention_profile import sample_components


def profile(probes, output, *, preflight=False):
    output.mkdir(exist_ok=False)
    specs = [s for s in cases(probes["representation"]) if s["id"] in PERFORMANCE]
    prepared = [softcap.make_case(probes, s) for s in specs]
    tasks = [
        prefill.wave(c, 0) if probes["representation"] == "prefill" else None for c in prepared
    ]
    for c, s in zip(prepared, specs):
        total = len(softcap.task_rows(probes, c, s))
        assert total == (
            c["params"].cores if probes["representation"] == "prefill" else c["schedule"].cores
        )
    (output / "matrix.json").write_text(json.dumps(specs, indent=2) + "\n")
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    previous_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(previous_affinity)})
    (output / "npu-before.log").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
    try:

        def call(index, name):
            # The uncapped CANN path is an overhead reference only when cap>0.
            cap = 0.0 if name == "oracle" else specs[index]["softcap"]
            code = softcap.launch(
                probes, prepared[index], specs[index], name, tasks[index], cap=cap
            )
            if code:
                raise RuntimeError(f"{name} softcap profile launch failed: {code}")
            torch.npu.synchronize()

        for index, spec in enumerate(specs):
            for _ in range(5):
                for name in ("oracle", "migrated"):
                    call(index, name)
            if spec["softcap"] == 0:
                values = (
                    prepared[index]["raw"]
                    if probes["representation"] == "prefill"
                    else {
                        name: partial.raw_results(prepared[index], name)
                        for name in ("oracle", "migrated")
                    }
                )
                for a, b in zip(values["oracle"], values["migrated"]):
                    a, b = a.cpu(), b.cpu()
                    for core, (_, _, visible) in enumerate(
                        softcap.task_rows(probes, prepared[index], spec)
                    ):
                        rows = visible.shape[0]
                        torch.testing.assert_close(
                            a[core, :rows], b[core, :rows], rtol=0, atol=0, equal_nan=True
                        )
        report = dict(
            scope="Preallocated partial attention component, ctypes/launch/synchronize wall and kernel-only timing; no public E2E claim.",
            dtype=probes["dtype"],
            representation=probes["representation"],
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            device=torch.npu.get_device_name(0),
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
            native_recipe="Pinned CANN source probes; no native package/public call required",
        )
        if preflight:
            report.update(status="fixtures-passed", results=specs)
        else:
            suffix = (
                "prefill_attention"
                if probes["representation"] == "prefill"
                else "split_kv_attention"
            )
            results, paths = sample_components(call, specs, output, suffix)
            for result in results:
                result["source_equivalent"] = result["config"]["softcap"] == 0
                result["comparison"] = (
                    "same-parameter source"
                    if result["source_equivalent"]
                    else "uncapped CANN overhead reference only; no equivalent softcap source"
                )
            report.update(results=results, **paths)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "npu-after.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        return report
    finally:
        os.sched_setaffinity(0, previous_affinity)


def profile_merge(probes, output, *, preflight=False):
    """One M block, as in the existing FD profiler; identical canonical partials."""
    output.mkdir(exist_ok=False)
    specs = [s for s in cases(probes["representation"]) if s["id"] in PERFORMANCE]
    prepared = []
    for spec in specs:
        attention = softcap.make_case(probes, spec)
        raw = softcap.execute(probes, spec, attention, repeats=1)["raw"]
        if "oracle" not in raw:
            raw = dict(raw, oracle=raw["migrated"])
        if probes["representation"] == "prefill":
            case = prefill.prepare_merge(attention, raw)
        else:
            case = merge.allocate_case(
                dict(dtype=probes["torch_dtype"]),
                spec,
                softcap.task_rows(probes, attention, spec),
                raw,
            )
        case["groups"] = case["groups"][:1]
        prepared.append(case)
    (output / "matrix.json").write_text(json.dumps(specs, indent=2) + "\n")
    (output / "schedules.json").write_text(
        json.dumps(
            [
                dict(rows=c["groups"][0]["params"].rows, splits=c["groups"][0]["params"].splits)
                for c in prepared
            ],
            indent=2,
        )
        + "\n"
    )
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    previous_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(previous_affinity)})
    (output / "npu-before.log").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
    try:

        def call(index, name, canonical=True):
            cap = 0.0 if name == "oracle" else specs[index]["softcap"]
            if probes["representation"] == "prefill":
                code = prefill.launch_merge(
                    probes["libs"], prepared[index], name, canonical=canonical, softcap=cap
                )
            else:
                code = merge.launch(
                    dict(libs=probes["libs"]),
                    prepared[index],
                    name,
                    canonical=canonical,
                    softcap=cap,
                )
            if code:
                raise RuntimeError(f"{name} softcap merge profile launch failed: {code}")
            torch.npu.synchronize()

        for index, spec in enumerate(specs):
            for _ in range(5):
                for name in ("oracle", "migrated"):
                    call(index, name)
            if spec["softcap"] == 0:
                for a, b in zip(
                    prepared[index]["outputs"]["oracle"], prepared[index]["outputs"]["migrated"]
                ):
                    torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
            call(index, "migrated", canonical=False)
        report = dict(
            scope="One M block of FD on identical canonical FP32 O/sum/max; partial attention, canonicalization and allocation excluded. Nonzero cap compares only FD arithmetic/cast against CANN on capped partials, not a softcap attention oracle. Separate adaptation wall includes row adaptation and real accumulators; no public E2E claim.",
            dtype=probes["dtype"],
            representation=probes["representation"],
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            device=torch.npu.get_device_name(0),
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
        )
        if preflight:
            report.update(status="fixtures-passed", results=specs)
        else:
            adaptation = []
            for index in range(len(specs)):
                for _ in range(5):
                    call(index, "migrated", canonical=False)
                samples = []
                for _ in range(5):
                    for _ in range(20):
                        start = time.perf_counter_ns()
                        call(index, "migrated", canonical=False)
                        samples.append((time.perf_counter_ns() - start) / 1000)
                adaptation.append(samples)
            (output / "adaptation-wall.json").write_text(json.dumps(adaptation) + "\n")
            results, paths = sample_components(call, specs, output, "split_kv_merge")
            for result in results:
                result["source_equivalent"] = result["config"]["softcap"] == 0
            report.update(results=results, **paths)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "npu-after.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        return report
    finally:
        os.sched_setaffinity(0, previous_affinity)
