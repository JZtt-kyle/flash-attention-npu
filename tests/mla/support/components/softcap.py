"""Reuse existing numerical probes, fragment equations and FD allocation for softcap."""

import ctypes as ct
import json
import os
from pathlib import Path
import shutil

import torch

from tests.mla.paths import work_dir
from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.mla_validation_build import matching_files
from tests.mla.support.mla_split_kv_merge_build import build_merge
from tests.mla.support.mla_split_kv_attention_reference import (
    fragments,
    references as compressed_refs,
)
from tests.mla.support.mla_prefill_reference import references as prefill_refs
from tests.mla.support.components import prefill_attention as prefill
from tests.mla.support.components import split_kv_attention as partial
from tests.mla.support.components import split_kv_merge as merge


def valid_merge(manifest, dtype):
    try:
        return (
            manifest.get("split_kv_merge") is True
            and manifest.get("dtype") == dtype
            and manifest.get("cann") == os.environ["ASCEND_HOME_PATH"]
            and bool(manifest["sources"])
            and matching_files({str(ROOT / k): v for k, v in manifest["sources"].items()})
            and bool(manifest["dependencies"])
            and str(Path(os.environ["ASCEND_HOME_PATH"]) / "bin/bisheng")
            in manifest["dependencies"]
            and shutil.which("g++") in manifest["dependencies"]
            and matching_files(manifest["dependencies"])
            and bool(manifest["libraries"])
            and matching_files({v["path"]: v["sha256"] for v in manifest["libraries"].values()})
        )
    except (AttributeError, KeyError, TypeError, OSError):
        return False


def load(representation, dtype, directory):
    directory = Path(directory) / representation / "probes" / dtype
    if representation == "prefill":
        libs, directory = prefill.load(dtype, directory.parent)
        return dict(representation=representation, dtype=dtype, libs=libs, directory=directory)
    cached = work_dir() / "softcap-compressed-probes" / dtype
    path = cached / "build-manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        manifest = None
    if not valid_merge(manifest, dtype):
        build_merge(cached, dtype)
        manifest = json.loads(path.read_text())
        for compiler in (
            Path(os.environ["ASCEND_HOME_PATH"]) / "bin/bisheng",
            Path(shutil.which("g++")),
        ):
            manifest["dependencies"][str(compiler)] = sha256(compiler)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not valid_merge(manifest, dtype):
        raise RuntimeError("Invalid softcap component build")
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, directory / path.name)
    libraries = {k: ct.CDLL(v["path"]) for k, v in manifest["libraries"].items()}
    for name in ("oracle", "migrated"):
        libraries[name].launch_merge.argtypes = [ct.c_void_p] * 9
        libraries[name].launch_merge.restype = ct.c_int
        libraries[name].launch_merge_softcap.argtypes = [ct.c_void_p] * 9 + [ct.c_float]
        libraries[name].launch_merge_softcap.restype = ct.c_int
    pd = {k: libraries["partial_" + k] for k in ("oracle", "migrated", "schedule")}
    for name in ("oracle", "migrated"):
        signature = [ct.c_void_p] * 10 + [ct.c_float] + [ct.c_void_p] * 3
        pd[name].launch_partial.argtypes = signature
        pd[name].launch_partial.restype = ct.c_int
        pd[name].launch_softcap.argtypes = signature + [ct.c_float]
        pd[name].launch_softcap.restype = ct.c_int
    pd["schedule"].prepare_partial.argtypes = (
        [ct.c_int64] * 4
        + [ct.c_void_p]
        + [ct.c_int64] * 4
        + [ct.c_uint64]
        + [ct.c_int64] * 3
        + [ct.c_void_p] * 3
    )
    pd["schedule"].prepare_partial.restype = ct.c_int
    pd["dtype"] = torch.float16 if dtype == "fp16" else torch.bfloat16
    pd["mask"] = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu()
    (directory / "loaded.json").write_text(json.dumps(manifest["libraries"], indent=2) + "\n")
    return dict(
        representation=representation,
        dtype=dtype,
        libs=libraries,
        partial=(pd, directory),
        directory=directory,
        torch_dtype=pd["dtype"],
    )


def pattern(inputs, kind, representation):
    """Two large components cancel only after summation; V stays independent for prefill."""
    if kind is None:
        return
    if kind == "rounding-ties":
        inputs["q"].zero_()
        if "qv" in inputs:
            inputs["qv"].zero_()
        v = inputs["v"]
        step = torch.finfo(v.dtype).eps
        # Visible keys 31/32 straddle a tile/split boundary. Each output is
        # exactly a positive/negative midpoint, with even and odd lower bits.
        lo = torch.tensor([1, -1, 1 + step, -1 - step], dtype=v.dtype)
        hi = torch.tensor([1 + step, -1 - step, 1 + 2 * step, -1 - 2 * step], dtype=v.dtype)
        v.zero_()
        v[:, 31] = lo.repeat(v.shape[-1] // 4)
        v[:, 32] = hi.repeat(v.shape[-1] // 4)
        return
    if representation == "compressed":
        qr, qc, kr, kc = (inputs[k] for k in ("q", "qv", "k", "v"))
    else:
        qc, qr = inputs["q"][..., :128], inputs["q"][..., 128:]
        kc, kr = inputs["k"][..., :128], inputs["k"][..., 128:]
    qc.zero_()
    qr.zero_()
    qc[..., 0] = 32
    values = torch.linspace(-24, 24, kc.shape[1]).to(kc.dtype)[None, :, None]
    kc[..., 0] = values
    if kind == "cancellation":
        qr[..., 0] = 32
        kr[..., 0] = -values + 0.5
    elif kind != "saturation":
        raise ValueError("Unknown softcap input pattern")


def make_case(probes, spec, first=0):
    rep = probes["representation"]
    if rep == "prefill":
        case = prefill.make_case(spec, probes["dtype"], probes["libs"]["cores"])
        pattern(case["inputs"], spec.get("pattern"), rep)
        for key, value in case["inputs"].items():
            case["device"][key].copy_(value)
            case["native"][key].copy_(value)
        return case
    case = partial.make_case(probes["partial"], spec, first)
    pattern(case["inputs"], spec.get("pattern"), rep)
    for value, (before, device, prefix) in zip(case["inputs"].values(), case["storage"][:4]):
        before[prefix : prefix + value.numel()] = value.flatten()
        device.copy_(before)
    return case


def task_rows(probes, case, spec):
    return (
        case["tasks"]
        if probes["representation"] == "prefill"
        else list(fragments(spec, spec["m"], spec["n"]))
    )


def reference(probes, case, spec):
    if probes["representation"] == "prefill":
        return prefill_refs(case["inputs"], spec)
    keys = [f"{k}_{p}" for p in ("high", "low") for k in ("out", "lse", "max", "sum")]
    values = dict(zip(keys, compressed_refs(case["inputs"], spec, spec["m"], spec["n"])))
    full = (
        values
        if spec["splits"] == 1
        else dict(
            zip(keys, compressed_refs(case["inputs"], dict(spec, splits=1), spec["m"], spec["n"]))
        )
    )
    for p in ("high", "low"):
        for k in ("out", "lse"):
            values[f"full_{k}_{p}"] = full[f"{k}_{p}"].repeat_interleave(spec["splits"], dim=0)
    return values


def launch(probes, case, spec, name, tasks=None, *, cap=None, direct=False):
    cap = spec["softcap"] if cap is None else cap
    if probes["representation"] == "prefill":
        previous = case["params"].softcap
        case["params"].softcap = cap
        try:
            return prefill.launch_case(probes["libs"], case, name, tasks, partial=not direct)
        finally:
            case["params"].softcap = previous
    return partial.launch(probes["partial"], case, name, softcap=cap)


def execute(probes, spec, case, repeats=3):
    """Execute all fragments; no reference or successful-test cache is consulted."""
    rep = probes["representation"]
    names = ("oracle", "migrated") if spec["softcap"] == 0 else ("migrated",)
    tasks = task_rows(probes, case, spec)
    chunks = {name: [[], [], []] for name in names}
    waves, direct = [], []
    first = 0
    while first < len(tasks):
        current_case = case if first == 0 or rep == "prefill" else make_case(probes, spec, first)
        descriptors = prefill.wave(current_case, first) if rep == "prefill" else None
        count = current_case["params"].cores if rep == "prefill" else current_case["schedule"].cores
        wave = []
        if rep == "prefill":
            for name in names:
                for t, value in zip(
                    current_case["raw"][name],
                    (0, 0, -torch.inf) if name == "oracle" else (12345,) * 3,
                ):
                    t.fill_(value)
        for _ in range(repeats):
            for name in names:
                code = launch(probes, current_case, spec, name, descriptors)
                if code:
                    raise RuntimeError(f"{name} softcap launch failed: {code}")
            torch.npu.synchronize()
            wave.append(
                {
                    name: (
                        [t[:count].cpu() for t in current_case["raw"][name]]
                        if rep == "prefill"
                        else partial.raw_results(current_case, name)
                    )
                    for name in names
                }
            )
        for name in names:
            for dest, value in zip(chunks[name], wave[-1][name]):
                dest.append(value)
        if rep == "prefill" and spec["splits"] == 1:
            for name in names:
                code = launch(probes, current_case, spec, name, descriptors, direct=True)
                if code:
                    raise RuntimeError(f"{name} direct launch failed: {code}")
            torch.npu.synchronize()
            direct.append(
                dict(
                    first=first,
                    count=count,
                    outputs={
                        name: [t.cpu() for t in current_case["outputs"][name]] for name in names
                    },
                )
            )
        waves.append(wave)
        if rep == "compressed":
            partial.assert_guards(current_case)
        first += count
    raw = {name: [torch.cat(t) for t in tensors] for name, tensors in chunks.items()}
    # The unchanged FD allocation helper requires both sets of partials. With
    # softcap there is no source attention oracle: only migrated FD is launched.
    merge_raw = raw if "oracle" in raw else dict(raw, oracle=raw["migrated"])
    if rep == "prefill":
        merged = prefill.prepare_merge(case, merge_raw)

        def merger(name):
            return prefill.launch_merge(probes["libs"], merged, name, softcap=spec["softcap"])
    else:
        merged = merge.allocate_case(dict(dtype=probes["torch_dtype"]), spec, tasks, merge_raw)

        def merger(name):
            return merge.launch(dict(libs=probes["libs"]), merged, name, softcap=spec["softcap"])

    merge_runs = []
    for _ in range(repeats):
        for name in names:
            code = merger(name)
            if code:
                raise RuntimeError(f"{name} FD merge failed: {code}")
        torch.npu.synchronize()
        merge_runs.append({name: [t.cpu() for t in merged["outputs"][name]] for name in names})
    return dict(
        raw=raw,
        waves=waves,
        direct=direct,
        merged=merge_runs,
        accum=[t.cpu() if t is not None else None for t in merged["accum"]],
        guards=merged.get("storage", []),
        input_guards=case.get("input_guards", []),
        attention_guards=case.get("guards", []),
        tasks=tasks,
    )
