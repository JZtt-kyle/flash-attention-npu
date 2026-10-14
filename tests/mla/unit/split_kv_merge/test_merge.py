"""Independent FD, source partials, raw-input math and public accumulator layout."""

import ctypes as ct
import json
import os
from pathlib import Path
import resource
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.compare import assert_fa_close
from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.mla_ref import normalize_fa3_window
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.support.mla_split_kv_merge_build import build_merge
from tests.mla.support.mla_split_kv_merge_cases import (
    CHECK_CASES,
    COST_CASES,
    BOUNDARY_CASES,
    SYNTHETIC_CASES,
)
from tests.mla.support.mla_split_kv_merge_reference import merge_references
from tests.mla.support.mla_split_kv_attention_reference import fragments, references
from tests.mla.unit.split_kv_attention import test_attention as partial


class Params(ct.Structure):
    _fields_ = (
        [
            (name, ct.c_uint32)
            for name in (
                "batch",
                "sq",
                "heads",
                "batch_index",
                "row_start",
                "rows",
                "m",
                "splits",
                "kv_length",
                "mask_kind",
            )
        ]
        + [(name, ct.c_int64) for name in ("left", "right")]
        + [(name, ct.c_uint16 * 256) for name in ("begin", "end")]
        + [("adapt", ct.c_uint32)]
    )


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def merge_probes(tmp_path_factory, request):
    torch.npu.set_device(0)
    assert "910B" in torch.npu.get_device_name(0)
    directory = (
        Path(os.environ.get("MLA_SPLIT_KV_MERGE_DIR", tmp_path_factory.mktemp("mla-merge")))
        / request.param
    )
    paths = build_merge(directory, request.param)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest
    for name, digest in manifest["dependencies"].items():
        assert sha256(Path(name)) == digest
    libraries = {name: ct.CDLL(path) for name, path in paths.items()}
    for name in ("oracle", "migrated"):
        libraries[name].launch_merge.argtypes = [ct.c_void_p] * 9
        libraries[name].launch_merge.restype = ct.c_int
    pd = {name: libraries[f"partial_{name}"] for name in ("oracle", "migrated", "schedule")}
    for name in ("oracle", "migrated"):
        pd[name].launch_partial.argtypes = [ct.c_void_p] * 10 + [ct.c_float] + [ct.c_void_p] * 3
        pd[name].launch_partial.restype = ct.c_int
    pd["schedule"].prepare_partial.argtypes = (
        [ct.c_int64] * 4
        + [ct.c_void_p]
        + [ct.c_int64] * 4
        + [ct.c_uint64]
        + [ct.c_int64] * 3
        + [ct.c_void_p] * 3
    )
    pd["schedule"].prepare_partial.restype = ct.c_int
    dtype = torch.float16 if request.param == "fp16" else torch.bfloat16
    pd["dtype"] = dtype
    pd["mask"] = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu()
    for record in manifest["libraries"].values():
        assert sha256(Path(record["path"])) == record["sha256"]
    (directory / "loaded.json").write_text(json.dumps(manifest["libraries"], indent=2) + "\n")
    return dict(
        libs=libraries, partial=(pd, directory / "partial"), directory=directory, dtype=dtype
    )


def guarded(tensor, storage):
    before = torch.full((tensor.numel() + 256,), 117, dtype=tensor.dtype)
    before[128:-128] = tensor.flatten()
    device = before.npu()
    storage.append((before, device, tensor.numel()))
    return device[128:-128].view(tensor.shape)


def generate_partials(probes, spec):
    pp = probes["partial"]
    first_case = partial.make_case(pp, spec)
    tasks = list(fragments(spec, first_case["schedule"].m_base, first_case["schedule"].s2_base))
    chunks = {name: [[], [], []] for name in ("oracle", "migrated")}
    first = 0
    while first < len(tasks):
        c = first_case if not first else partial.make_case(pp, spec, first)
        for name in chunks:
            assert partial.launch(pp, c, name) == 0
        torch.npu.synchronize()
        for name in chunks:
            for dest, value in zip(chunks[name], partial.raw_results(c, name)):
                dest.append(value)
        partial.assert_guards(c)
        first += c["schedule"].cores
    return (
        first_case["inputs"],
        tasks,
        {name: [torch.cat(x) for x in values] for name, values in chunks.items()},
    )


def make_params(spec, tasks, group):
    batch, sq, heads, sk = spec["shape"]
    b, m_block = tasks[group * spec["splits"]][0][:2]
    row = m_block * spec["m"]
    p = Params(
        batch,
        sq,
        heads,
        b,
        row,
        min(spec["m"], sq * heads - row),
        spec["m"],
        spec["splits"],
        spec.get("lengths", [sk] * batch)[b],
    )
    left, right = normalize_fa3_window(
        sq, sk, causal=spec.get("causal", False), window_size=spec.get("window", (-1, -1))
    )
    p.mask_kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
    p.left, p.right = sk - 1 if left < 0 else left, sq - 1 if right < 0 else right
    for s in range(p.splits):
        begin, end = tasks[group * p.splits + s][0][-2:]
        p.begin[s], p.end[s] = (
            min(p.kv_length, begin * spec["n"]),
            min(p.kv_length, end * spec["n"]),
        )
    p.adapt = 1
    return p


def allocate_case(probes, spec, tasks, raw, inputs=None):
    batch, sq, heads, _ = spec["shape"]
    storage = []
    outputs = {
        name: [
            guarded(torch.full((batch, sq, heads, 512), 12345, dtype=probes["dtype"]), storage),
            guarded(torch.full((batch, heads, sq), 12345.0), storage),
        ]
        for name in ("oracle", "migrated")
    }
    acc_shape = (spec["splits"], batch, heads, sq)
    accum = (
        [
            guarded(torch.full((*acc_shape, 512), 12345.0), storage),
            guarded(torch.full(acc_shape, 12345.0), storage),
        ]
        if spec["splits"] > 1
        else [None, None]
    )
    groups = []
    for group in range(len(tasks) // spec["splits"]):
        p = make_params(spec, tasks, group)
        start, end = group * p.splits, (group + 1) * p.splits
        visible = torch.stack([mask.any(-1) for _, _, mask in tasks[start:end]])
        valid = torch.zeros((p.splits, p.m), dtype=torch.bool)
        valid[:, : p.rows] = visible
        for a, b in zip(raw["migrated"], raw["oracle"]):
            torch.testing.assert_close(
                a[start:end, : p.rows], b[start:end, : p.rows], rtol=0, atol=0, equal_nan=True
            )
        original = [value[start:end].clone() for value in raw["migrated"]]
        source = [value[start:end].clone() for value in raw["oracle"]]
        source[0][~valid] = 0
        source[1][~valid] = 0
        source[2][~valid] = -torch.finfo(torch.float32).max
        # A native FD task has contributing fragments. For an entirely empty
        # row supply a neutral zero-O term, then check source's finite sentinel
        # LSE separately from the required FA3 -inf adaptation.
        source[1][0, ~valid.any(0)] = 1
        groups.append(
            dict(
                params=p,
                valid=valid,
                raw=original,
                canonical=source,
                device={
                    "migrated": [guarded(t, storage) for t in original],
                    "oracle": [guarded(t, storage) for t in source],
                },
            )
        )
    return dict(
        spec=spec,
        storage=storage,
        outputs=outputs,
        accum=accum,
        groups=groups,
        inputs=inputs,
        raw=raw,
    )


def make_case(probes, spec):
    inputs, tasks, raw = generate_partials(probes, spec)
    return allocate_case(probes, spec, tasks, raw, inputs)


def launch(probes, case, name, *, canonical=False):
    for group in case["groups"]:
        p = Params.from_buffer_copy(group["params"])
        if name == "oracle" or canonical:
            p.adapt = 0
        tensors = [
            *group["device"]["oracle" if canonical else name],
            *case["outputs"][name],
            *case["accum"],
        ]
        status = probes["libs"][name].launch_merge(
            torch.npu.current_stream().npu_stream,
            *(t.data_ptr() if t is not None else 0 for t in tensors),
            ct.byref(p),
        )
        if status:
            return status
    return 0


def check_guards(case):
    for before, after, size in case["storage"]:
        after = after.cpu()
        assert torch.equal(after[:128], before[:128]) and torch.equal(after[-128:], before[-128:])
    for group in case["groups"]:
        for name, cpu in (("migrated", group["raw"]), ("oracle", group["canonical"])):
            for actual, expected in zip(group["device"][name], cpu):
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)


def check_case(probes, case):
    started = time.monotonic()
    source_supported = case["spec"]["splits"] <= 192
    previous = None
    for _ in range(3):
        assert launch(probes, case, "migrated") == 0
        assert launch(probes, case, "oracle") == (0 if source_supported else -2)
        torch.npu.synchronize()
        result = [t.cpu() for t in case["outputs"]["migrated"]] + [
            t.cpu() for t in case["accum"] if t is not None
        ]
        assert all(not t.isnan().any() for t in result)
        if previous is not None:
            for actual, last in zip(result, previous):
                torch.testing.assert_close(actual, last, rtol=0, atol=0)
        previous = result
    source = [t.cpu() for t in case["outputs"]["oracle"]]
    if not source_supported:
        assert all((t == 12345).all() for t in source)
    comparisons = []
    math_evidence = []
    for group in case["groups"]:
        p, valid = group["params"], group["valid"]
        selection = slice(p.row_start, p.row_start + p.rows)
        o = result[0][p.batch_index].reshape(-1, 512)[selection]
        lse_value = result[1][p.batch_index].T.reshape(-1)[selection]
        refs = merge_references(
            *(t[:, : p.rows] for t in group["raw"]), valid[:, : p.rows], probes["dtype"]
        )
        math_evidence.append(refs)
        src_o = source[0][p.batch_index].reshape(-1, 512)[selection]
        # Source BSND LSE is [B,H,Q], identical to FA3 for this layout.
        src_l = source[1][p.batch_index].T.reshape(-1)[selection].clone()
        empty = ~valid[:, : p.rows].any(0)
        if source_supported:
            assert (src_o[empty] == 0).all()
            assert (src_l[empty] == -torch.finfo(torch.float32).max).all()
            src_l[empty] = -torch.inf
        for name, actual, native, high, low in (
            ("merge-out", o, src_o, refs[0], refs[2]),
            ("merge-lse", lse_value, src_l, refs[1], refs[3]),
        ):
            if source_supported:
                comparisons.append(assert_mla_source_close(actual, native, high, low, name=name))
            else:
                assert_fa_close(actual, high, low, name=name)
                comparisons.append(dict(source="outside-native-buffer-domain", math="passed"))
        if p.splits > 1:
            for s in range(p.splits):
                acc = result[2][s, p.batch_index].transpose(0, 1).reshape(-1, 512)[selection]
                alse = result[3][s, p.batch_index].T.reshape(-1)[selection]
                expected = group["raw"][0][s, : p.rows].clone()
                expected[~valid[s, : p.rows]] = 0
                torch.testing.assert_close(acc, expected, rtol=0, atol=0)
                assert_fa_close(alse, refs[4][s], refs[5][s], name="accumulator-lse")
    full = None
    if case["inputs"] is not None:
        # Raw-input whole-attention math uses one full fragment per M block;
        # normalization uses original allocation maxima, including short KV.
        spec = dict(case["spec"], splits=1)
        full = references(case["inputs"], spec, spec["m"], spec["n"])
        for group_id, group in enumerate(case["groups"]):
            p = group["params"]
            selection = slice(p.row_start, p.row_start + p.rows)
            actual = (
                result[0][p.batch_index].reshape(-1, 512)[selection],
                result[1][p.batch_index].T.reshape(-1)[selection],
            )
            native = (
                source[0][p.batch_index].reshape(-1, 512)[selection],
                source[1][p.batch_index].T.reshape(-1)[selection].clone(),
            )
            native[1][~group["valid"][:, : p.rows].any(0)] = -torch.inf
            for i, name in enumerate(("whole-out", "whole-lse")):
                high, low = full[i][group_id, : p.rows], full[i + 4][group_id, : p.rows]
                if i == 0:
                    high, low = high.to(probes["dtype"]), low.to(probes["dtype"])
                if source_supported:
                    comparisons.append(
                        assert_mla_source_close(actual[i], native[i], high, low, name=name)
                    )
                else:
                    assert_fa_close(actual[i], high, low, name=name)
    check_guards(case)
    directory = probes["directory"] / "accuracy"
    directory.mkdir(exist_ok=True)
    output = directory / (case["spec"]["id"] + ".pt")
    torch.save(
        dict(
            inputs=case["inputs"],
            raw=case["raw"],
            result=result,
            source=source,
            valid=[g["valid"] for g in case["groups"]],
            merge_references=math_evidence,
            whole_references=full,
        ),
        output,
    )
    report = dict(
        config=case["spec"],
        dtype=str(probes["dtype"]),
        source_supported=source_supported,
        comparisons=comparisons,
        seconds=time.monotonic() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        evidence_sha256=sha256(output),
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize("spec", CHECK_CASES, ids=lambda s: s["id"])
def test_representative(merge_probes, spec):
    check_case(merge_probes, make_case(merge_probes, spec))


@pytest.mark.parametrize("spec", COST_CASES, ids=lambda s: s["id"])
def test_cost_boundary(merge_probes, spec):
    check_case(merge_probes, make_case(merge_probes, spec))


@pytest.mark.parametrize("spec", BOUNDARY_CASES, ids=lambda s: s["id"])
def test_boundaries(merge_probes, spec):
    check_case(merge_probes, make_case(merge_probes, spec))


def synthetic_case(probes, spec):
    splits, rows = spec["splits"], spec["rows"]
    gen = torch.Generator().manual_seed(20261014)
    o = torch.randn(splits, rows, 512, generator=gen) * 0.25
    total = torch.ones(splits, rows, 8)
    maximum = torch.zeros_like(total)
    if spec["mode"] == "span":
        maximum += torch.linspace(-80, 80, splits)[:, None, None]
        total *= torch.logspace(-3, 3, splits)[:, None, None]
    if spec["mode"] == "last":
        o.zero_()
        o[-1] = 256
    if spec["mode"] == "sentinel":
        maximum[:] = -2e12
    actual = dict(id=spec["id"], shape=(1, rows, 1, splits), splits=splits, m=rows, n=1)
    tasks = list(fragments(actual, rows, 1))
    if spec["mode"] in ("sentinel", "empty"):
        # Leave one valid term below the source mask sentinel, or no valid terms.
        for i in range(1 if spec["mode"] == "sentinel" else 0, splits):
            task, row, visible = tasks[i]
            tasks[i] = ((*task[:3], splits, splits), row, torch.zeros_like(visible))
            maximum[i] = -1e12 if i % 2 else -torch.inf
            total[i] = 7 if i % 2 else 0
            o[i] = torch.nan
    raw = {name: [o.clone(), total.clone(), maximum.clone()] for name in ("oracle", "migrated")}
    return allocate_case(probes, actual, tasks, raw)


@pytest.mark.parametrize("spec", SYNTHETIC_CASES, ids=lambda s: s["id"])
def test_synthetic_statistics(merge_probes, spec):
    check_case(merge_probes, synthetic_case(merge_probes, spec))


def test_host_failure_atomicity(merge_probes):
    case = synthetic_case(merge_probes, SYNTHETIC_CASES[1])
    group = case["groups"][0]
    original = bytes(group["params"])
    for field, value in (
        ("batch", 0),
        ("batch_index", 1),
        ("heads", 0),
        ("m", 257),
        ("rows", 0),
        ("rows", 10),
        ("splits", 0),
        ("splits", 257),
        ("kv_length", 16385),
        ("mask_kind", 3),
        ("adapt", 2),
        ("row_start", 1),
    ):
        group["params"] = Params.from_buffer_copy(original)
        setattr(group["params"], field, value)
        assert launch(merge_probes, case, "migrated") == -1
    group["params"] = Params.from_buffer_copy(original)
    group["params"].begin[0] = group["params"].end[0] + 1
    assert launch(merge_probes, case, "migrated") == -1
    group["params"] = Params.from_buffer_copy(original)
    group["params"].end[0] = group["params"].kv_length + 1
    assert launch(merge_probes, case, "migrated") == -1
    group["params"] = Params.from_buffer_copy(original)
    group["params"].mask_kind, group["params"].left = 2, 2**63 - 1
    assert launch(merge_probes, case, "migrated") == -1
    group["params"] = Params.from_buffer_copy(original)
    group["params"].sq, group["params"].heads = 131072, 65536
    assert launch(merge_probes, case, "migrated") == -1
    group["params"] = Params.from_buffer_copy(original)
    group["params"].batch, group["params"].sq, group["params"].heads = 65536, 65535, 65536
    group["params"].splits = 256
    assert launch(merge_probes, case, "migrated") == -1
    group["params"] = Params.from_buffer_copy(original)
    acc = case["accum"]
    case["accum"] = [None, None]
    assert launch(merge_probes, case, "migrated") == -1
    case["accum"] = acc
    torch.npu.synchronize()
    assert all((t.cpu() == 12345).all() for t in case["outputs"]["migrated"] + case["accum"])
    check_guards(case)
