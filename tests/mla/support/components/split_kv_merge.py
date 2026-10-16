"""Shared helpers extracted from tests/mla/unit/split_kv_merge/test_merge.py.

Independent FD, source partials, raw-input math and public accumulator layout.
"""

import ctypes as ct
import json
import os
from pathlib import Path
import pytest
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.mla_ref import normalize_fa3_window
from tests.mla.support.mla_split_kv_merge_build import build_merge
from tests.mla.support.mla_split_kv_attention_reference import fragments
from tests.mla.support.components import split_kv_attention as partial


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
        + [(name, ct.c_uint32) for name in ("adapt", "query_length", "kv_heads", "kv_head")]
    )

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("kv_heads", 1)
        super().__init__(*args, **kwargs)


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


def launch(probes, case, name, *, canonical=False, softcap=0.0):
    for group in case["groups"]:
        p = Params.from_buffer_copy(group["params"])
        if name == "oracle" or canonical:
            p.adapt = 0
        tensors = [
            *group["device"]["oracle" if canonical else name],
            *case["outputs"][name],
            *case["accum"],
        ]
        library = probes["libs"][name]
        function = library.launch_merge if softcap == 0 else library.launch_merge_softcap
        status = function(
            torch.npu.current_stream().npu_stream,
            *(t.data_ptr() if t is not None else 0 for t in tensors),
            ct.byref(p),
            *(() if softcap == 0 else (softcap,)),
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
