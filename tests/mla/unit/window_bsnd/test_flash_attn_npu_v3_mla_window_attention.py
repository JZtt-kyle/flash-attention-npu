"""Independent BSND window component, pinned source and dual CPU math.

The source mask/length arrays are constructed directly from raw case inputs;
no oracle reads the migrated schedule to determine which keys are visible.
"""

import ctypes as ct
import hashlib
import json
import os
import resource
from pathlib import Path
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_dense_build import ROOT, build_probes, sha256
from tests.mla.support.mla_window_attention_cases import (
    CHECK_CASES,
    COST_CASES,
    PERFORMANCE_CASES,
    BOUNDARY_CASES,
    EMPTY_CASES,
)
from tests.mla.support.mla_source_compare import assert_mla_source_close

from tests.mla.support.components.window_attention import (
    WindowParams as WindowParams,
    empty_rows as empty_rows,
    launch as launch,
    make_case as make_case,
    normalized_window as normalized_window,
    references as references,
)


def load_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["window"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in manifest["dependencies"].items():
        assert sha256(Path(name)) == digest, name
    libraries = {}
    for name, record in manifest["libraries"].items():
        assert sha256(Path(record["path"])) == record["sha256"]
        libraries[name] = ct.CDLL(record["path"])
    for name in ("oracle", "migrated"):
        libraries[name].launch_window.argtypes = (
            [ct.c_void_p] * 10 + [ct.c_float] + [ct.c_void_p] * 3
        )
        libraries[name].launch_window.restype = ct.c_int
    libraries["schedule"].prepare_window.argtypes = (
        [ct.c_int64] * 4
        + [ct.c_void_p]
        + [ct.c_int64] * 3
        + [ct.c_void_p] * 2
        + [ct.c_int64] * 5
        + [ct.c_void_p]
    )
    libraries["schedule"].prepare_window.restype = ct.c_int
    libraries["dtype"] = torch.float16 if manifest["dtype"] == "fp16" else torch.bfloat16
    # Sparse modes 3/4 consume this real compressed upper triangle; band also inverts it for the left bound.
    libraries["mask"] = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu()
    (directory / "loaded.json").write_text(json.dumps(manifest["libraries"], indent=2) + "\n")
    return libraries, directory


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def window_attention(tmp_path_factory, request):
    base = Path(os.environ.get("MLA_WINDOW_ATTENTION_DIR", tmp_path_factory.mktemp("mla-window")))
    directory = base / request.param
    build_probes(directory, request.param, window=True)
    return load_attention(directory)


def check_case(probes, case, repeats=3):
    started = time.monotonic()
    first = None
    for _ in range(repeats):
        for name in ("oracle", "migrated"):
            assert launch(probes, case, name) == 0
            try:
                torch.npu.synchronize()
            except RuntimeError as exc:
                raise RuntimeError(f"{name} window launch failed for {case['config']}") from exc
        torch.npu.synchronize()
        results = {
            name: [t.cpu() for t in case["outputs"][name]] for name in ("oracle", "migrated")
        }
        if first is not None:
            for actual, previous in zip(results["migrated"], first):
                torch.testing.assert_close(actual, previous, rtol=0, atol=0)
        first = results["migrated"]
        batch, sq, heads, capacity = case["shape"]
        for b, length in enumerate(case["lengths"]):
            empty = empty_rows(case, length)
            for name in results:
                out, lse = results[name]
                assert not out[b].isnan().any() and not lse[b].isnan().any()
                assert (out[b, :empty] == 0).all()
                expected = torch.inf if name == "oracle" or capacity == 0 else -torch.inf
                assert (lse[b, :, :empty] == expected).all()
            torch.testing.assert_close(first[0][b], results["oracle"][0][b], rtol=0, atol=0)
            torch.testing.assert_close(
                first[1][b, :, empty:], results["oracle"][1][b, :, empty:], rtol=0, atol=0
            )
    # Each independent math call sees just the raw effective sequence. Global
    # empty-capacity and a zero effective KV sample have different LSE signs.
    refs = references(case)
    source = [t.clone() for t in results["oracle"]]
    # Empty-row signs were checked on both raw results above. Normalize only
    # that documented FA3 adaptation for the source/math comparison.
    for b, length in enumerate(case["lengths"]):
        source[1][b, :, : empty_rows(case, length)] = torch.inf if capacity == 0 else -torch.inf
    comparisons = {}
    for i, name in enumerate(("out", "lse")):
        comparisons[name] = assert_mla_source_close(
            first[i], source[i], refs[i], refs[i + 2], name=name
        )
    for i, (before, device, prefix) in enumerate(case["storage"]):
        after = device.cpu()
        if i < 4:
            torch.testing.assert_close(after, before, rtol=0, atol=0, equal_nan=True)
        else:
            assert torch.equal(after[:prefix], before[:prefix])
            assert torch.equal(after[-16:], before[-16:])
    for workspace in case["ws_storage"].values():
        assert (workspace[:512].cpu() == 205).all() and (workspace[-512:].cpu() == 205).all()
    identity = hashlib.sha256(json.dumps(case["config"], sort_keys=True).encode()).hexdigest()[:16]
    torch.save(
        dict(
            inputs=case["inputs"],
            lengths=case["lengths"],
            scale=case["scale"],
            outputs=results,
            references=refs,
        ),
        probes[1] / f"case-{identity}.pt",
    )
    s = case["schedule"]
    record = dict(
        config=case["config"],
        comparisons=comparisons,
        dtype=case["dtype"],
        seconds=time.monotonic() - started,
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        npu_peak_allocated_bytes=torch.npu.max_memory_allocated(),
        reference_workers=min(4, max(1, batch)),
        reference_intraop_threads=torch.get_num_threads(),
        repeats=repeats,
        tile=[s.m_base, s.s2_base],
        cores=s.cores,
        endpoints=list(zip(s.b_end[: s.cores], s.m_end[: s.cores], s.s_end[: s.cores])),
        evidence=f"case-{identity}.pt",
    )
    with (probes[1] / "accuracy.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


@pytest.mark.parametrize("spec", CHECK_CASES)
def test_representative(window_attention, spec):
    check_case(window_attention, make_case(window_attention, spec))


@pytest.mark.parametrize("spec", COST_CASES)
def test_cost_boundary(window_attention, spec):
    check_case(window_attention, make_case(window_attention, spec))


@pytest.mark.parametrize(
    "spec", [s for s in PERFORMANCE_CASES if s not in CHECK_CASES + COST_CASES]
)
def test_performance_geometries(window_attention, spec):
    check_case(window_attention, make_case(window_attention, spec))


@pytest.mark.parametrize("spec", BOUNDARY_CASES)
def test_window_boundaries(window_attention, spec):
    check_case(window_attention, make_case(window_attention, spec))


@pytest.mark.parametrize("spec", EMPTY_CASES)
def test_empty_and_pipeline(window_attention, spec):
    check_case(window_attention, make_case(window_attention, spec), repeats=5)


def test_interleaved_streams(window_attention):
    cases = [make_case(window_attention, s) for s in CHECK_CASES[:2]]
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    torch.npu.synchronize()
    for stream, case in zip(streams, cases):
        with torch.npu.stream(stream):
            for name in ("oracle", "migrated"):
                assert launch(window_attention, case, name) == 0
    torch.npu.synchronize()
    before = [[t.clone() for t in c["outputs"]["migrated"]] for c in cases]
    for case, expected in zip(cases, before):
        check_case(window_attention, case)
        for result, old in zip(case["outputs"]["migrated"], expected):
            torch.testing.assert_close(result, old, rtol=0, atol=0)
