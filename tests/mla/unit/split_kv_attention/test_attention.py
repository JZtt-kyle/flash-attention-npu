"""Independent BSND FD partial buffers, pinned source and two CPU equations."""

import ctypes as ct
import json
import os
from pathlib import Path
import resource
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_dense_build import ROOT, build_probes, sha256
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.support.mla_split_kv_attention_cases import (
    CHECK_CASES,
    COST_CASES,
    BOUNDARY_CASES,
    EMPTY_CASES,
)
from tests.mla.support.mla_split_kv_attention_reference import fragments, references
from tests.mla.support.components.dense import Schedule

from tests.mla.support.components.split_kv_attention import (
    Partial as Partial,
    Task as Task,
    assert_guards as assert_guards,
    launch as launch,
    make_case as make_case,
    prepare as prepare,
    raw_results as raw_results,
)


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def partial_attention(tmp_path_factory, request):
    base = Path(
        os.environ.get("MLA_SPLIT_KV_ATTENTION_DIR", tmp_path_factory.mktemp("mla-partial"))
    )
    directory = base / request.param
    libraries = build_probes(directory, request.param, split_kv=True)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["split_kv"] and manifest["dtype"] == request.param
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in manifest["dependencies"].items():
        assert sha256(Path(name)) == digest, name
    for record in manifest["libraries"].values():
        assert sha256(Path(record["path"])) == record["sha256"]
    libs = {name: ct.CDLL(path) for name, path in libraries.items()}
    for name in ("oracle", "migrated"):
        libs[name].launch_partial.argtypes = [ct.c_void_p] * 10 + [ct.c_float] + [ct.c_void_p] * 3
        libs[name].launch_partial.restype = ct.c_int
    libs["schedule"].prepare_partial.argtypes = (
        [ct.c_int64] * 4
        + [ct.c_void_p]
        + [ct.c_int64] * 4
        + [ct.c_uint64]
        + [ct.c_int64] * 3
        + [ct.c_void_p] * 3
    )
    libs["schedule"].prepare_partial.restype = ct.c_int
    libs["dtype"] = torch.float16 if request.param == "fp16" else torch.bfloat16
    libs["mask"] = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu()
    (directory / "loaded.json").write_text(json.dumps(manifest["libraries"], indent=2) + "\n")
    return libs, directory


def check_case(probes, spec, repeats=3):
    started = time.monotonic()
    first_case = make_case(probes, spec)
    s = first_case["schedule"]
    expected = list(fragments(spec, s.m_base, s.s2_base))
    assert len(expected) == first_case["partial"].total
    refs = references(first_case["inputs"], spec, s.m_base, s.s2_base)
    rows = spec["shape"][1] * spec["shape"][2]
    evidence = {"inputs": first_case["inputs"], "references": refs, "waves": []}
    comparisons, first = [], 0
    masked_source_rows = 0
    while first < len(expected):
        case = first_case if first == 0 else make_case(probes, spec, first)
        s, p = case["schedule"], case["partial"]
        for core in range(s.cores):
            task = p.tasks[core]
            assert (
                tuple(getattr(task, key) for key in ("batch", "m", "split", "begin", "end"))
                == expected[first + core][0]
            )
        previous = None
        for _ in range(repeats):
            for name in ("oracle", "migrated"):
                assert launch(probes, case, name) == 0
                torch.npu.synchronize()
            results = {name: raw_results(case, name) for name in ("oracle", "migrated")}
            if previous is not None:
                for a, b in zip(results["migrated"], previous):
                    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
            previous = results["migrated"]
        for core in range(s.cores):
            i = first + core
            (_, m, _, begin, end), _, visible = expected[i]
            count = min(s.m_base, rows - m * s.m_base)
            for actual, source in zip(results["migrated"], results["oracle"]):
                torch.testing.assert_close(
                    actual[core, :count], source[core, :count], rtol=0, atol=0, equal_nan=True
                )
            out, sums, maxima = [t[core, :count] for t in results["migrated"]]
            if begin == end:
                assert torch.equal(out, torch.zeros_like(out))
                assert torch.equal(sums, torch.zeros_like(sums)) and torch.isneginf(maxima).all()
                continue
            assert torch.equal(sums, sums[:, :1].expand_as(sums))
            assert torch.equal(maxima, maxima[:, :1].expand_as(maxima))
            valid = visible.any(-1)
            masked_source_rows += int((~valid).sum())
            if not valid.any():
                continue
            source = results["oracle"][0][core, :count][valid]
            comparisons.append(
                assert_mla_source_close(
                    out[valid],
                    source,
                    refs[0][i, :count][valid],
                    refs[4][i, :count][valid],
                    name="partial-out",
                )
            )
            # Statistics are raw sum/max, not LSE. Conversion here is a test
            # equation, outside Kernel timing and outside any public interface.
            lse = maxima[valid, 0] + torch.log(sums[valid, 0])
            source_lse = results["oracle"][2][core, :count][valid, 0] + torch.log(
                results["oracle"][1][core, :count][valid, 0]
            )
            comparisons.append(
                assert_mla_source_close(
                    lse,
                    source_lse,
                    refs[1][i, :count][valid],
                    refs[5][i, :count][valid],
                    name="partial-lse-from-source-equal-stats",
                )
            )
            comparisons.append(
                assert_mla_source_close(
                    maxima[valid, 0],
                    results["oracle"][2][core, :count][valid, 0],
                    refs[2][i, :count][valid],
                    refs[6][i, :count][valid],
                    name="partial-max",
                )
            )
            comparisons.append(
                assert_mla_source_close(
                    sums[valid, 0],
                    results["oracle"][1][core, :count][valid, 0],
                    refs[3][i, :count][valid],
                    refs[7][i, :count][valid],
                    name="partial-sum",
                )
            )
        assert_guards(case)
        evidence["waves"].append(
            dict(first=first, tasks=[expected[first + j][0] for j in range(s.cores)], raw=results)
        )
        first += s.cores
    output = probes[1] / "accuracy"
    output.mkdir(exist_ok=True)
    torch.save(evidence, output / f"{spec['id']}.pt")
    report = dict(
        config=spec,
        dtype=str(probes[0]["dtype"]),
        fragments=len(expected),
        seconds=time.monotonic() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_masked_rows=masked_source_rows,
        comparisons=comparisons,
        raw_sha256=sha256(output / f"{spec['id']}.pt"),
    )
    (output / f"{spec['id']}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@pytest.mark.parametrize("spec", CHECK_CASES, ids=lambda s: s["id"])
def test_representative(partial_attention, spec):
    check_case(partial_attention, spec)


@pytest.mark.parametrize("spec", COST_CASES, ids=lambda s: s["id"])
def test_cost_boundary(partial_attention, spec):
    check_case(partial_attention, spec)


@pytest.mark.parametrize("spec", BOUNDARY_CASES + EMPTY_CASES, ids=lambda s: s["id"])
def test_partial_boundaries(partial_attention, spec):
    check_case(partial_attention, spec)


def test_invalid_host_fragments_are_atomic(partial_attention):
    host = partial_attention[0]["schedule"]
    # The fragment adapter is the only new host bridge. Existing M10a layout
    # and window algorithms retain their own unit-test ownership.
    valid = [1, 3, 8, 1031, None, 2, 16, 512, 2, 0, -1, -1, 0]
    for field, value in (
        (0, 0),
        (1, 0),
        (2, 0),
        (3, -1),
        (5, 27),
        (6, 257),
        (7, 513),
        (8, 0),
        (8, 257),
        (9, 1000),
        (12, 2),
        (4, -1),
        (4, 1032),
    ):
        args = valid.copy()
        raw = (ct.c_int64 * 1)(value if field == 4 else 1031)
        if field != 4:
            args[field] = value
        args[4] = raw
        schedule, partial, details = Schedule(), Partial(), (ct.c_uint64 * 16)()
        for obj in (schedule, partial, details):
            ct.memset(ct.byref(obj), 0xA5, ct.sizeof(obj))
        previous = tuple(bytes(obj) for obj in (schedule, partial, details))
        assert host.prepare_partial(*args, ct.byref(schedule), ct.byref(partial), details) != 0
        assert tuple(bytes(obj) for obj in (schedule, partial, details)) == previous
