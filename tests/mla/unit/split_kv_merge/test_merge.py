"""Independent FD, source partials, raw-input math and public accumulator layout."""

import json
import resource
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.compare import assert_fa_close
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.support.mla_split_kv_merge_cases import (
    CHECK_CASES,
    COST_CASES,
    BOUNDARY_CASES,
    SYNTHETIC_CASES,
)
from tests.mla.support.mla_split_kv_merge_reference import merge_references
from tests.mla.support.mla_split_kv_attention_reference import fragments, references

from tests.mla.support.components.split_kv_merge import (
    Params as Params,
    allocate_case as allocate_case,
    check_guards as check_guards,
    generate_partials as generate_partials,
    guarded as guarded,
    launch as launch,
    make_case as make_case,
    make_params as make_params,
    merge_probes as merge_probes,
)


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
