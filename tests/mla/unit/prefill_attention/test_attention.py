"""192/128 source instructions, raw-input math, masks and independent K/V."""

import ctypes as ct
import json
import os
from pathlib import Path
import resource
import time

import pytest
import torch

from tests.common.compare import assert_fa_close
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_prefill_cases import CASES, REPRESENTATIVE_IDS
from tests.mla.support.mla_prefill_reference import references
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.support.components.prefill_attention import (
    Params,
    load,
    make_case,
    wave,
    launch_case,
    merge_case,
)


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def probes(request, tmp_path_factory):
    torch.npu.set_device(0)
    assert "910B" in torch.npu.get_device_name(0)
    root = Path(os.environ.get("MLA_PREFILL_DIR", tmp_path_factory.mktemp("prefill")))
    return load(request.param, root), request.param


def check_guards(storage):
    for before, after, _ in storage:
        after = after.cpu()
        assert torch.equal(before[:128], after[:128])
        assert torch.equal(before[-128:], after[-128:])


def check_case(probes, spec):
    started = time.monotonic()
    (libs, directory), dtype = probes
    case = make_case(spec, dtype, libs["cores"])
    refs = references(case["inputs"], spec)
    batch, sq, heads, sk, kv_heads = spec["shape"]
    g = heads // kv_heads
    chunks = {name: [[], [], []] for name in ("oracle", "migrated")}
    comparisons = []
    direct = []
    masked = 0
    for first in range(0, len(case["tasks"]), case["capacity"]):
        tasks = wave(case, first)
        cores = case["params"].cores
        # Native skips empty fragments. Canonical initialization is part of
        # the harness; the migrated fragment must write every valid row.
        for name in chunks:
            for tensor, value in zip(
                case["raw"][name], (0, 0, -torch.inf) if name == "oracle" else (12345,) * 3
            ):
                tensor.fill_(value)
        previous = None
        for _ in range(3):
            for name in chunks:
                assert launch_case(libs, case, name, tasks) == 0
            torch.npu.synchronize()
            current = {name: [x[:cores].cpu() for x in case["raw"][name]] for name in chunks}
            if previous is not None:
                for actual, last in zip(current["migrated"], previous):
                    torch.testing.assert_close(actual, last, rtol=0, atol=0, equal_nan=True)
            previous = current["migrated"]
        for name in chunks:
            for dest, value in zip(chunks[name], current[name]):
                dest.append(value)
        for core in range(cores):
            i = first + core
            (_, _, _, begin, end), row, visible = case["tasks"][i]
            count = visible.shape[0]
            valid = visible.any(-1)
            masked += int((~valid).sum())
            actual = [t[core, :count] for t in current["migrated"]]
            source = [t[core, :count] for t in current["oracle"]]
            for a, b in zip(actual, source):
                torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
            out, sums, maxima = actual
            assert torch.equal(sums, sums[:, :1].expand_as(sums))
            assert torch.equal(maxima, maxima[:, :1].expand_as(maxima))
            if begin == end:
                assert (out == 0).all() and (sums == 0).all() and torch.isneginf(maxima).all()
            if valid.any():
                values = (out, maxima[:, 0] + sums[:, 0].log(), maxima[:, 0], sums[:, 0])
                natives = (
                    source[0],
                    source[2][:, 0] + source[1][:, 0].log(),
                    source[2][:, 0],
                    source[1][:, 0],
                )
                for key, a, b in zip(("out", "lse", "max", "sum"), values, natives):
                    comparisons.append(
                        assert_mla_source_close(
                            a[valid],
                            b[valid],
                            refs[key + "_high"][i, :count][valid],
                            refs[key + "_low"][i, :count][valid],
                            name="partial-" + key,
                        )
                    )
        if spec["splits"] == 1:
            for name in chunks:
                assert launch_case(libs, case, name, tasks, partial=False) == 0
            torch.npu.synchronize()
            outputs = {name: [t.cpu() for t in case["outputs"][name]] for name in chunks}
            direct.append(outputs)
            for core in range(cores):
                i = first + core
                (virtual, _, _, _, _), row, mask = case["tasks"][i]
                b, h = divmod(virtual, kv_heads)
                count = mask.shape[0]
                selection = slice(row, row + count)
                empty = ~mask.any(-1)
                gathered = {
                    name: (
                        values[0][b, :, h * g : (h + 1) * g].reshape(-1, 128)[selection],
                        values[1][b, h * g : (h + 1) * g].T.reshape(-1)[selection].clone(),
                    )
                    for name, values in outputs.items()
                }
                assert (gathered["migrated"][0][empty] == 0).all()
                assert torch.isneginf(gathered["migrated"][1][empty]).all()
                assert (gathered["oracle"][0][empty] == 0).all()
                assert torch.isposinf(gathered["oracle"][1][empty]).all()
                gathered["oracle"][1][empty] = -torch.inf
                for index, key in enumerate(("full_out", "full_lse")):
                    high, low = (refs[key + "_" + p][i, :count] for p in ("high", "low"))
                    if index == 0:
                        high, low = high.to(case["dtype"]), low.to(case["dtype"])
                    comparisons.append(
                        assert_mla_source_close(
                            gathered["migrated"][index],
                            gathered["oracle"][index],
                            high,
                            low,
                            name="direct-" + key,
                        )
                    )
        check_guards(case["guards"])
    raw = {name: [torch.cat(x) for x in values] for name, values in chunks.items()}
    merged, accum, merge_guards = merge_case(libs, case, raw)
    for i in range(0, len(case["tasks"]), spec["splits"]):
        (virtual, _, _, _, _), row, mask = case["tasks"][i]
        count = mask.shape[0]
        selection = slice(row, row + count)
        valid = torch.stack([v.any(-1) for _, _, v in case["tasks"][i : i + spec["splits"]]])
        empty = ~valid.any(0)
        gathered = {
            name: (
                values[0][virtual].reshape(-1, 128)[selection],
                values[1][virtual].T.reshape(-1)[selection].clone(),
            )
            for name, values in merged.items()
        }
        assert (gathered["migrated"][0][empty] == 0).all()
        assert torch.isneginf(gathered["migrated"][1][empty]).all()
        if spec["splits"] <= 192:
            assert (gathered["oracle"][0][empty] == 0).all()
            assert (gathered["oracle"][1][empty] == -torch.finfo(torch.float32).max).all()
            gathered["oracle"][1][empty] = -torch.inf
        for index, key in enumerate(("full_out", "full_lse")):
            high, low = (refs[key + "_" + p][i, :count] for p in ("high", "low"))
            if index == 0:
                high, low = high.to(case["dtype"]), low.to(case["dtype"])
            if spec["splits"] <= 192:
                comparisons.append(
                    assert_mla_source_close(
                        gathered["migrated"][index],
                        gathered["oracle"][index],
                        high,
                        low,
                        name="merged-" + key,
                    )
                )
            else:
                assert_fa_close(gathered["migrated"][index], high, low, name="extended-" + key)
                comparisons.append(dict(source="outside-native-buffer-domain", math="passed"))
        if spec["splits"] > 1:
            for s in range(spec["splits"]):
                expected = raw["migrated"][0][i + s, :count].clone()
                expected[~valid[s]] = 0
                actual = accum[0][s, virtual].transpose(0, 1).reshape(-1, 128)[selection]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual = accum[1][s, virtual].T.reshape(-1)[selection]
                assert torch.isneginf(actual[~valid[s]]).all()
                if valid[s].any():
                    src = (
                        raw["oracle"][2][i + s, :count, 0]
                        + raw["oracle"][1][i + s, :count, 0].log()
                    )
                    # Device Log and CPU Log may round differently; accumulator
                    # LSE uses the unchanged mathematical threshold.
                    assert_fa_close(
                        actual[valid[s]], src[valid[s]], src[valid[s]], name="accumulator-lse"
                    )
    check_guards(merge_guards)
    for before, after in case["input_guards"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    directory = directory / "accuracy"
    directory.mkdir(exist_ok=True)
    evidence = directory / (spec["id"] + ".pt")
    torch.save(
        dict(
            inputs=case["inputs"],
            references=refs,
            raw=raw,
            direct=direct,
            merged=merged,
            accum=accum,
        ),
        evidence,
    )
    report = dict(
        config=spec,
        dtype=dtype,
        seconds=time.monotonic() - started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_masked_rows=masked,
        comparisons=comparisons,
        raw_sha256=sha256(evidence),
    )
    evidence.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize(
    "spec", [s for s in CASES if s["id"] in REPRESENTATIVE_IDS], ids=lambda s: s["id"]
)
def test_representative(probes, spec):
    check_case(probes, spec)


@pytest.mark.parametrize(
    "spec", [s for s in CASES if s["id"] not in REPRESENTATIVE_IDS], ids=lambda s: s["id"]
)
def test_boundaries(probes, spec):
    check_case(probes, spec)


def test_invalid_fragment_is_atomic(probes):
    (libs, _), dtype = probes
    case = make_case(next(s for s in CASES if s["id"] == "gqa-stride192"), dtype, libs["cores"])
    tasks = wave(case, 0)
    original = bytes(case["params"])
    for field, value in (
        ("cores", 0),
        ("cores", 27),
        ("kv_heads", 0),
        ("heads", 7),
        ("m", 0),
        ("m", 513),
        ("n", 0),
        ("n", 8193),
        ("scale", float("nan")),
    ):
        case["params"] = Params.from_buffer_copy(original)
        setattr(case["params"], field, value)
        assert launch_case(libs, case, "migrated", tasks) == -1
    case["params"] = Params.from_buffer_copy(original)
    original = bytes(tasks)
    for field, value in (
        ("batch", case["params"].batch),
        ("head", case["params"].kv_heads),
        ("row", 1),
        ("rows", 0),
        ("rows", 129),
        ("kv_length", 32),
        ("begin", 2),
        ("end", 2),
    ):
        ct.memmove(ct.byref(tasks), original, len(original))
        setattr(tasks[0], field, value)
        assert launch_case(libs, case, "migrated", tasks) == -1
    torch.npu.synchronize()
    assert all(
        (t.cpu() == 12345).all() for t in (*case["outputs"]["migrated"], *case["raw"]["migrated"])
    )
    check_guards(case["guards"])
