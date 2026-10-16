"""Softcap has no same-parameter CANN oracle: nonzero caps must pass dual math."""

import json
import resource
import time

import pytest
import torch

from tests.common.compare import assert_fa_close
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.support.mla_softcap_cases import cases, REPRESENTATIVES
from tests.mla.support.components import softcap
from tests.mla.support.components import prefill_attention as prefill


def check_case(probes, spec):
    start = time.monotonic()
    case = softcap.make_case(probes, spec)
    refs = softcap.reference(probes, case, spec)
    result = softcap.execute(probes, spec, case)
    # Preserve inputs, both equations and device results even if an assertion
    # fails. A .pt file alone is not a passing test or a reusable driver result.
    directory = probes["directory"] / "accuracy"
    directory.mkdir(exist_ok=True)
    evidence = directory / (spec["id"] + ".pt")
    torch.save(
        dict(
            inputs=case["inputs"],
            references=refs,
            **{k: result[k] for k in ("raw", "direct", "merged", "accum")},
        ),
        evidence,
    )
    rep = probes["representation"]
    width = 128 if rep == "prefill" else 512
    dtype = case["inputs"]["q"].dtype
    comparisons = []

    def compare(actual, high, low, name, source=None):
        if spec["softcap"] == 0 and source is not None:
            comparisons.append(assert_mla_source_close(actual, source, high, low, name=name))
        else:
            assert_fa_close(actual, high, low, softcap=spec["softcap"], name=name)
            comparisons.append(dict(math="passed", source="not_applicable", name=name))

    for wave in result["waves"]:
        for repeat in wave[1:]:
            for value, previous in zip(repeat["migrated"], wave[0]["migrated"]):
                torch.testing.assert_close(value, previous, rtol=0, atol=0, equal_nan=True)
    for repeat in result["merged"][1:]:
        for value, previous in zip(repeat["migrated"], result["merged"][0]["migrated"]):
            torch.testing.assert_close(value, previous, rtol=0, atol=0)
    raw = result["raw"]
    for i, ((virtual, _, split, begin, end), row, visible) in enumerate(result["tasks"]):
        count = visible.shape[0]
        valid = visible.any(-1)
        out, sums, maxima = [t[i, :count] for t in raw["migrated"]]
        if "oracle" in raw:
            for actual, source in zip(raw["migrated"], raw["oracle"]):
                torch.testing.assert_close(
                    actual[i, :count], source[i, :count], rtol=0, atol=0, equal_nan=True
                )
        if begin == end:
            assert (out == 0).all() and (sums == 0).all() and torch.isneginf(maxima).all()
        if valid.any():
            assert torch.equal(sums[valid], sums[valid, :1].expand_as(sums[valid]))
            assert torch.equal(maxima[valid], maxima[valid, :1].expand_as(maxima[valid]))
            values = (out, maxima[:, 0] + sums[:, 0].log(), maxima[:, 0], sums[:, 0])
            originals = None
            if "oracle" in raw:
                o, s, m = [t[i, :count] for t in raw["oracle"]]
                originals = (o, m[:, 0] + s[:, 0].log(), m[:, 0], s[:, 0])
            for j, key in enumerate(("out", "lse", "max", "sum")):
                compare(
                    values[j][valid],
                    refs[key + "_high"][i, :count][valid],
                    refs[key + "_low"][i, :count][valid],
                    "partial-" + key,
                    None if originals is None else originals[j][valid],
                )
        if spec["splits"] > 1:
            actual = (
                result["accum"][0][split, virtual]
                .transpose(0, 1)
                .reshape(-1, width)[row : row + count]
            )
            expected = out.clone()
            expected[~valid] = 0
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual_lse = result["accum"][1][split, virtual].T.reshape(-1)[row : row + count]
            assert torch.isneginf(actual_lse[~valid]).all()
            if valid.any():
                compare(
                    actual_lse[valid],
                    refs["lse_high"][i, :count][valid],
                    refs["lse_low"][i, :count][valid],
                    "accumulator-lse",
                    # Source FD has no public accumulator. Require math
                    # here even for cap=0; never use a source exemption.
                    None,
                )
        if split:
            continue
        valid_rows = torch.stack(
            [v.any(-1) for _, _, v in result["tasks"][i : i + spec["splits"]]]
        ).any(0)
        empty = ~valid_rows
        selected = {}
        for name, tensors in result["merged"][-1].items():
            selected[name] = [
                tensors[0][virtual].reshape(-1, width)[row : row + count],
                tensors[1][virtual].T.reshape(-1)[row : row + count].clone(),
            ]
        assert (selected["migrated"][0][empty] == 0).all()
        assert torch.isneginf(selected["migrated"][1][empty]).all()
        if "oracle" in selected:
            assert (selected["oracle"][0][empty] == 0).all()
            assert (selected["oracle"][1][empty] == -torch.finfo(torch.float32).max).all()
            selected["oracle"][1][empty] = -torch.inf
        for j, key in enumerate(("full_out", "full_lse")):
            high, low = [refs[key + "_" + p][i, :count] for p in ("high", "low")]
            if j == 0:
                high, low = high.to(dtype), low.to(dtype)
            compare(
                selected["migrated"][j],
                high,
                low,
                "merged-" + key,
                selected["oracle"][j] if "oracle" in selected else None,
            )
            if j == 0 and spec.get("pattern") == "rounding-ties" and spec["softcap"]:
                torch.testing.assert_close(selected["migrated"][j], high, rtol=0, atol=0)
    for direct in result["direct"]:
        kv_heads, g = spec["shape"][-1], spec["shape"][2] // spec["shape"][-1]
        for i in range(direct["first"], direct["first"] + direct["count"]):
            (virtual, _, _, _, _), row, visible = result["tasks"][i]
            b, h = divmod(virtual, kv_heads)
            count, empty = visible.shape[0], ~visible.any(-1)
            gathered = {}
            for name, (out, lse) in direct["outputs"].items():
                gathered[name] = [
                    out[b, :, h * g : (h + 1) * g].reshape(-1, width)[row : row + count],
                    lse[b, h * g : (h + 1) * g].T.reshape(-1)[row : row + count].clone(),
                ]
            assert (gathered["migrated"][0][empty] == 0).all()
            assert torch.isneginf(gathered["migrated"][1][empty]).all()
            if "oracle" in gathered:
                assert torch.isposinf(gathered["oracle"][1][empty]).all()
                gathered["oracle"][1][empty] = -torch.inf
            for j, key in enumerate(("full_out", "full_lse")):
                high, low = [refs[key + "_" + p][i, :count] for p in ("high", "low")]
                if j == 0:
                    high, low = high.to(dtype), low.to(dtype)
                compare(
                    gathered["migrated"][j],
                    high,
                    low,
                    "direct-" + key,
                    gathered["oracle"][j] if "oracle" in gathered else None,
                )
                if j == 0 and spec.get("pattern") == "rounding-ties" and spec["softcap"]:
                    torch.testing.assert_close(gathered["migrated"][j], high, rtol=0, atol=0)
    for before, after, _ in result["guards"] + result["attention_guards"]:
        after = after.cpu()
        assert torch.equal(before[:128], after[:128]) and torch.equal(before[-128:], after[-128:])
    for before, after in result["input_guards"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    report = dict(
        config=spec,
        representation=rep,
        dtype=probes["dtype"],
        comparisons=comparisons,
        raw_sha256=sha256(evidence),
        seconds=time.monotonic() - start,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    evidence.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


ROWS = [(r, s) for r in ("compressed", "prefill") for s in cases(r)]


@pytest.mark.parametrize(
    "representation,spec",
    [(r, s) for r, s in ROWS if s["id"] in REPRESENTATIVES],
    ids=[r + "-" + s["id"] for r, s in ROWS if s["id"] in REPRESENTATIVES],
    scope="module",
)
def test_representative(probes, spec):
    check_case(probes, spec)


@pytest.mark.parametrize(
    "representation,spec",
    [(r, s) for r, s in ROWS if s["id"] not in REPRESENTATIVES],
    ids=[r + "-" + s["id"] for r, s in ROWS if s["id"] not in REPRESENTATIVES],
    scope="module",
)
def test_boundaries(probes, spec):
    check_case(probes, spec)


@pytest.mark.parametrize("representation", ("compressed", "prefill"), scope="module")
@pytest.mark.parametrize("cap", (-1.0, float("nan"), float("inf"), -float("inf")))
def test_invalid_cap_is_atomic(probes, cap):
    spec = next(s for s in cases(probes["representation"]) if s["id"] == "bypass-none")
    case = softcap.make_case(probes, spec)
    tasks = prefill.wave(case, 0) if probes["representation"] == "prefill" else None
    tensors = (
        [*case["outputs"]["migrated"], *case["raw"]["migrated"]]
        if tasks is not None
        else [case["workspace"]["migrated"]]
    )
    before = [t.cpu() for t in tensors]
    assert softcap.launch(probes, case, spec, "migrated", tasks, cap=cap) == -1
    torch.npu.synchronize()
    for actual, old in zip(tensors, before):
        torch.testing.assert_close(actual.cpu(), old, rtol=0, atol=0)


@pytest.mark.parametrize("representation", ("compressed", "prefill"), scope="module")
def test_source_rejects_nonzero_cap(probes):
    spec = next(s for s in cases(probes["representation"]) if s["id"] == "softcap-single")
    case = softcap.make_case(probes, spec)
    tasks = prefill.wave(case, 0) if probes["representation"] == "prefill" else None
    assert softcap.launch(probes, case, spec, "oracle", tasks) == -1
