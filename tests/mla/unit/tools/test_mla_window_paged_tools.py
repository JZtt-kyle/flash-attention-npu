"""PA window selection, actual band arguments and timing evidence rejection."""

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_window_paged_cases import (
    PERFORMANCE_CASES,
    REPRESENTATIVES,
    COST_CASES,
    ACCURACY_CASES,
)
from tests.mla.support.mla_public_performance import summarize_mixed
from tests.mla.unit.tools.test_mla_validation_policy import mixed_reports
from tests.mla.unit.tools.test_mla_validation import args
from tests.mla.support.components.window_schedule import normalized


def test_selection_and_environment(tmp_path):
    assert runner.SUITES["window-paged-profile"] == [
        "tests/mla/e2e/window_paged/test_mla_window_paged_profile.py"
    ]
    assert (
        runner.SUITES["window-paged-tools"][0]
        == "tests/mla/unit/tools/test_mla_window_paged_tools.py"
    )
    assert all(s in REPRESENTATIVES + COST_CASES + ACCURACY_CASES for s in PERFORMANCE_CASES)
    assert all(s not in ACCURACY_CASES for s in REPRESENTATIVES + COST_CASES)
    assert all(s not in REPRESENTATIVES for s in COST_CASES)
    assert 2 * len(PERFORMANCE_CASES) <= 100
    for suite in ("window-paged-cpu", "window-paged-sanitize", "window-paged-tools"):
        env = runner.environment(args(tmp_path), tmp_path, suite)
        assert env["MLA_VALIDATION_NPU"] == "0" and env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    for suite in ("window-paged-check", "window-paged-public"):
        env = runner.environment(args(tmp_path), tmp_path, suite)
        assert env["MLA_VALIDATION_NPU"] == "1" and "MLA_PAGED_ATTENTION_DIR" in env
        assert suite not in runner.STANDALONE_CHECK_SUITES
    assert "window-paged-profile" in runner.PERFORMANCE_SUITES
    boundary = runner.SUITES["window-paged-boundary"]
    assert any("causal_paged_tnd_public" in node for node in boundary)
    assert any("paged_metadata_public" in node for node in boundary)
    assert "window_size" in boundary[-1] and "compile_and_length_updates" in boundary[-1]


@pytest.mark.parametrize(
    "bounds,window,causal,mode,left,right",
    [
        ((7, 33), (2, 1), False, 4, 2, 1),
        ((7, 33), (-1, 1), False, 4, 32, 1),
        ((7, 33), (2, -1), False, 4, 2, 6),
        ((7, 33), (2, 1), True, 4, 2, 0),
        ((7, 33), (32, 6), False, 0, 2**31 - 1, 2**31 - 1),
        ((7, 33), (-1, 0), False, 3, 2**31 - 1, 0),
        ((64, 1025), (32, 6), False, 4, 32, 6),
    ],
)
def test_native_band_uses_call_bounds(bounds, window, causal, mode, left, right):
    class Base:
        def __init__(self, *a):
            self.kwargs = dict(sparse_mode=0, pre_tokens=2**31 - 1, next_tokens=2**31 - 1)

    tree = ast.parse(Path("tests/mla/support/mla_window_paged_profile.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WindowPagedCall"]

    def window_for(case):
        return normalized([*bounds, *window, int(causal)])

    namespace = dict(CannPagedCall=Base, window_for=window_for)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "window-paged-native", "exec"), namespace)
    mask = torch.ones((2048, 2048), dtype=torch.bool).triu(1)
    call = namespace["WindowPagedCall"](
        dict(args={}, config=dict(lengths=[2], q=[3]), scale=1 / 24, mask=mask)
    )
    assert call.description() == dict(sparse_mode=mode, pre_tokens=left, next_tokens=right)
    if mode:
        assert call.kwargs["atten_mask"] is mask
        assert not mask[0, 0] and mask[0, 1] and not mask[1, 0]


def window_reports():
    specs, reports = mixed_reports()
    for s in specs:
        s.update(window=[2, 1], causal=False)
    for report in reports:
        report.update(causal=False, window=True)
    return specs, reports


def test_native_input_identity_survives_json_and_detects_changes():
    tree = ast.parse(Path("tests/mla/support/mla_causal_paged_profile.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "identity"]
    namespace = dict(json=json, hashlib=hashlib, torch=torch)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "paged-native-identity", "exec"), namespace
    )
    identity = namespace["identity"]
    query = torch.arange(24, dtype=torch.float16).view(2, 12)[:, 1::2]
    case = dict(host=dict(q=query), table_host=torch.tensor([[1, 0]]), config=dict(causal=True))
    for spec in PERFORMANCE_CASES:
        proof = json.loads(json.dumps(identity(case, spec)))
        assert proof == identity(case, spec)
        assert proof["inputs"]["q"]["stride"] == list(query.stride())
        assert proof["inputs"]["q"]["offset"] == query.storage_offset()
        changed = dict(spec, window=(spec["window"][0] + 1, spec["window"][1]))
        assert proof != identity(case, changed)
    before = identity(case)
    assert before == json.loads(json.dumps(before)) and before["spec"] == {}
    query[0, 0] += 1
    assert before != identity(case)


def test_window_average_and_unchanged_causal():
    specs, reports = window_reports()
    result = summarize_mixed(reports, specs)
    assert result["passed"] and result["regressed_cases"] == 2
    assert summarize_mixed(*reversed(mixed_reports()))["passed"]


def test_full_window_matrix_summary_after_json():
    tree = ast.parse(
        Path("tests/mla/e2e/window_paged/test_mla_window_paged_profile.py").read_text()
    )
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "summarize_reports"
    ]
    namespace = dict(
        json=json, summarize_mixed=summarize_mixed, PERFORMANCE_CASES=PERFORMANCE_CASES
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "window-profile-summary", "exec"),
        namespace,
    )
    _, reports = window_reports()
    for report in reports:
        template = report["results"][0]
        report["results"] = [
            dict(copy.deepcopy(template), case=i, spec=spec)
            for i, spec in enumerate(PERFORMANCE_CASES)
        ]
    captured = json.loads(json.dumps(reports))
    result = namespace["summarize_reports"](captured)
    assert result["passed"] and result["measured_cases"] == 2 * len(PERFORMANCE_CASES)
    captured[0]["results"][0]["spec"]["window"][0] += 1
    with pytest.raises(ValueError, match="input matrix"):
        namespace["summarize_reports"](captured)


@pytest.mark.parametrize(
    "fault", ["window", "missing", "segment", "timing", "dtype", "identity", "failed"]
)
def test_window_average_failure_propagation(fault):
    specs, reports = copy.deepcopy(window_reports())
    if fault == "window":
        reports[0]["results"][0]["spec"] = dict(specs[0], window=[3, 1])
    elif fault == "missing":
        reports[0]["results"].pop()
    elif fault == "segment":
        del reports[0]["results"][0]["entries"]["reuse"]["r4"]
    elif fault == "timing":
        reports[0]["rounds"] = 1
    elif fault == "dtype":
        reports[0]["dtype"] = "bf16"
    elif fault == "identity":
        del specs[0]["window"]
    else:
        reports[0]["results"][0]["entries"]["reuse"]["r1"] = dict(fa3=10000, cann=100, ratio=100)
        assert not summarize_mixed(reports, specs)["passed"]
        return
    with pytest.raises((ValueError, KeyError)):
        summarize_mixed(reports, specs)


def test_public_paged_route_and_protocol_size():
    text = Path("flash_attn_npu_3/flash_attn_npu_interface.py").read_text()
    tree = ast.parse(text)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    forward = ast.get_source_segment(text, functions["_mla_paged_bsnd_forward_fake"])
    assert "window_left=window_left" in forward and "window_right=window_right" in forward
    metadata = ast.get_source_segment(text, functions["_mla_paged_metadata_fake"])
    assert "411 if is_window else 408" in metadata
    assert "MLA TND PA window metadata is not implemented yet" not in metadata


def test_source_probe_uses_band_and_independent_window():
    text = Path("tests/mla/support/mla_window_paged_attention_probe.cpp").read_text()
    assert "sourceSchedule.window" in text and "window.kind == 1 ? 3 : 4" in text
    assert "window.kind == 1 ? INT32_MAX : window.left" in text
    assert "window.kind == 1 ? 0 : window.right" in text
    assert "ReadPagedMetadata<InputT, false, true>" in text


def test_reference_window_capacity_and_cache(tmp_path, monkeypatch):
    from tests.mla.support import mla_window_paged_reference as reference

    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path / "golden"))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    monkeypatch.setenv("GOLDEN_CACHE_REFRESH", "0")
    spec = dict(page=16, q=[3], lengths=[2], heads=1, capacity=32, window=(0, 0), causal=False)
    host = {
        n: torch.zeros(
            (1, 3 if n in ("q", "qv") else 16, 1, 64 if n in ("q", "k") else 512),
            dtype=torch.float16,
        )
        for n in ("q", "qv", "k", "v")
    }
    host["v"].fill_(3)
    case = dict(
        config=spec,
        item=dict(q=[3], uq=None, layout=0),
        context=SimpleNamespace(heads=1),
        host=host,
        table_host=torch.tensor([[0, -1]], dtype=torch.int32),
        scale=1 / 24,
    )
    first = reference.references(case)
    assert first[0][0].eq(0).all() and first[1][:, 0].isneginf().all()
    assert first[0][1:].eq(3).all() and first[1][:, 1:].eq(0).all()
    original = reference.uncached_references
    monkeypatch.setattr(
        reference,
        "uncached_references",
        lambda c: (_ for _ in ()).throw(AssertionError("cache miss")),
    )
    for a, b in zip(first, reference.references(case)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    monkeypatch.setattr(reference, "uncached_references", original)
    case["config"] = dict(spec, window=(1, 0))
    second = reference.references(case)
    assert second[1][0, -1] > first[1][0, -1]
