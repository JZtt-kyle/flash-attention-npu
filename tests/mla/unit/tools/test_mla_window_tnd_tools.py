"""TND window selection, actual band arguments and timing evidence rejection."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_window_tnd_cases import (
    PERFORMANCE_CASES,
    REPRESENTATIVES,
    COST_CASES,
    ACCURACY_CASES,
)
from tests.mla.support.mla_public_performance import summarize_varlen
from tests.mla.unit.tools.test_mla_validation_policy import varlen_reports
from tests.mla.unit.tools.test_mla_validation import args
from tests.mla.support.components.window_schedule import normalized


def test_selection_and_environment(tmp_path):
    assert runner.SUITES["window-tnd-profile"] == [
        "tests/mla/e2e/window_tnd/test_mla_window_tnd_profile.py"
    ]
    assert (
        runner.SUITES["window-tnd-tools"][0] == "tests/mla/unit/tools/test_mla_window_tnd_tools.py"
    )
    assert all(s in REPRESENTATIVES + COST_CASES + ACCURACY_CASES for s in PERFORMANCE_CASES)
    assert all(s not in ACCURACY_CASES for s in REPRESENTATIVES + COST_CASES)
    assert all(s not in REPRESENTATIVES for s in COST_CASES)
    assert 2 * len(PERFORMANCE_CASES) <= 100
    for suite in ("window-tnd-cpu", "window-tnd-sanitize", "window-tnd-tools"):
        env = runner.environment(args(tmp_path), tmp_path, suite)
        assert env["MLA_VALIDATION_NPU"] == "0" and env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    for suite in ("window-tnd-check", "window-tnd-public"):
        env = runner.environment(args(tmp_path), tmp_path, suite)
        assert env["MLA_VALIDATION_NPU"] == "1" and "MLA_WINDOW_ATTENTION_DIR" in env
        assert suite not in runner.STANDALONE_CHECK_SUITES
    assert "window-tnd-profile" in runner.PERFORMANCE_SUITES


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
    # Extract only the real adapter, replacing the hardware base constructor.
    # The independent fixed-FA3 oracle determines its normalized parameters.
    class Base:
        def __init__(self, *a):
            self.native = SimpleNamespace(
                kwargs=dict(sparse_mode=0, pre_tokens=2**31 - 1, next_tokens=2**31 - 1)
            )

    def normalize(s):
        _, q, _, k = s["shape"]
        values = normalized([q, k, *s["window"], int(s["causal"])])
        return SimpleNamespace(**dict(zip(("kind", "left", "right"), values)))

    tree = ast.parse(Path("tests/mla/support/mla_window_tnd_profile.py").read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and n.name in ("window_for", "WindowTndCall")
    ]
    namespace = dict(torch=torch, CannTndUsedCall=Base, normalized_window=normalize)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "window-tnd-native", "exec"), namespace)
    case = dict(
        spec=dict(q=[3], heads=1, window=window, causal=causal),
        raw=dict(q=None),
        args=dict(
            q=torch.empty(3, 1, 64),
            max_seqlen_q=bounds[0],
            max_seqlen_k=bounds[1],
            softmax_scale=None,
        ),
    )
    call = namespace["WindowTndCall"](case)
    assert call.description() == dict(sparse_mode=mode, pre_tokens=left, next_tokens=right)
    if mode:
        mask = call.native.kwargs["atten_mask"]
        assert mask.shape == (2048, 2048) and mask.dtype == torch.bool
        assert not mask[0, 0] and mask[0, 1] and not mask[1, 0]


def window_reports():
    specs, reports = varlen_reports()
    for s in specs:
        s.update(window=[2, 1], causal=False)
    for report in reports:
        report.update(causal=False, window=True)
    return specs, reports


def test_window_average_and_unchanged_causal():
    specs, reports = window_reports()
    result = summarize_varlen(reports, specs)
    assert result["passed"] and result["regressed_cases"] == 2
    assert summarize_varlen(*reversed(varlen_reports()))["passed"]


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
        del reports[0]["results"][0]["r4"]
    elif fault == "timing":
        reports[0]["rounds"] = 1
    elif fault == "dtype":
        reports[0]["dtype"] = "bf16"
    elif fault == "identity":
        del specs[0]["window"]
    else:
        reports[0]["results"][0]["r1"] = dict(fa3=1000, cann=100, ratio=10)
        assert not summarize_varlen(reports, specs)["passed"]
        return
    with pytest.raises((ValueError, KeyError)):
        summarize_varlen(reports, specs)


@pytest.mark.parametrize("entry", ["public", "op", "eager", "cpp"])
def test_forward_driver_preserves_window_arguments(entry):
    calls = []

    def record(**kw):
        calls.append(kw)
        return ("out", "lse", "empty", "empty")

    tree = ast.parse(Path("tests/mla/support/public_cases/window_tnd_public.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "call")
    namespace = dict(
        api=SimpleNamespace(flash_attn_varlen_func=record),
        torch=SimpleNamespace(
            ops=SimpleNamespace(flash_attn_npu_3=SimpleNamespace(_mla_window_tnd_forward=record))
        ),
        _forward=lambda kind, kw: record(**kw),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "window-tnd-calls", "exec"), namespace)
    original = dict(window_size=[513, 7], causal=True, seqused_q="q-used", seqused_k="k-used")
    namespace["call"](original, entry)
    kw = calls[0]
    assert kw["causal"] and kw["seqused_q"] == "q-used" and kw["seqused_k"] == "k-used"
    if entry == "public":
        assert kw["window_size"] == [513, 7]
    else:
        keys = (
            ("window_left", "window_right")
            if entry == "op"
            else ("window_size_left", "window_size_right")
        )
        assert tuple(kw[k] for k in keys) == (513, 7) and "window_size" not in kw
    assert original["window_size"] == [513, 7]


def test_parallel_reference_uses_allocation_starts_and_used_lengths():
    import itertools
    from concurrent.futures import ThreadPoolExecutor

    tree = ast.parse(Path("tests/mla/support/public_cases/window_tnd_public.py").read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("effective", "reference_pairs")
    ]
    namespace = dict(
        itertools=itertools,
        ThreadPoolExecutor=ThreadPoolExecutor,
        mla_reference_pair=lambda **kw: kw,
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "window-tnd-reference", "exec"), namespace
    )
    q = torch.arange(7).reshape(7, 1, 1)
    k = torch.arange(12).reshape(12, 1, 1)
    spec = dict(q=[3, 2, 2], kv=[4, 3, 5], uq=[2, 0, 1], uk=[1, 2, 3], scale=0.25)
    case = dict(spec=spec, raw=dict(q=q, qv=q + 100, k=k, v=k + 100))
    pairs = namespace["reference_pairs"](case, SimpleNamespace(kind=2, left=5, right=1))
    assert pairs[1] is None
    assert pairs[0][0]["q"].flatten().tolist() == [0, 1]
    assert pairs[2][0]["q"].flatten().tolist() == [5]
    assert pairs[2][0]["k"].flatten().tolist() == [7, 8, 9]
    for _, refs in (pairs[0], pairs[2]):
        assert refs["window_size"] == (5, 1) and refs["softmax_scale"] == 0.25
        assert not refs["causal"]
