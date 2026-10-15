"""Window registration, native mask identity and public timing failure propagation."""

import ast
from pathlib import Path

import pytest
import torch
from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_window_public_cases import (
    PERFORMANCE_CASES,
    REPRESENTATIVES,
    COST_CASES,
    ACCURACY_CASES,
)
from tests.mla.support.mla_public_performance import summarize
from tests.mla.unit.tools.test_mla_validation_policy import reports
from tests.mla.unit.tools.test_mla_validation import args
from tests.mla.support.mla_window_public_profile import WindowCannCall
from tests.mla.support.mla_cann_baseline import CannCall


def test_selection_and_environment(tmp_path):
    assert all(s in REPRESENTATIVES + COST_CASES + ACCURACY_CASES for s in PERFORMANCE_CASES)
    assert all(s not in ACCURACY_CASES for s in REPRESENTATIVES + COST_CASES)
    assert all(s not in REPRESENTATIVES for s in COST_CASES)
    assert 2 * len(PERFORMANCE_CASES) <= 100
    for suite in ("window-bsnd-cpu", "window-bsnd-sanitize", "window-bsnd-tools"):
        env = runner.environment(args(tmp_path), tmp_path, suite)
        assert env["MLA_VALIDATION_NPU"] == "0" and env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    for suite in ("window-bsnd-check", "window-bsnd-public"):
        env = runner.environment(args(tmp_path), tmp_path, suite)
        assert env["MLA_VALIDATION_NPU"] == "1" and "MLA_WINDOW_ATTENTION_DIR" in env
        assert suite not in runner.STANDALONE_CHECK_SUITES
    assert "window-bsnd-profile" in runner.PERFORMANCE_SUITES


@pytest.mark.parametrize(
    "window,causal,mode,left,right",
    [
        ((2, 1), False, 4, 2, 1),
        ((-1, 1), False, 4, 32, 1),
        ((2, -1), False, 4, 2, 6),
        ((2, 1), True, 4, 2, 0),
        ((32, 6), False, 0, 2**31 - 1, 2**31 - 1),
        ((-1, 0), False, 3, 2**31 - 1, 0),
    ],
)
def test_native_band_parameters(monkeypatch, window, causal, mode, left, right):
    # Stub only the base constructor's hardware binding; exercise the actual
    # band adapter and real boolean mask on CPU, independently of production.
    def init(self, *a, **kw):
        self.kwargs = dict(sparse_mode=0, pre_tokens=2**31 - 1, next_tokens=2**31 - 1)

    monkeypatch.setattr(CannCall, "__init__", init)
    call = WindowCannCall(
        dict(q=torch.empty(2, 7, 16, 64), k=torch.empty(2, 33, 1, 64)),
        spec=dict(window=window, causal=causal),
    )
    assert (call.kwargs["sparse_mode"], call.kwargs["pre_tokens"], call.kwargs["next_tokens"]) == (
        mode,
        left,
        right,
    )
    if mode:
        mask = call.kwargs["atten_mask"]
        assert mask.shape == (2048, 2048) and mask.dtype == torch.bool
        assert not mask[0, 0] and mask[0, 1] and not mask[1, 0]


def window_reports():
    data = reports()
    for report in data:
        report.update(causal=False, window=True)
        for row in report["results"]:
            row.update(window=[2, 1], causal=False)
    return data


def test_window_average_identity_and_old_results():
    result = summarize(window_reports(), 2)
    assert result["passed"] and result["cases"][0]["window"] == [2, 1]
    assert summarize(reports(), 2)["passed"]


@pytest.mark.parametrize("fault", ["window", "missing", "segment", "timing", "dtype"])
def test_window_average_rejects_incomplete(fault):
    data = window_reports()
    if fault == "window":
        data[0]["results"][0]["window"] = [3, 1]
    elif fault == "missing":
        del data[0]["results"][0]["window"]
    elif fault == "segment":
        del data[0]["results"][0]["reuse_4_ratio"]
    elif fault == "timing":
        data[0]["rounds"] = 1
    else:
        data[0]["dtype"] = "bf16"
    with pytest.raises((ValueError, KeyError)):
        summarize(data, 2)


def test_fake_normalization_matches_fixed_rules():
    from tests.mla.support.components.window_schedule import normalized

    tree = ast.parse(Path("flash_attn_npu_3/flash_attn_npu_interface.py").read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_normalize_mla_window"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "fake-window", "exec"), namespace)
    normalize = namespace[node.name]
    for q, k in [(0, 0), (0, 7), (1, 33), (7, 0), (7, 33), (33, 7), (2**30, 16384)]:
        for left, right in [(-1, -1), (0, 0), (2, 1), (2**63 - 1, 2**63 - 1), (-2, -2)]:
            for causal in (False, True):
                assert list(normalize(q, k, left, right, causal)) == normalized(
                    [q, k, left, right, causal]
                )
    with pytest.raises(RuntimeError):
        normalize(7, 33, -(2**63), 1, False)
