"""Cheap softcap equations, component cache identity and unified-driver behavior."""

import json
import math

import pytest
import torch

from tests.mla.support import mla_validation as driver
from tests.mla.support.components import softcap
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_prefill_reference import equations
from tests.mla.support.mla_split_kv_attention_reference import partial_math
from tests.mla.support.mla_softcap_cases import cases, PERFORMANCE, REPRESENTATIVES


@pytest.mark.parametrize("representation", ("compressed", "prefill"))
@pytest.mark.parametrize("cap", (0, 15, 30))
def test_reference_matches_scalar_total_score(representation, cap):
    # Exact binary scale and integer components isolate softcap placement.
    q = torch.zeros(1, 2, 2, 192, dtype=torch.float16)
    k = torch.zeros(1, 3, 1, 192, dtype=torch.float16)
    q[..., 0], q[..., 128] = 32, 16
    k[0, :, 0, 0] = torch.tensor([-24, 0, 24])
    k[0, :, 0, 128] = torch.tensor([48, 1, -47])
    spec = dict(shape=(1, 2, 2, 3), m=16, n=32, splits=1, softcap=cap, scale=0.25)
    scores = [0.0, 4.0, 4.0]
    if cap:
        scores = [cap * math.tanh(v / cap) for v in scores]
    total = sum(math.exp(v) for v in scores)
    expected_lse = math.log(total)
    if representation == "prefill":
        values = torch.zeros(1, 3, 1, 128, dtype=q.dtype)
        values[0, :, 0, 0] = torch.tensor([1, 2, 3])
        result = equations(dict(q=q, k=k, v=values), dict(spec, shape=(*spec["shape"], 1)), True)
        value_scores = [1, 2, 3]
    else:
        qc, v = torch.zeros(1, 2, 2, 512, dtype=q.dtype), torch.zeros(1, 3, 1, 512, dtype=q.dtype)
        qc[..., :128], v[..., :128] = q[..., :128], k[..., :128]
        result = partial_math(dict(q=q[..., 128:], qv=qc, k=k[..., 128:], v=v), spec, 16, 32, True)
        value_scores = [-24, 0, 24]
    expected_out = sum(math.exp(s) * v for s, v in zip(scores, value_scores)) / total
    torch.testing.assert_close(
        result[0][0, :4, 0], torch.full((4,), expected_out), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(result[1][0, :4], torch.full((4,), expected_lse), rtol=0, atol=1e-6)
    if cap:
        wrong = [
            cap * (math.tanh(a / cap) + math.tanh(b / cap))
            for a, b in ((-192, 192), (0, 4), (192, -188))
        ]
        assert max(abs(a - b) for a, b in zip(scores, wrong)) > 1


@pytest.mark.parametrize(
    "changed", ("source", "dependency", "library", "missing", "dtype", "kind", "cann", "compiler")
)
def test_component_cache_invalidation(tmp_path, monkeypatch, changed):
    monkeypatch.setattr(softcap, "ROOT", tmp_path)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "cann"))
    monkeypatch.setattr(softcap.shutil, "which", lambda _: str(tmp_path / "g++"))
    files = {k: tmp_path / k for k in ("source", "dependency", "library", "g++")}
    files["compiler"] = tmp_path / "cann/bin/bisheng"
    for k, p in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(k)
    manifest = dict(
        split_kv_merge=True,
        dtype="fp16",
        cann=str(tmp_path / "cann"),
        sources={"source": sha256(files["source"])},
        dependencies={str(files[k]): sha256(files[k]) for k in ("dependency", "compiler", "g++")},
        libraries={"migrated": dict(path=str(files["library"]), sha256=sha256(files["library"]))},
    )
    assert softcap.valid_merge(manifest, "fp16")
    if changed in ("source", "dependency", "library", "compiler"):
        files[changed].write_text("changed")
    elif changed == "missing":
        files["library"].unlink()
    elif changed == "kind":
        manifest["split_kv_merge"] = False
    else:
        manifest[changed] = "different"
    assert not softcap.valid_merge(manifest, "fp16")


@pytest.mark.parametrize(
    "suite",
    ("softcap-tools", "softcap-check", "unit-softcap", "softcap-profile-check", "softcap-profile"),
)
def test_driver_success_failure_and_no_unit_resume(tmp_path, monkeypatch, suite):
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume as check,
    )

    monkeypatch.setattr(driver, "validate_causal_builds", lambda *a, **k: None)
    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    check(tmp_path, monkeypatch, suite)


def test_selection_ownership_and_fixed_coverage():
    assert "unit-softcap" in driver.STANDALONE_CHECK_SUITES
    assert not driver.is_e2e(driver.SUITES["unit-softcap"])
    assert all("/unit/softcap/" in p for p in driver.SUITES["unit-softcap"])
    assert "softcap-profile" in driver.PERFORMANCE_SUITES
    for representation in ("compressed", "prefill"):
        specs = cases(representation)
        assert len({s["id"] for s in specs}) == len(specs)
        assert PERFORMANCE <= {s["id"] for s in specs}
        assert "cost-boundary" in REPRESENTATIVES
        assert {s["softcap"] for s in specs} == {0, 15, 30, 1e8, 1e-4}
        assert {s.get("pattern") for s in specs} >= {"saturation", "cancellation"}
        # Selection is serializable and records no timing-based exclusions.
        assert json.loads(json.dumps(specs))
