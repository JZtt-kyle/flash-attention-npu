"""Build identity and explicit suite selection for the independent prefill probe."""

import pytest
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_prefill_build import valid
from tests.mla.support import mla_validation as driver


@pytest.mark.parametrize("changed", ("source", "dependency", "library", "missing", "kind", "dtype"))
def test_prefill_manifest_invalidation(tmp_path, monkeypatch, changed):
    from tests.mla.support import mla_prefill_build as build

    monkeypatch.setattr(build, "ROOT", tmp_path)
    files = {k: tmp_path / k for k in ("source", "dependency", "library")}
    for k, p in files.items():
        p.write_text(k)
    m = dict(
        prefill_merge=True,
        dtype="fp16",
        sources={"source": sha256(files["source"])},
        dependencies={str(files["dependency"]): sha256(files["dependency"])},
        libraries={"migrated": dict(path=str(files["library"]), sha256=sha256(files["library"]))},
    )
    assert valid(m)
    if changed == "kind":
        m["prefill_merge"] = False
    elif changed == "dtype":
        del m["dtype"]
    elif changed == "missing":
        files["library"].unlink()
    else:
        files[changed].write_text("modified")
    assert not valid(m)


def test_prefill_selection_and_unit_ownership():
    assert "prefill-check" in driver.STANDALONE_CHECK_SUITES
    assert "unit-prefill_attention" in driver.STANDALONE_CHECK_SUITES
    assert all("/unit/prefill_attention/" in p for p in driver.SUITES["unit-prefill_attention"])
    assert not driver.is_e2e(driver.SUITES["unit-prefill_attention"])
    assert "-k" in driver.SUITES["prefill-check"]


@pytest.mark.parametrize(
    "suite", ("prefill-tools", "prefill-check", "unit-prefill_attention", "prefill-profile")
)
def test_prefill_success_failure_interruption(tmp_path, monkeypatch, suite):
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume as check_states,
    )

    monkeypatch.setattr(driver, "validate_causal_builds", lambda *a, **k: None)
    # The child is a synthetic CPU process; it must not take the real NPU lock.
    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    check_states(tmp_path, monkeypatch, suite)


@pytest.mark.parametrize(
    "manifest",
    (None, {}, {"prefill_merge": True, "sources": {}, "dependencies": {}, "libraries": {}}),
)
def test_incomplete_manifest_is_not_a_build(manifest):
    assert not valid(manifest)


def test_prefill_builder_failure_is_not_cached(tmp_path, monkeypatch):
    from tests.mla.support import mla_prefill_build as build
    import subprocess

    monkeypatch.setattr(build, "work_dir", lambda: tmp_path)
    path = tmp_path / "prefill-probes/fp16/build-manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("broken JSON")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "bisheng")

    monkeypatch.setattr(build, "build_probes", fail)
    with pytest.raises(subprocess.CalledProcessError):
        build.build_prefill("fp16")
    assert path.read_text() == "broken JSON"
    with pytest.raises(ValueError):
        build.build_prefill("fp32")


@pytest.mark.parametrize("template", (0, 3, 4))
def test_prefill_native_preflight_requires_generic(tmp_path, monkeypatch, template):
    from tests.mla.support import mla_prefill_native as native

    monkeypatch.setattr(native, "native_vendor", lambda: tmp_path)
    monkeypatch.setattr(
        native, "parse_dispatch", lambda *a: dict(cases=[dict(native_template=template)] * 3)
    )
    if template in (3, 4):
        assert native.check("log", "fp16", tmp_path / "result.json")["cases"]
    else:
        with pytest.raises(RuntimeError, match="generic"):
            native.check("log", "fp16", tmp_path / "result.json")
        assert not (tmp_path / "result.json").exists()


def test_independent_v_reference_and_gqa_mapping():
    import torch
    from tests.mla.support.mla_prefill_reference import equations, fragment_rows

    spec = dict(shape=(1, 2, 4, 3, 2), m=32, n=32, splits=2)
    inputs = dict(
        q=torch.zeros(1, 2, 4, 192, dtype=torch.float16),
        k=torch.zeros(1, 3, 2, 192, dtype=torch.float16),
        v=torch.zeros(1, 3, 2, 128, dtype=torch.float16),
    )
    inputs["v"][:, :, 0] = 3
    inputs["v"][:, :, 1] = 7
    tasks = fragment_rows(spec)
    assert len(tasks) == 4 and [v[0][0] for v in tasks] == [0, 0, 1, 1]
    for high in (False, True):
        result = equations(inputs, spec, high)
        assert (result[0][0, :4] == 3).all() and (result[0][2, :4] == 7).all()
        assert (result[0][1] == 0).all() and torch.isneginf(result[1][1]).all()
        assert torch.equal(result[4], result[0])


@pytest.mark.parametrize("change", ("dtype", "cann", "compiler"))
def test_prefill_environment_change_requires_build(tmp_path, monkeypatch, change):
    import json
    from tests.mla.support import mla_prefill_build as build

    monkeypatch.setattr(build, "work_dir", lambda: tmp_path)
    monkeypatch.setattr(build, "valid", lambda m: True)
    monkeypatch.setenv("ASCEND_HOME_PATH", "/cann/current")
    monkeypatch.setattr(build.shutil, "which", lambda _: "/toolchain/g++")
    manifest = dict(dtype="fp16", cann="/cann/current", dependencies={"/toolchain/g++": "hash"})
    if change == "dtype":
        manifest["dtype"] = "bf16"
    elif change == "cann":
        manifest["cann"] = "/cann/old"
    else:
        manifest["dependencies"] = {"/old/g++": "hash"}
    path = tmp_path / "prefill-probes/fp16/build-manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))

    def fail(*a, **kw):
        raise RuntimeError("new build required")

    monkeypatch.setattr(build, "build_probes", fail)
    with pytest.raises(RuntimeError, match="new build required"):
        build.build_prefill("fp16")
