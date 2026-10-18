"""Cheap INT8 fixtures, recipe identity and failure propagation before NPU runs."""

import json

import pytest
import torch

from tests.mla.support.mla_int8_cases import SOURCE_CASES, make_inputs, pack_nz, unpack_nz
from tests.mla.support import mla_int8_native as native
from tests.mla.support.mla_int8_reference import equations


@pytest.mark.parametrize("dtype,dim,c0", [(torch.int8, 512, 32), (torch.bfloat16, 64, 16)])
def test_nz_coordinates_and_strided_page_pool(dtype, dim, c0):
    raw = (torch.arange(6 * 128 * dim).remainder(121) - 60).to(dtype).reshape(6, 128, 1, dim)
    nd = raw[1::2]
    nz = pack_nz(nd)
    assert nz.shape == (3, 1, dim // c0, 128, c0)
    for p, row, channel in ((0, 0, 0), (1, 127, dim - 1), (2, 33, c0 + 3)):
        assert nz[p, 0, channel // c0, row, channel % c0] == raw[1 + 2 * p, row, 0, channel]
    torch.testing.assert_close(unpack_nz(nz), nd, rtol=0, atol=0)


def test_raw_fixture_and_mixed_signature():
    case = make_inputs(SOURCE_CASES[0])
    assert {k: t.dtype for k, t in case.items()} == dict(
        qv=torch.int8,
        q=torch.bfloat16,
        v=torch.int8,
        k=torch.bfloat16,
        qs=torch.float32,
        ks=torch.float32,
        vs=torch.float32,
        table=torch.int32,
    )
    declaration = native.declaration()["op_list"][0]
    args = {
        (e[0] if isinstance(e, list) else e)["name"]: (e[0] if isinstance(e, list) else e)["dtype"]
        for e in declaration["inputs"]
    }
    assert [args[k] for k in ("query", "key", "value")] == ["int8"] * 3
    assert [args[k] for k in ("query_rope", "key_rope")] == ["bfloat16"] * 2
    assert [
        args[k] for k in ("dequant_scale_query", "key_antiquant_scale", "value_antiquant_scale")
    ] == ["float32"] * 3
    assert declaration["outputs"][0]["dtype"] == "bfloat16"
    assert len(native.INT8_KEYS) == len(set(native.INT8_KEYS)) == 12


def test_reference_single_key_and_q_scale_head_address():
    spec = ("reference-one-key", "TND", (1, 1), 2, (1, 1), 0)
    inputs = make_inputs(spec)
    out, _ = equations(inputs, spec, torch.float64)
    expected = inputs["v"][inputs["table"][:, 0].long(), 0, 0].float() * inputs["vs"]
    torch.testing.assert_close(out, expected[:, None].expand(-1, 2, -1), rtol=0, atol=0)
    spec = ("reference-head-scale", "BSND", (1,), 2, (2,), 0)
    inputs = make_inputs(spec)
    inputs["q"].zero_()
    inputs["k"].zero_()
    inputs["qs"][..., 0] = 0
    out, _ = equations(inputs, spec, torch.float64)
    page = inputs["table"][0, 0].item()
    expected = inputs["v"][page, :2, 0].double().mean(0) * inputs["vs"].double()
    torch.testing.assert_close(out[0, 0, 0], expected.float(), rtol=0, atol=0)
    assert not torch.equal(out[0, 0, 0], out[0, 0, 1])


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.setenv("MLA_INT8_NATIVE_ROOT", str(tmp_path))
    artifact = tmp_path / "opp/vendors" / native.native_package.VENDOR / "test.o"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"test-only")
    manifest = dict(
        commit=native.COMMIT,
        recipe=native.sha256(native.Path(native.__file__)),
        inputs=native.recipe_inputs(),
        keys=list(native.INT8_KEYS),
        files={str(artifact.relative_to(tmp_path)): native.sha256(artifact)},
    )
    (tmp_path / "package.json").write_text(json.dumps(manifest))
    return tmp_path, artifact, manifest


@pytest.mark.parametrize("fault", ["missing", "recipe", "inputs", "keys", "bytes", "escape"])
def test_package_identity_rejects_stale_or_incomplete(package, fault):
    root, artifact, manifest = package
    assert native.native_vendor() == artifact.parent
    if fault == "missing":
        artifact.unlink()
    elif fault == "bytes":
        artifact.write_bytes(b"changed")
    elif fault == "escape":
        manifest["files"] = {"../foreign.o": "0"}
    else:
        manifest[fault] = None
    (root / "package.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        native.native_vendor()


def test_host_build_failure_does_not_publish_or_change_existing_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MLA_NATIVE_ROOT", "/existing/nonquant")
    monkeypatch.setenv("MLA_INT8_NATIVE_ROOT", str(tmp_path))

    def fail(**kwargs):
        assert kwargs == {"host_only": True}
        raise RuntimeError("host compiler failed")

    monkeypatch.setattr(native.native_package, "build", fail)
    with pytest.raises(RuntimeError, match="host compiler failed"):
        native.build()
    assert native.os.environ["MLA_NATIVE_ROOT"] == "/existing/nonquant"
    assert not (tmp_path / "package.json").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "missing-chunk",
        "repeat-chunk",
        "mixed-op",
        "size",
        "trailer",
        "core-zero",
        "core-large",
        "used-core",
        "bound",
        "signature",
    ],
)
def test_tiling_replay_rejects_incomplete_geometry(tmp_path, monkeypatch, fault):
    import ctypes as ct
    from tests.mla.support import mla_int8_tiling as tiling

    params = tiling.Params()
    params.increFlashAttentionSingleCoreParams.usedCoreNum = 2 if fault == "used-core" else 1
    words = list((ct.c_uint32 * (ct.sizeof(params) // 4)).from_buffer_copy(params)) + [
        0,
        0xA5A5A5A5,
    ]
    if fault == "size":
        words.append(0)
    elif fault == "trailer":
        words[-1] = 0
    chunks = [
        "OpName:[test_FusedInferAttentionScore] PrintTilingData[0]: data(uint32_t): "
        + " ".join(map(str, words))
    ]
    if fault == "missing-chunk":
        chunks[0] = chunks[0].replace("Data[0]", "Data[1]")
    elif fault == "repeat-chunk":
        chunks *= 2
    elif fault == "mixed-op":
        chunks += ["OpName:[other_FusedInferAttentionScore] PrintTilingData[1]: 0"]
    specs = None
    if fault in ("bound", "signature"):
        from tests.mla.support.mla_int8_cases import case_identity

        specs = [SOURCE_CASES[0]]
        signature = case_identity(specs[0]) if fault == "bound" else "0" * 64
        chunks += ["MLA_INT8_CASE_SIGNATURE " + signature]
    cores = 0 if fault == "core-zero" else 21 if fault == "core-large" else 1
    lines = [
        "BEGIN",
        *chunks,
        f"LaunchKernel: kernel info kernel_name=FusedInferAttentionScore_test coreDim={cores}",
        "END",
    ]
    log = tmp_path / "source.log"
    log.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(
        tiling,
        "parse_dispatch",
        lambda *a, **kw: dict(
            artifacts={},
            cases=[
                dict(
                    begin_line=1,
                    end_line=len(lines),
                    case=0,
                    native_tiling_key=15000000020222331,
                    native_fd=False,
                )
            ],
        ),
    )
    if fault in ("none", "bound"):
        assert tiling.parse_tiling(log, 1, tmp_path, specs)["cases"][0]["cores"] == 1
    else:
        with pytest.raises(RuntimeError):
            tiling.parse_tiling(log, 1, tmp_path, specs)


@pytest.mark.parametrize("fault", ["valid", "recipe", "library", "missing", "invalid-json"])
def test_probe_cache_identity_and_rebuild_failure(tmp_path, monkeypatch, fault):
    from tests.mla.support import mla_int8_build as build

    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "cann"))
    recipe, library = tmp_path / "recipe", tmp_path / "test.so"
    recipe.write_text("recipe")
    library.write_text("binary")
    manifest = dict(
        commit=build.COMMIT,
        cann=str(tmp_path / "cann"),
        dependencies={str(recipe): build.sha256(recipe)},
        libraries={"test": dict(path=str(library), sha256=build.sha256(library))},
    )
    record = tmp_path / "build-manifest.json"
    record.write_text(json.dumps(manifest))
    if fault == "recipe":
        recipe.write_text("changed")
    elif fault == "library":
        library.write_text("changed")
    elif fault == "missing":
        library.unlink()
    elif fault == "invalid-json":
        record.write_text("{")

    def fail(*args, **kwargs):
        raise RuntimeError("source archive unavailable")

    monkeypatch.setattr(build.subprocess, "check_output", fail)
    if fault == "valid":
        assert build._build_probes(tmp_path) == manifest
    else:
        with pytest.raises(RuntimeError, match="source archive unavailable"):
            build._build_probes(tmp_path)


@pytest.mark.parametrize("suite", ["unit-int8", "int8-source-check", "int8-component-profile"])
def test_driver_failure_interruption_and_no_unit_or_performance_resume(
    tmp_path, monkeypatch, suite
):
    from tests.mla.support import mla_validation as driver
    from tests.mla.support import mla_int8_build as build
    from tests.mla.support import mla_native_dispatch as dispatch
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume,
    )

    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    monkeypatch.setattr(native, "native_vendor", lambda: tmp_path)
    monkeypatch.setattr(build, "build_probes", lambda: {})
    monkeypatch.setattr(dispatch, "parse_dispatch", lambda *a, **kw: {})
    test_success_failure_interruption_and_resume(tmp_path, monkeypatch, suite)


def test_driver_dependency_change_rejects_successful_child(tmp_path, monkeypatch):
    from tests.mla.support import mla_validation as driver
    from tests.mla.support import mla_int8_build as build
    from tests.mla.unit.tools.test_mla_validation import (
        test_causal_dependencies_changing_during_execution_fail,
    )

    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {"unit-int8"})
    monkeypatch.setattr(native, "native_vendor", lambda: tmp_path)
    monkeypatch.setattr(build, "build_probes", lambda: {})
    test_causal_dependencies_changing_during_execution_fail(tmp_path, monkeypatch, "unit-int8")


def test_driver_missing_native_dispatch_rejects_successful_child(tmp_path, monkeypatch):
    from tests.mla.support import mla_validation as driver
    from tests.mla.support import mla_native_dispatch as dispatch
    from tests.mla.unit.tools.test_mla_validation import args, evidence

    suite = "int8-source-check"
    config = args(tmp_path)
    config.suite = [suite]
    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    monkeypatch.setattr(driver, "sources", lambda: {})
    monkeypatch.setattr(driver, "suite_dependencies", lambda _: {})
    monkeypatch.setattr(native, "native_vendor", lambda: tmp_path)

    def execute(command, env, directory, lock_fds):
        evidence(directory)
        return dict(returncode=0, seconds=0)

    def missing(*args, **kwargs):
        raise RuntimeError("missing native dispatch")

    monkeypatch.setattr(driver, "execute", execute)
    monkeypatch.setattr(dispatch, "parse_dispatch", missing)
    assert driver.run(config) == 1
    group = json.loads((config.output / "summary.json").read_text())["groups"][suite]
    assert group["status"] == "failed" and group["error"] == "missing native dispatch"
    assert suite not in json.loads((config.output / "state.json").read_text())
