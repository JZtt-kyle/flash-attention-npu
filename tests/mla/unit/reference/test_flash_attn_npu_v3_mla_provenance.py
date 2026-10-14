"""CPU checks for refusing incomplete or mixed CANN baseline packages."""

import hashlib
import json
import subprocess

import pytest

from tests.mla.support.mla_cann_baseline import check_vendor_configuration
from tests.mla.support import mla_cann_baseline as baseline


@pytest.fixture
def vendor(tmp_path, monkeypatch):
    root = tmp_path / "vendor"
    kernel = root / "op_impl/ai_core/tbe/kernel"
    config = kernel / "config/ascend910b/fused_infer_attention_score.json"
    config.parent.mkdir(parents=True)
    bins = []
    for dtype in ("float16", "bfloat16"):
        # These are inert files: the guard must reject bad packages before NPU use.
        binary = kernel / f"ascend910b/fused_infer_attention_score/{dtype}.o"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(dtype.encode())
        metadata = binary.with_suffix(".json")
        metadata.write_text(
            json.dumps(
                dict(
                    binFileName=dtype,
                    binFileSuffix=".o",
                    sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    kernelList=[dict(tilingKey=105000000010000001)],
                )
            )
        )
        bins.append(
            dict(
                inputs=[dict(dtype=dtype)],
                outputs=[dict(dtype=dtype)],
                binInfo=dict(jsonFilePath=str(metadata.relative_to(kernel))),
            )
        )
    config.write_text(json.dumps(dict(binList=bins)))
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(root))
    return root, kernel, config


def test_vendor_accepts_native_set_env_trailing_separator(vendor, monkeypatch):
    root, _, _ = vendor
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(root) + ":")
    assert check_vendor_configuration(root) == root


def test_vendor_rejects_jit_only_package(vendor):
    root, _, config = vendor
    config.unlink()
    with pytest.raises(RuntimeError, match="compiled 910B FIA kernel configuration"):
        check_vendor_configuration(root)


def test_vendor_rejects_ambiguous_opp_environment(vendor, monkeypatch):
    root, _, _ = vendor
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(root) + ":/another/vendor")
    with pytest.raises(RuntimeError, match="exactly the requested CANN vendor"):
        check_vendor_configuration(root)


@pytest.mark.parametrize("damage", ["missing", "stale"])
def test_vendor_rejects_incomplete_device_binary(vendor, damage):
    root, kernel, _ = vendor
    binary = kernel / "ascend910b/fused_infer_attention_score/bfloat16.o"
    if damage == "missing":
        binary.unlink()
    else:
        binary.write_bytes(b"different build")
    with pytest.raises(RuntimeError, match="device binary"):
        check_vendor_configuration(root)


def test_vendor_rejects_kernel_symlink_outside_package(vendor, tmp_path):
    root, kernel, _ = vendor
    binary = kernel / "ascend910b/fused_infer_attention_score/float16.o"
    outside = tmp_path / "installed.o"
    outside.write_bytes(binary.read_bytes())
    binary.unlink()
    binary.symlink_to(outside)
    with pytest.raises(RuntimeError, match="outside vendor"):
        check_vendor_configuration(root)


@pytest.mark.parametrize("damage", [None, "missing_commit", "header", "unloaded"])
def test_archive_provenance_ignores_checkout_head(vendor, tmp_path, monkeypatch, damage):
    root, _, _ = vendor
    source = tmp_path / "source"
    source.mkdir()

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@invalid",
                *args,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

    git("init", "-q")
    relative = "attention/fused_infer_attention_score/op_kernel/arch22"
    headers = (
        "fia_kernel_nonquant_mla.h",
        "fia_block_cube_nonquant_mla.h",
        "fia_block_vec_nonquant_mla.h",
        "fia_kernel_nonquant.h",
    )
    for name in headers:
        for base in (source, root):
            path = base / relative / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
    git("add", ".")
    git("commit", "-qm", "Pinned source")
    pinned = git("rev-parse", "HEAD")
    (source / relative / headers[0]).write_text("different HEAD")
    git("commit", "-qam", "Unrelated current checkout")
    (source / relative / headers[1]).write_text("uncommitted edit")
    assert git("rev-parse", "HEAD") != pinned
    monkeypatch.setattr(baseline, "PINNED_CANN", "0" * 40 if damage == "missing_commit" else pinned)
    loaded = []
    for name in ("libcust_opapi.so", "libcust_opmaster_rt2.0.so"):
        library = root / name
        library.write_bytes(b"inert host library")
        loaded.append(str(library))
    if damage == "header":
        (root / relative / headers[0]).write_text("wrong package source")
    if damage == "unloaded":
        loaded.pop()
    errors = {
        "missing_commit": "requires the pinned CANN commit",
        "header": "does not match the pinned source",
        "unloaded": "was not uniquely loaded",
    }
    if damage:
        with pytest.raises(RuntimeError, match=errors[damage]):
            baseline.verify_loaded_vendor(root, source, loaded)
    else:
        result = baseline.verify_loaded_vendor(root, source, loaded)
        assert result["source_commit"] == pinned
        assert len(result["pinned_headers"]) == 4
