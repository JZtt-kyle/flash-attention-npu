"""CPU checks for refusing incomplete or mixed CANN baseline packages."""

import hashlib
import json

import pytest

from tests.mla_cann_baseline import check_vendor_configuration


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
