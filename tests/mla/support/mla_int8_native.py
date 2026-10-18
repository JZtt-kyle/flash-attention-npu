"""Pinned INT8 comparison package; NZ packing is confined to source tests.

The existing nonquantized package and its recipe remain unchanged. This uses
the same archive-only host builder and manifest fields, with a separate root.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from tests.mla.paths import ROOT, work_dir
from tests.mla.support.mla_dense_build import COMMIT, sha256
from tests.mla.support import native_package


# ca18aee incre_flash_attention_tilingkey.h: BSH decode, balanced BSH/TND,
# each with/without native FD, and the C1V1 variants on the same arch22.
INT8_KEYS = tuple(
    base + cv + fd
    for base in (15000000020222331, 15000001020222331, 15000001020222332)
    for cv in (0, 10000000000)
    for fd in (0, 100000)
)


def package_root():
    return Path(os.environ.get("MLA_INT8_NATIVE_ROOT", work_dir() / "int8-native")).resolve()


def declaration():
    # Reuse FIA's full argument list, preserving optional-argument positions.
    data = json.loads(
        (Path(native_package.__file__).with_name("native_inputs") / "bf16.json").read_text()
    )
    op = data["op_list"][0]
    for entry in op["inputs"]:
        arg = entry[0] if isinstance(entry, list) else entry
        if arg["name"] in ("query", "key", "value", "key_shared_prefix", "value_shared_prefix"):
            arg["dtype"] = "int8"
        if arg["name"] in ("key_antiquant_scale", "value_antiquant_scale"):
            arg["dtype"] = "float32"
    # The name is deterministic and identifies this mixed-dtype signature.
    signature = json.dumps(op["inputs"], sort_keys=True).encode()
    op["bin_filename"] = "FusedInferAttentionScore_" + hashlib.md5(signature).hexdigest()
    return data


def recipe_inputs():
    return {
        str(Path(native_package.__file__).resolve()): sha256(Path(native_package.__file__)),
        "declaration": hashlib.sha256(
            json.dumps(declaration(), sort_keys=True).encode()
        ).hexdigest(),
    }


def native_vendor():
    root = package_root()
    path = root / "package.json"
    if not path.is_file():
        raise RuntimeError(
            "Build INT8 native package: python3 -m tests.mla.support.mla_int8_native"
        )
    data = json.loads(path.read_text())
    if (
        data.get("commit") != COMMIT
        or data.get("recipe") != sha256(Path(__file__))
        or data.get("inputs") != recipe_inputs()
    ):
        raise RuntimeError("INT8 native recipe changed; rebuild the comparison package")
    if data.get("keys") != list(INT8_KEYS) or not data.get("files"):
        raise RuntimeError("Incomplete INT8 native package")
    for relative, digest in data["files"].items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha256(p) != digest:
            raise RuntimeError(f"INT8 native artifact changed: {relative}")
    return root / "opp/vendors" / native_package.VENDOR


def build():
    root = package_root()
    if root.is_relative_to(ROOT / "build"):
        raise ValueError("INT8 native artifacts must be outside build/")
    previous = os.environ.get("MLA_NATIVE_ROOT")
    os.environ["MLA_NATIVE_ROOT"] = str(root)
    try:
        native_package.build(host_only=True)
    finally:
        if previous is None:
            os.environ.pop("MLA_NATIVE_ROOT", None)
        else:
            os.environ["MLA_NATIVE_ROOT"] = previous
    vendor = root / "opp/vendors" / native_package.VENDOR
    kernel = vendor / "op_impl/ai_core/tbe/kernel"
    target = kernel / "ascend910b/fused_infer_attention_score"
    target.mkdir(parents=True, exist_ok=True)
    inputs = root / "int8-input.json"
    inputs.write_text(json.dumps(declaration(), indent=2) + "\n")
    output = root / "device/int8"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        ASCEND_CUSTOM_OPP_PATH=str(vendor),
        ASCEND_CACHE_PATH=str(root / "cache/int8"),
        ASCEND_PROCESS_LOG_PATH=str(root / "compile-logs/int8"),
    )
    native_package.run(
        [
            "opc",
            str(
                vendor
                / f"op_impl/ai_core/tbe/{native_package.VENDOR}_impl/dynamic/fused_infer_attention_score.py"
            ),
            "--main_func=fused_infer_attention_score",
            f"--input_param={inputs}",
            "--soc_version=Ascend910B3",
            f"--output={output}",
            "--impl_mode=high_performance,optional",
            "--simplified_key_mode=0",
            "--op_mode=dynamic",
            "--tiling_key=" + ",".join(map(str, INT8_KEYS)),
        ],
        root,
        "device-int8",
        env=env,
    )
    metadata = list(output.glob("*.json"))
    if len(metadata) != 1:
        raise RuntimeError("Expected one INT8/BF16 comparison binary")
    info = json.loads(metadata[0].read_text())
    binary = metadata[0].with_suffix(".o")
    if sha256(binary) != info["sha256"] or not set(INT8_KEYS) <= {
        int(k["tilingKey"]) for k in info["kernelList"]
    }:
        raise RuntimeError("Incomplete INT8 compiled key set")
    for p in (metadata[0], binary):
        shutil.copy2(p, target / p.name)
    native_package.run(
        [
            sys.executable,
            str(root / "source/cmake/scripts/util/ascendc_ops_config.py"),
            "-p",
            str(kernel / "ascend910b"),
            "-s",
            "ascend910b",
            "-o",
            str(kernel / "config/ascend910b"),
            "--op-range",
            "normal",
        ],
        root,
        "kernel-config",
        env=env,
    )
    record = dict(
        commit=COMMIT,
        recipe=sha256(Path(__file__)),
        inputs=recipe_inputs(),
        keys=list(INT8_KEYS),
        files={
            str(p.relative_to(root)): sha256(p) for p in sorted(vendor.rglob("*")) if p.is_file()
        },
    )
    (root / "package.json").write_text(json.dumps(record, indent=2) + "\n")
    print(native_vendor())


if __name__ == "__main__":
    build()
