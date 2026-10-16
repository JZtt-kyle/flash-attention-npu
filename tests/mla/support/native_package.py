"""Rebuild the pinned CANN comparison package without any historical build directory.

python -m tests.mla.support.native_package
The upstream build downloads its declared dependencies if MLA_NATIVE_DEPS is empty.
It only receives a git archive; the CANN source checkout is never modified.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from tests.mla.paths import ROOT, work_dir
from tests.mla.support.mla_dense_build import COMMIT, sha256

VENDOR = "mla_ca18aee_transformer"
DENSE_KEYS = (
    103000000010000001,
    103000000010400001,
    103000000010100001,
    103000000010500001,
    104000000010000001,
    104000000010400001,
    104000000010100001,
    104000000010500001,
    105000000010000001,
    105000000110000001,
    105000000010100001,
    105000000110100001,
)
# Generic prefill TND uses query-layout digit 3 (compressed MLA uses digit 2).
LAYOUT_KEYS = (
    105000000030000002,
    105000000010200001,
    105000000010200002,
    103000000030000003,
    104000000030000003,
    # Generic BSND PA: query/KV layout digit 1, PA bit 2, BRC bit 4.
    # The generated +100000 variants set the native FD bit, independent of mask.
    103000000010200001,
    104000000010200001,
    103000000010600001,
    104000000010600001,
)
# FIA's empty-KV template uses the same key for both output dtypes; the MLA
# dtype offset does not apply (arch22/fia_tiling_empty_tensor.cpp).
EMPTY_KV_KEY = 100000000000000020


def native_root():
    return Path(os.environ.get("MLA_NATIVE_ROOT", work_dir() / "native")).resolve()


def native_vendor():
    root = native_root()
    manifest = root / "package.json"
    if not manifest.is_file():
        raise RuntimeError(
            "Build the native comparison package: python -m tests.mla.support.native_package"
        )
    data = json.loads(manifest.read_text())
    if data.get("commit") != COMMIT or data.get("recipe") != sha256(Path(__file__)):
        raise RuntimeError("Native package recipe changed; rebuild the comparison package")
    inputs = {p.name: sha256(p) for p in Path(__file__).with_name("native_inputs").glob("*.json")}
    if data.get("inputs") != inputs:
        raise RuntimeError("Native input declarations changed; rebuild the comparison package")
    for relative, digest in data["files"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Native comparison package changed: {path}")
    return root / "opp/vendors" / VENDOR


def run(command, root, name, *, cwd=None, env=None):
    (root / f"{name}.command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (root / f"{name}.log").open("w") as log:
        subprocess.run(
            command,
            cwd=root if cwd is None else cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def build(*, host_only=False):
    root = native_root()
    if root.is_relative_to(ROOT / "build"):
        raise ValueError("Native artifacts must be outside build/ (reserved for headless logs)")
    root.mkdir(parents=True, exist_ok=True)
    (root / "package.json").unlink(missing_ok=True)
    checkout = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    source = root / "source"
    source.mkdir(exist_ok=True)
    archive = subprocess.check_output(["git", "-C", str(checkout), "archive", COMMIT])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    deps = Path(os.environ.get("MLA_NATIVE_DEPS", root / "deps")).resolve()
    deps.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", CMAKE_BUILD_PARALLEL_LEVEL="8")
    run(
        [
            "bash",
            "build.sh",
            "--jit",
            "--noaicpu",
            "--soc=ascend910b",
            "--ops=fused_infer_attention_score",
            "--vendor_name=mla_ca18aee",
            f"--cann_3rd_lib_path={deps}",
            "-j8",
            "-O2",
        ],
        root,
        "host-build",
        cwd=source,
        env=env,
    )
    packages = list((source / "build/_CPack_Packages/Linux/External").glob("*.run"))
    if len(packages) != 1 or not packages[0].is_dir():
        raise RuntimeError("Expected one unpacked native JIT installer")
    run(
        ["bash", "install.sh", "--quiet", f"--install-path={root / 'opp'}"],
        root,
        "host-install",
        cwd=packages[0],
        env=env,
    )
    if host_only:
        return
    vendor = root / "opp/vendors" / VENDOR
    kernel = vendor / "op_impl/ai_core/tbe/kernel"
    target = kernel / "ascend910b/fused_infer_attention_score"
    target.mkdir(parents=True, exist_ok=True)
    # All layouts share one binary per dtype, as required by the provenance checker.
    for dtype in ("fp16", "bf16"):
        offset = 0 if dtype == "fp16" else 22220
        keys = sorted(
            {k + offset for k in DENSE_KEYS}
            | {k + offset + causal for k in LAYOUT_KEYS for causal in (0, 100000)}
            | {EMPTY_KV_KEY}
        )
        output = root / "device" / dtype
        output.mkdir(parents=True, exist_ok=True)
        command = [
            "opc",
            str(
                vendor / f"op_impl/ai_core/tbe/{VENDOR}_impl/dynamic/fused_infer_attention_score.py"
            ),
            "--main_func=fused_infer_attention_score",
            f"--input_param={Path(__file__).with_name('native_inputs') / (dtype + '.json')}",
            "--soc_version=Ascend910B3",
            f"--output={output}",
            "--impl_mode=high_performance,optional",
            "--simplified_key_mode=0",
            "--op_mode=dynamic",
            "--tiling_key=" + ",".join(map(str, keys)),
        ]
        run(
            command,
            root,
            f"device-{dtype}",
            env=dict(
                env,
                ASCEND_CUSTOM_OPP_PATH=str(vendor),
                ASCEND_CACHE_PATH=str(root / "cache" / dtype),
                ASCEND_PROCESS_LOG_PATH=str(root / "compile-logs" / dtype),
            ),
        )
        metadata = list(output.glob("*.json"))
        if len(metadata) != 1:
            raise RuntimeError(f"Expected one compiled {dtype} binary")
        info = json.loads(metadata[0].read_text())
        binary = metadata[0].with_suffix(".o")
        if sha256(binary) != info["sha256"] or not set(keys) <= {
            int(k["tilingKey"]) for k in info["kernelList"]
        }:
            raise RuntimeError(f"Incomplete compiled {dtype} key set")
        for file in (metadata[0], binary):
            shutil.copy2(file, target / file.name)
    run(
        [
            sys.executable,
            str(source / "cmake/scripts/util/ascendc_ops_config.py"),
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
        source_archive_sha256=hashlib.sha256(archive).hexdigest(),
        recipe=sha256(Path(__file__)),
        inputs={
            p.name: sha256(p) for p in Path(__file__).with_name("native_inputs").glob("*.json")
        },
        files={
            str(p.relative_to(root)): sha256(p) for p in sorted(vendor.rglob("*")) if p.is_file()
        },
    )
    (root / "package.json").write_text(json.dumps(record, indent=2) + "\n")
    print(native_vendor())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host-only",
        action="store_true",
        help="Prepare host package; not usable as acceptance evidence",
    )
    args = parser.parse_args()
    build(host_only=args.host_only)


if __name__ == "__main__":
    main()
