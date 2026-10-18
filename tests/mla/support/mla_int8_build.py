"""Isolated INT8 oracle and ND component builds, with content identities."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

from tests.mla.paths import ROOT, work_dir
from tests.mla.support.mla_dense_build import COMMIT, sha256
from tests.mla.support.mla_validation_build import matching_files


def build_probes():
    directory = work_dir() / "int8-probes"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build_probes(directory)


def _build_probes(directory):
    manifest_path = directory / "build-manifest.json"
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            previous = {}
        if (
            previous.get("commit") == COMMIT
            and previous.get("cann") == str(cann)
            and previous.get("dependencies")
            and matching_files(previous["dependencies"])
            and previous.get("libraries")
            and matching_files({v["path"]: v["sha256"] for v in previous["libraries"].values()})
        ):
            return previous
    source = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    paths = [
        "attention/incre_flash_attention/op_kernel",
        "attention/incre_flash_attention/op_host/incre_flash_attention_tiling_mla.h",
    ]
    archive = subprocess.check_output(["git", "-C", str(source), "archive", COMMIT, *paths])
    oracle = directory / "oracle-source"
    oracle.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(oracle, filter="data")
    structs = []
    for name, body in re.findall(
        r"BEGIN_TILING_DATA_DEF\((\w+)\)(.*?)END_TILING_DATA_DEF",
        (oracle / paths[-1]).read_text(),
        re.S,
    ):
        fields = []
        for macro, args in re.findall(r"(TILING_DATA_FIELD_DEF(?:_ARR|_STRUCT)?)\(([^)]*)\)", body):
            args = [v.strip() for v in args.split(",")]
            if macro.endswith("_ARR"):
                dtype, count, field = args
                fields.append(f"    {dtype} {field}[{count}];")
            else:
                dtype, field = args
                fields.append(f"    {dtype} {field};")
        structs.append(f"struct alignas(8) {name} {{\n" + "\n".join(fields) + "\n};")
    (directory / "oracle_tiling.h").write_text(
        "#pragma once\n#include <cstdint>\nconstexpr uint32_t MAX_AIC_CORE_NUM = 26;\nstruct IncreFlashAttentionTilingDataV2;\n"
        + "\n".join(structs)
        + "\n"
    )
    base = oracle / paths[0]
    # Same V220 numerical compilation settings as the fixed native recipe.
    # In particular source FD passes stack-resident arrays through pointers.
    common = [
        str(cann / "bin/bisheng"),
        "-x",
        "asc",
        "--npu-arch=dav-2201",
        "--cce-auto-infer-kernel-type=false",
        "--cce-aicore-input-parameter-size=4096",
        "--cce-auto-sync=off",
        "-O3",
        "-g",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-DCATLASS_ARCH=2201",
        "-DASC_DEVKIT_MAJOR=9",
        "-DNDEBUG",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=false",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "-mllvm",
        "-cce-vf-remove-membar=false",
        "-mllvm",
        "-cce-aicore-hoist-movemask=false",
        *[
            f"-I{cann}/{p}"
            for p in (
                "include",
                "include/experiment/runtime",
                "pkg_inc/runtime",
                "pkg_inc",
                "pkg_inc/profiling",
                "include/experiment/msprof",
            )
        ],
        f"-L{cann}/lib64",
        "-lascendcl",
        "-lruntime",
        "-Wl,--wrap=aclrtLaunchKernelWithHostArgs",
        str(ROOT / "tests/mla/support/mla_int8_probe.cpp"),
        f"-I{ROOT}/csrc/ascend910/flash_attn_npu_3",
        f"-I{ROOT}/csrc/catlass/include",
    ]

    def compile_one(name):
        cmd = [*common]
        if name == "oracle":
            cmd += ["-DMLA_INT8_ORACLE", f"-I{base}/arch22", f"-I{base}", f"-I{directory}"]
        (directory / f"{name}-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        with (directory / f"{name}-compile.log").open("w") as log:
            subprocess.run(
                [*cmd, "-o", str(directory / f"{name}.so")],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=240,
            )
        cmd = [x for x in cmd if x != "-shared" and not x.startswith(("-l", "-L", "-Wl,"))]
        depfile = directory / f"{name}.d"
        with (directory / f"{name}-deps.log").open("w") as log:
            subprocess.run(
                [*cmd, "--offload-device-only", "-M", "-MF", str(depfile)],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=90,
            )
        assert ("oracle-source" in depfile.read_text()) == (name == "oracle")
        return dict(path=str(directory / f"{name}.so"), sha256=sha256(directory / f"{name}.so"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        libraries = dict(zip(("oracle", "migrated"), pool.map(compile_one, ("oracle", "migrated"))))
    deps = {Path(__file__).resolve(), cann / "bin/bisheng", oracle / paths[-1]}
    for file in directory.glob("*.d"):
        deps.update(Path(v).resolve() for v in file.read_text().replace("\\\n", " ").split()[1:])
    result = dict(
        commit=COMMIT,
        cann=str(cann),
        libraries=libraries,
        dependencies={str(p): sha256(p) for p in sorted(deps)},
    )
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(build_probes(), indent=2))
