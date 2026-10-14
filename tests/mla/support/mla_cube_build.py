"""Build isolated FP16 Cube probes from current headers and pinned CANN blobs."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile


from tests.mla.paths import ROOT

COMMIT = "ca18aee610729ca5270a44e2d84c751bc687fbf0"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_probes(artifacts):
    artifacts = Path(artifacts).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    source = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    paths = ["attention/common/op_kernel", "attention/fused_infer_attention_score/op_kernel"]
    archive = subprocess.check_output(["git", "-C", str(source), "archive", COMMIT, *paths])
    oracle = artifacts / "oracle-source"
    oracle.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(oracle, filter="data")
    manifest = {
        "commit": COMMIT,
        "files": {
            str(p.relative_to(oracle)): sha256(p) for p in sorted(oracle.rglob("*")) if p.is_file()
        },
    }
    (artifacts / "oracle-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    common = [
        str(cann / "bin/bisheng"),
        "-x",
        "asc",
        "--npu-arch=dav-2201",
        "--cce-auto-infer-kernel-type=false",
        "-O2",
        "-g",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-DCATLASS_ARCH=2201",
        "-DASC_DEVKIT_MAJOR=9",
        f"-I{cann}/include",
        f"-I{cann}/include/experiment/runtime",
        f"-I{cann}/pkg_inc/runtime",
        f"-I{cann}/pkg_inc",
        f"-I{cann}/pkg_inc/profiling",
        f"-I{cann}/include/experiment/msprof",
        f"-L{cann}/lib64",
        "-lascendcl",
        "-lruntime",
    ]

    def compile_probe(name):
        output = artifacts / f"{name}.so"
        cmd = [*common, str(ROOT / "tests/mla/support/mla_cube_probe.cpp")]
        if name == "oracle":
            base = oracle / "attention/fused_infer_attention_score/op_kernel"
            cmd += ["-DMLA_CUBE_ORACLE", f"-I{base}/arch22", f"-I{base}"]
        else:
            cmd += [f"-I{ROOT}/csrc/catlass/include", f"-I{ROOT}/csrc/ascend910/flash_attn_npu_3"]
        cmd += ["-MMD", "-MF", str(artifacts / f"{name}.d"), "-o", str(output)]
        (artifacts / f"{name}-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        with (artifacts / f"{name}-compile.log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise RuntimeError(f"{name} build failed; see {artifacts / f'{name}-compile.log'}")
        # The final host pass overwrites -MMD with host-only dependencies. The
        # device guards require a separate device preprocessing pass. Every
        # invocation rebuilds both probes; no pre-existing .so is accepted.
        deps = [
            arg
            for arg in cmd[: cmd.index("-MMD")]
            if arg != "-shared" and not arg.startswith(("-l", "-L"))
        ]
        depfile = artifacts / f"{name}-device.d"
        deps += ["--offload-device-only", "-M", "-MF", str(depfile)]
        (artifacts / f"{name}-deps-command.json").write_text(json.dumps(deps, indent=2) + "\n")
        with (artifacts / f"{name}-deps.log").open("w") as log:
            subprocess.run(deps, stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
        dependencies = depfile.read_text()
        expected = (
            ["fia_block_cube_nonquant_mla.h"]
            if name == "oracle"
            else ["mla_cube.hpp", "mla_copy.hpp", "mla_mmad.hpp"]
        )
        assert all(header in dependencies for header in expected)
        assert ("oracle-source" in dependencies) == (name == "oracle")
        return str(output)

    with ThreadPoolExecutor(max_workers=2) as pool:
        libraries = dict(
            zip(("oracle", "migrated"), pool.map(compile_probe, ("oracle", "migrated")))
        )
    current = [
        ROOT / "tests/mla/support/mla_cube_probe.cpp",
        *sorted((ROOT / "csrc/ascend910/flash_attn_npu_3/mla").glob("*/*.hpp")),
    ]
    manifest = {
        "libraries": {
            name: {"path": path, "sha256": sha256(Path(path))} for name, path in libraries.items()
        },
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in current},
        "compiler": subprocess.check_output([str(cann / "bin/bisheng"), "--version"], text=True),
        "cann": str(cann),
        "catlass": subprocess.check_output(
            ["git", "-C", str(ROOT / "csrc/catlass"), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (artifacts / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return libraries


if __name__ == "__main__":
    print(json.dumps(build_probes(os.environ["MLA_CUBE_ARTIFACT_DIR"]), indent=2))
