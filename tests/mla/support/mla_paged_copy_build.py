"""Build isolated PA Copy probes from current headers and pinned CANN blobs."""

from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

from tests.mla.support.mla_cube_build import COMMIT, ROOT, sha256


class Geometry(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in ("page", "batch", "columns", "physical_pages")] + [
        (n, ct.c_uint64) for n in ("table_stride", "k_page", "k_row", "v_page", "v_row")
    ]


def build_probes(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    source = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    archive = subprocess.check_output(
        ["git", "-C", str(source), "archive", COMMIT, "attention/common/op_kernel", "LICENSE"]
    )
    oracle = directory / "oracle-source"
    oracle.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(oracle, filter="data")
    source_manifest = {
        str(p.relative_to(oracle)): sha256(p) for p in oracle.rglob("*") if p.is_file()
    }
    (directory / "oracle-source-manifest.json").write_text(
        json.dumps(dict(commit=COMMIT, files=source_manifest), indent=2) + "\n"
    )
    source_file = ROOT / "tests/mla/support/mla_paged_copy_probe.cpp"

    def compile_one(pair):
        dtype, name = pair
        stem = f"{dtype}-{name}"
        output = directory / f"{stem}.so"
        cmd = [
            str(cann / "bin/bisheng"),
            "-x",
            "asc",
            "--npu-arch=dav-2201",
            "--cce-auto-infer-kernel-type=false",
            "-O2",
            "-g",
            "-DNDEBUG",
            "-std=c++17",
            "-fPIC",
            "-DCATLASS_ARCH=2201",
            "-DASC_DEVKIT_MAJOR=9",
            f"-I{cann}/include",
            str(source_file),
        ]
        if dtype == "bf16":
            cmd += ["-DMLA_PAGED_COPY_BF16"]
        if name == "oracle":
            cmd += ["-DMLA_PAGED_COPY_ORACLE", f"-I{oracle}/attention/common/op_kernel/memcopy"]
        else:
            cmd += [f"-I{ROOT}/csrc/catlass/include", f"-I{ROOT}/csrc/ascend910/flash_attn_npu_3"]
        depfile = directory / f"{stem}-device.d"
        commands = [
            [*cmd, "-shared", f"-L{cann}/lib64", "-lascendcl", "-o", str(output)],
            [*cmd, "--offload-device-only", "-M", "-MF", str(depfile)],
        ]
        (directory / f"{stem}-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with (directory / f"{stem}-build.log").open("w") as log:
            for command in commands:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=240, check=True
                )
        dependencies = [
            Path(p).resolve()
            for p in depfile.read_text().replace("\\\n", " ").split(":", 1)[1].split()
        ]
        if name == "oracle":
            assert (
                oracle / "attention/common/op_kernel/memcopy/attn_copy_gm_to_l1.h" in dependencies
            )
            assert not any("flash_attn_npu_3/mla/" in str(p) for p in dependencies)
        else:
            assert (
                ROOT / "csrc/ascend910/flash_attn_npu_3/mla/tile/mla_paged_copy.hpp" in dependencies
            )
            assert not any(p.is_relative_to(oracle) for p in dependencies)
        return stem, dict(path=str(output), sha256=sha256(output)), dependencies

    pairs = [(dtype, name) for dtype in ("fp16", "bf16") for name in ("oracle", "migrated")]
    libraries, dependencies = {}, {source_file, Path(__file__).resolve()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for stem, library, deps in pool.map(compile_one, pairs):
            libraries[stem] = library
            dependencies.update(deps)
    manifest = dict(
        libraries=libraries,
        dependencies={str(p): sha256(p) for p in sorted(dependencies) if p.is_file()},
        cann=str(cann.resolve()),
        compiler=subprocess.check_output([str(cann / "bin/bisheng"), "--version"], text=True),
        catlass=subprocess.check_output(
            ["git", "-C", str(ROOT / "csrc/catlass"), "rev-parse", "HEAD"], text=True
        ).strip(),
        commit=COMMIT,
        ndebug=True,
        scope="Independent ND paged Copy only; no production PA. Oracle layout stride injection is test-only.",
    )
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_probes(directory, dtype):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    for filename, digest in manifest["dependencies"].items():
        assert sha256(Path(filename)) == digest, filename
    loaded = {}
    for name in ("oracle", "migrated"):
        item = manifest["libraries"][f"{dtype}-{name}"]
        assert sha256(Path(item["path"])) == item["sha256"]
        lib = ct.CDLL(item["path"])
        lib.launch_paged_copy.argtypes = [ct.c_void_p] * 7 + [
            ct.POINTER(Geometry),
            ct.c_uint32,
            ct.c_uint32,
            ct.c_bool,
        ]
        lib.launch_paged_copy.restype = ct.c_int
        loaded[name] = lib
    (directory / f"{dtype}-loaded.json").write_text(
        json.dumps(manifest["libraries"], indent=2) + "\n"
    )
    return loaded


if __name__ == "__main__":
    build_probes(os.environ["MLA_PAGED_COPY_DIR"])
