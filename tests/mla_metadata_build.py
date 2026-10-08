"""Build the actual AICPU source with setup.py's compiler/dependency policy."""

import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import types

ROOT = Path(__file__).resolve().parents[1]
MLA = ROOT / "csrc/ascend910/flash_attn_npu_3/mla"


def metadata_builder(force=False):
    # Avoid setup() and NPU initialization when testing build policy on CPU.
    tree = ast.parse((ROOT / "setup.py").read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("get_cann_arch_dir", "_cmdhash", "_obj_is_fresh")
    ]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BishengBuildExt")
    methods = ("_build_aicpu_metadata", "_build_aicpu_object", "_force_rebuild")
    nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    scope = dict(
        os=os,
        Path=Path,
        hashlib=hashlib,
        subprocess=subprocess,
        platform=platform,
        this_dir=str(ROOT),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "setup.py", "exec"), scope)
    builder = types.SimpleNamespace(force=force)
    for name in methods:
        setattr(builder, name, types.MethodType(scope[name], builder))
    return builder


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_probe(directory, *, bsnd=False):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "metadata.so"
    with (directory / "aicpu-build.log").open("w") as log, contextlib.redirect_stdout(log):
        obj = metadata_builder(force=True)._build_aicpu_object(
            str(output), str(MLA / "mla_metadata.aicpu")
        )
        objects = [obj]
        if bsnd:
            objects.append(
                metadata_builder(force=True)._build_aicpu_object(
                    str(output), str(MLA / "mla_bsnd_metadata.aicpu")
                )
            )
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    command = [
        str(cann / "bin/bisheng"),
        "-x",
        "asc",
        "--npu-arch=dav-2201",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        *(["-DMLA_BSND_METADATA"] if bsnd else []),
        f"-I{MLA.parent}",
        f"-I{cann}/include",
        f"-L{cann}/lib64",
        str(ROOT / "tests/mla_metadata_probe.cpp"),
        str(MLA / "mla_metadata.cpp"),
        str(MLA / "mla_schedule.cpp"),
        str(MLA / "detail/cann_split_core.cpp"),
        *([str(MLA / "mla_bsnd_metadata.cpp"), str(MLA / "mla_bsnd_schedule.cpp")] if bsnd else []),
        "-x",
        "none",
        *objects,
        "-lascendcl",
        "-lruntime",
        "-ltiling_api",
        "-lplatform",
        "-o",
        str(output),
    ]
    (directory / "probe-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "probe-build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    dependencies = Path(obj + ".d").read_text().replace("\\\n", " ").split(":", 1)[1].split()
    for name in (
        "mla_metadata.h",
        "mla_metadata_args.h",
        "mla_metadata.cpp",
        "mla_schedule.h",
        "mla_schedule.cpp",
        "detail/cann_split_core.h",
        "detail/cann_split_core.cpp",
    ):
        assert str(MLA / name) in dependencies
    sources = [ROOT / "setup.py", ROOT / "tests/mla_metadata_probe.cpp", Path(__file__)]
    if bsnd:
        extra_dependencies = (
            Path(objects[1] + ".d").read_text().replace("\\\n", " ").split(":", 1)[1].split()
        )
        for name in (
            "mla_bsnd_metadata.h",
            "mla_bsnd_metadata.cpp",
            "mla_bsnd_schedule.h",
            "mla_bsnd_schedule.cpp",
        ):
            assert str(MLA / name) in extra_dependencies
        dependencies += extra_dependencies
    sources += [Path(p) for p in dependencies if Path(p).is_relative_to(ROOT)]
    manifest = dict(
        library=str(output),
        library_sha256=sha256(output),
        object=str(obj),
        object_sha256=sha256(obj),
        objects={str(p): sha256(p) for p in objects},
        dependencies_sha256=sha256(obj + ".d"),
        sources={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        cann=str(cann),
        compiler=subprocess.check_output([str(cann / "bin/bisheng"), "--version"], text=True),
    )
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
