"""Build/reuse the existing release probes and identify the loaded NPU runtime."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from tests.mla.support.mla_dense_build import ROOT, build_probes, sha256
from tests.mla.support.mla_metadata_build import build_probe
from tests.mla.support.mla_validation_dependencies import python_dependencies


def files_identity(paths):
    return {str(p.resolve()): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def build_inputs():
    paths = [ROOT / "setup.py", Path(__file__)]
    paths += list((ROOT / "csrc/ascend910").rglob("*"))
    paths += list((ROOT / "csrc/catlass/include").rglob("*"))
    # Only builders used by prepare() belong to production build identity.
    # CPU oracle/profile helpers have their own manifests and must not force
    # a production rebuild after an unrelated test edit.
    baseline = {str(p.relative_to(ROOT)): None for p in (ROOT / "tests").rglob("*.py")}
    baseline["setup.py"] = None
    paths += [
        ROOT / path
        for path in python_dependencies(
            ROOT, ["tests/mla/support/mla_validation_build.py"], baseline
        )
    ]
    for suffix in ("*.cpp", "*.h", "*.aicpu"):
        paths += list((ROOT / "tests/mla/support").glob(suffix))
    cann = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    return dict(
        sources=files_identity(paths),
        python=sys.version,
        packages={n: importlib.metadata.version(n) for n in ("torch", "torch_npu", "setuptools")},
        cann=str(cann),
        compiler=files_identity([cann / "bin/bisheng", Path("/usr/bin/g++")]),
        compiler_version=subprocess.check_output(
            [str(cann / "bin/bisheng"), "--version"], text=True
        ),
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                (
                    "MLA_CANN_SOURCE_ROOT",
                    "ASCEND_",
                    "CMAKE_",
                    "CXX",
                    "CC",
                    "CPATH",
                    "LIBRARY_PATH",
                    "LD_LIBRARY_PATH",
                    "FLASH_ATTN_",
                )
            )
        },
    )


def matching_files(identity):
    try:
        return all(
            Path(p).is_file() and sha256(Path(p)) == digest for p, digest in identity.items()
        )
    except OSError:
        return False


def prepare(directory):
    """Reuse only a successful build with unchanged source, dependencies and outputs."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "prepare.json"
    current = build_inputs()
    if manifest.exists():
        try:
            old = json.loads(manifest.read_text())
        except (OSError, ValueError):
            old = {}
        if (
            old.get("inputs") == current
            and old.get("dependencies")
            and old.get("outputs")
            and matching_files(old["dependencies"])
            and matching_files(old["outputs"])
        ):
            print("Reusing verified production extension and PA probes", flush=True)
            return
    manifest.unlink(missing_ok=True)
    # A cache miss may be a compiler/header change with preserved mtimes.
    # setup's usual timestamp shortcut cannot establish freshness in that case.
    build_log = directory / "production-build.log"
    with build_log.open("w") as stream:
        subprocess.run(
            [
                sys.executable,
                "setup.py",
                "build_ext",
                "--inplace",
                "--build-temp",
                str(directory / "production/temp"),
                "--build-lib",
                str(directory / "production/lib"),
            ],
            cwd=ROOT,
            env=dict(os.environ, FLASH_ATTN_FORCE_REBUILD="TRUE"),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    links = [
        shlex.split(line.removeprefix("[link-cmd] "))
        for line in build_log.read_text().splitlines()
        if line.startswith("[link-cmd] ")
    ]
    if len(links) != 1:
        raise RuntimeError(f"Expected one FA3 link command; see {build_log}")
    # Historical builds leave obsolete .d files behind. Only the objects in
    # this actual link command may contribute production dependencies.
    objects = [
        Path(word) if Path(word).is_absolute() else ROOT / word
        for word in links[0]
        if word.endswith(".o")
    ]
    if not objects:
        raise RuntimeError("The production link command contains no objects")
    for dtype in ("fp16", "bf16"):
        build_probes(directory / "probes" / dtype, dtype, paged=True)
    build_probe(directory / "producer", paged=True)
    dependencies = []
    depfiles = list(directory.rglob("*.d"))
    depfiles += [Path(str(p) + ".d") for p in objects]
    depfiles += [Path(str(p) + ".device.d") for p in objects if p.name.startswith("mla_dispatch_")]
    for depfile in depfiles:
        words = depfile.read_text().replace("\\\n", " ").split(":", 1)[1].split()
        dependencies += [Path(word) if Path(word).is_absolute() else ROOT / word for word in words]
    # Dependency files contain generated oracle headers too. Their bytes, not
    # only toolkit version strings or object mtimes, guard reuse.
    missing = [str(p) for p in dependencies if not p.is_file()]
    if missing:
        raise RuntimeError(f"Missing build dependencies: {missing[:5]}")
    outputs = list(directory.rglob("*.so")) + list(directory.rglob("build-manifest.json"))
    outputs += list((ROOT / "flash_attn_npu_3").glob("*.so"))
    if not outputs:
        raise RuntimeError("Build produced no libraries")
    if build_inputs() != current:
        raise RuntimeError("Build inputs changed while compiling; rerun preparation")
    manifest.write_text(
        json.dumps(
            dict(
                inputs=current,
                dependencies=files_identity(dependencies),
                outputs=files_identity(outputs),
            ),
            indent=2,
        )
        + "\n"
    )


def runtime():
    import torch
    import torch_npu

    extension = importlib.import_module("flash_attn_npu_3.flash_attn_npu_3")
    path = Path(extension.__file__).resolve()
    if path.parent != ROOT / "flash_attn_npu_3":
        raise RuntimeError(f"Unexpected installed extension: {path}")
    torch.npu.set_device(0)
    torch.npu.synchronize()
    libraries = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        candidate = line.split()[-1]
        if candidate.startswith("/") and ".so" in Path(candidate).name:
            libraries.add(Path(candidate))
    return dict(
        python=sys.version,
        executable=sys.executable,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        chip=torch.npu.get_device_name(0),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        extension=str(path),
        extension_sha256=sha256(path),
        loaded_libraries=files_identity(libraries),
    )


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(sys.argv[2])
    elif sys.argv[1] == "runtime":
        Path(sys.argv[2]).write_text(json.dumps(runtime(), indent=2) + "\n")
    else:
        raise SystemExit("Expected prepare or runtime")
