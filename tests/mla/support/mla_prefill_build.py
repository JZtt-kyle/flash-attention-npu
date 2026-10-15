"""Cached component builds with current sources, device dependencies and outputs."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess

from tests.mla.paths import work_dir
from tests.mla.support.mla_dense_build import ROOT, build_probes, sha256
from tests.mla.support.mla_validation_build import matching_files


def valid(manifest):
    try:
        return (
            manifest.get("prefill_merge") is True
            and manifest.get("dtype") in ("fp16", "bf16")
            and bool(manifest["sources"])
            and matching_files({str(ROOT / k): v for k, v in manifest["sources"].items()})
            and bool(manifest.get("dependencies"))
            and matching_files(manifest["dependencies"])
            and bool(manifest["libraries"])
            and matching_files({v["path"]: v["sha256"] for v in manifest["libraries"].values()})
        )
    except (AttributeError, KeyError, TypeError, OSError):
        return False


def build_prefill(dtype):
    if dtype not in ("fp16", "bf16"):
        raise ValueError("Unsupported prefill probe dtype")
    directory = work_dir() / "prefill-probes" / dtype
    path = directory / "build-manifest.json"
    if path.is_file():
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            manifest = None
        if (
            valid(manifest)
            and manifest["dtype"] == dtype
            and manifest.get("cann") == os.environ["ASCEND_HOME_PATH"]
            and shutil.which("g++") in manifest["dependencies"]
        ):
            return manifest
    libraries = build_probes(directory, dtype, prefill=True)
    manifest = json.loads(path.read_text())
    probe = ROOT / "tests/mla/support/mla_split_kv_merge_probe.cpp"

    def compile_merge(name):
        command = json.loads((directory / f"{name}-command.json").read_text())
        command[command.index(str(ROOT / "tests/mla/support/mla_prefill_probe.cpp"))] = str(probe)
        command.insert(1, "-DMLA_PREFILL")
        command[command.index("-MF") + 1] = str(directory / f"merge-{name}.d")
        command[command.index("-o") + 1] = str(directory / f"merge-{name}.so")
        (directory / f"merge-{name}-command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (directory / f"merge-{name}-compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        deps = [
            a
            for a in command[: command.index("-MMD")]
            if a != "-shared" and not a.startswith(("-l", "-L", "-Wl,"))
        ]
        depfile = directory / f"merge-{name}-device.d"
        deps += ["--offload-device-only", "-M", "-MF", str(depfile)]
        with (directory / f"merge-{name}-deps.log").open("w") as log:
            subprocess.run(deps, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
        assert ("oracle-source" in depfile.read_text()) == (name == "oracle")
        return str(directory / f"merge-{name}.so")

    with ThreadPoolExecutor(max_workers=2) as pool:
        libraries.update(
            zip(("merge_oracle", "merge_migrated"), pool.map(compile_merge, ("oracle", "migrated")))
        )
    for depfile in directory.glob("*-device.d"):
        for item in depfile.read_text().replace("\\\n", " ").split()[1:]:
            p = Path(item).resolve()
            manifest["dependencies"][str(p)] = sha256(p)
    for p in (Path(__file__).resolve(), probe):
        manifest["sources"][str(p.relative_to(ROOT))] = sha256(p)
    # Header dependencies alone cannot detect a toolchain replacement.
    for compiler in (
        Path(os.environ["ASCEND_HOME_PATH"]) / "bin/bisheng",
        Path(shutil.which("g++")),
    ):
        manifest["dependencies"][str(compiler)] = sha256(compiler)
    manifest["prefill_merge"] = True
    manifest["libraries"] = {k: dict(path=v, sha256=sha256(Path(v))) for k, v in libraries.items()}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
