"""Build validation-only release probes and record actual device dependencies."""

import json
import os
from pathlib import Path
import subprocess

from tests.mla_metadata_build import MLA, ROOT, sha256


def build_consumers(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    source = ROOT / "tests/mla_bsnd_metadata_consumer_probe.cpp"
    libraries, sources = {}, {source, Path(__file__)}
    for dtype in ("fp16", "bf16"):
        output = directory / f"{dtype}.so"
        command = [
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
            f"-I{cann}/include",
            f"-I{MLA.parent}",
            *(["-DMLA_BSND_BF16"] if dtype == "bf16" else []),
            str(source),
        ]
        compile_command = [*command, "-shared", f"-L{cann}/lib64", "-lascendcl", "-o", str(output)]
        deps = directory / f"{dtype}-device.d"
        dependency_command = [*command, "--offload-device-only", "-M", "-MF", str(deps)]
        commands = [compile_command, dependency_command]
        (directory / f"{dtype}-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with (directory / f"{dtype}-build.log").open("w") as log:
            for cmd in commands:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        dependencies = [
            Path(p).resolve()
            for p in deps.read_text().replace("\\\n", " ").split(":", 1)[1].split()
        ]
        for name in (
            "mla_bsnd_metadata_consumer.h",
            "mla_metadata_consumer.h",
            "kernel/mla_bsnd_metadata.hpp",
        ):
            assert MLA / name in dependencies
        sources.update(p for p in dependencies if p.is_relative_to(ROOT))
        libraries[dtype] = dict(path=str(output), sha256=sha256(output))
    manifest = dict(
        libraries=libraries,
        sources={str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)},
        compiler=subprocess.check_output([str(cann / "bin/bisheng"), "--version"], text=True),
        cann=str(cann),
        ndebug=True,
        scope="v2 validation only; no attention forward",
    )
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
