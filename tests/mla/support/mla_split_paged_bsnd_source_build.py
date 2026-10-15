"""Build the pinned PA+FD partial specialization beside the fixed FD probe."""

import json
from pathlib import Path
import subprocess
from tests.mla.support.mla_dense_build import ROOT, sha256


def build_paged_partial(directory):
    directory = Path(directory)
    partial = directory / "partial"
    probe = ROOT / "tests/mla/support/mla_split_paged_bsnd_attention_probe.cpp"
    command = json.loads((partial / "oracle-command.json").read_text())
    command[command.index(str(ROOT / "tests/mla/support/mla_split_kv_attention_probe.cpp"))] = str(
        probe
    )
    command[command.index("-o") + 1] = str(directory / "paged-oracle.so")
    command[command.index("-MF") + 1] = str(directory / "paged-oracle.d")
    (directory / "paged-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "paged-compile.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    deps = [
        a
        for a in command[: command.index("-MMD")]
        if a != "-shared" and not a.startswith(("-l", "-L", "-Wl,"))
    ]
    deps += ["--offload-device-only", "-M", "-MF", str(directory / "paged-device.d")]
    with (directory / "paged-deps.log").open("w") as log:
        subprocess.run(deps, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
    dependencies = (directory / "paged-device.d").read_text()
    assert (
        "fia_kernel_nonquant_mla.h" in dependencies
        and "fia_block_cube_nonquant_mla.h" in dependencies
    )
    inputs = {
        str(Path(n).resolve()): sha256(Path(n))
        for n in dependencies.replace("\\\n", " ").split()[1:]
    }
    inputs[str(Path(__file__).resolve())] = sha256(Path(__file__))
    library = directory / "paged-oracle.so"
    (directory / "paged-build-manifest.json").write_text(
        json.dumps(dict(inputs=inputs, library=str(library), sha256=sha256(library)), indent=2)
        + "\n"
    )
    return library
