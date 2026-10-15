"""Fresh paged_bsnd plan/protocol executables and pinned CANN/FA3 source oracles."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from tests.mla.support.mla_split_kv_build import (
    compile_drivers as compile_oracles,
    MLA,
    ROOT,
    split_toolchain_identity,
)
from tests.mla.support.components.tnd_schedule import PARSER, SOURCE_FILES, CANN_COMMIT


def paged_split_toolchain_identity(sanitize):
    identity = split_toolchain_identity(sanitize)
    root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    identity["paged_source_blobs"] = {
        name: hashlib.sha256(
            subprocess.check_output(
                ["git", "-C", str(root), "show", CANN_COMMIT + ":" + name], timeout=30
            )
        ).hexdigest()
        for name in [*SOURCE_FILES, PARSER]
    }
    return identity


def compile_drivers(directory):
    _, native, contracts = compile_oracles(directory)
    # Reuse the checked compiler, sanitizer and source closure. The older
    # adapter is not executed or counted as this module's acceptance.
    base = json.loads((directory / "adapter-compile.json").read_text())
    previous = str(ROOT / "tests/mla/support/mla_split_kv_driver.cpp")
    binaries = []
    for kind in ("plan", "metadata"):
        command = list(base)
        command[command.index(previous)] = str(
            ROOT / f"tests/mla/support/mla_split_paged_bsnd_{kind}_driver.cpp"
        )
        sources = ["mla_split_paged_bsnd_plan.cpp", "mla_split_kv_plan.cpp"]
        if kind == "metadata":
            sources += [
                "mla_split_paged_bsnd_metadata.cpp",
                "mla_paged_metadata.cpp",
                "mla_metadata.cpp",
            ]
        for name in sources:
            command.insert(command.index("-o"), str(MLA / name))
        binary = directory / kind
        command[command.index("-o") + 1] = str(binary)
        (directory / f"paged_bsnd-{kind}-compile.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (directory / f"paged_bsnd-{kind}-compile.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        binaries.append(binary)
    manifest = directory / "binaries.json"
    identities = json.loads(manifest.read_text())
    identities.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries})
    manifest.write_text(json.dumps(identities, indent=2) + "\n")
    from tests.mla.support.components.tnd_schedule import build_oracle

    paged = build_oracle(directory, used=True, paged=True)
    identities[str(paged)] = hashlib.sha256(paged.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(identities, indent=2) + "\n")
    (directory / "toolchain.json").write_text(
        json.dumps(
            paged_split_toolchain_identity(os.environ.get("MLA_SCHEDULE_SANITIZE") == "1"), indent=2
        )
        + "\n"
    )
    return *binaries, native, contracts, paged
