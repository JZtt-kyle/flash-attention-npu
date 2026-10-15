"""Fresh paged_tnd plan/protocol executables and pinned CANN/FA3 source oracles."""

import hashlib
import json
import os
import subprocess

from tests.mla.support.mla_split_kv_build import (
    compile_drivers as compile_oracles,
    MLA,
    ROOT,
)
from tests.mla.support.mla_split_paged_bsnd_build import paged_split_toolchain_identity


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
            ROOT / f"tests/mla/support/mla_split_paged_tnd_{kind}_driver.cpp"
        )
        sources = [
            "mla_split_paged_tnd_plan.cpp",
            "mla_split_mixed_plan.cpp",
            "mla_split_kv_plan.cpp",
        ]
        if kind == "metadata":
            sources += [
                "mla_split_paged_tnd_metadata.cpp",
                "mla_paged_metadata.cpp",
                "mla_metadata.cpp",
            ]
        for name in sources:
            command.insert(command.index("-o"), str(MLA / name))
        binary = directory / kind
        command[command.index("-o") + 1] = str(binary)
        (directory / f"paged_tnd-{kind}-compile.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (directory / f"paged_tnd-{kind}-compile.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        binaries.append(binary)
    manifest = directory / "binaries.json"
    identities = json.loads(manifest.read_text())
    identities.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries})
    manifest.write_text(json.dumps(identities, indent=2) + "\n")
    from tests.mla.unit.tnd.test_flash_attn_npu_v3_mla_tnd_schedule import build_oracle

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
