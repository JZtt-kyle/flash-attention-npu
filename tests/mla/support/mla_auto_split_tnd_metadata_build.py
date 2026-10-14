"""Fresh metadata executable and the existing immutable CANN/FA3 CPU oracles."""

import hashlib
import json
import subprocess

from tests.mla.support.mla_split_kv_build import compile_drivers as compile_oracles, MLA, ROOT


def compile_drivers(directory):
    _, native, contracts = compile_oracles(directory)
    # Reuse exactly the checked compiler/sanitizer options and source closure.
    # The older adapter is built by the oracle helper but is never executed or
    # counted as unit acceptance for this module.
    command = json.loads((directory / "adapter-compile.json").read_text())
    previous = ROOT / "tests/mla/support/mla_split_kv_driver.cpp"
    command[command.index(str(previous))] = str(
        ROOT / "tests/mla/support/mla_auto_split_tnd_metadata_driver.cpp"
    )
    command.insert(command.index("-o"), str(MLA / "mla_auto_split_tnd_plan.cpp"))
    for name in ("mla_auto_split_tnd_metadata.cpp", "mla_used_metadata.cpp", "mla_metadata.cpp"):
        command.insert(command.index("-o"), str(MLA / name))
    binary = directory / "auto_tnd_metadata"
    command[command.index("-o") + 1] = str(binary)
    (directory / "metadata-compile.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / "metadata-compile.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    manifest = directory / "binaries.json"
    identities = json.loads(manifest.read_text())
    identities[str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(identities, indent=2) + "\n")
    # Exercise the existing v18 wrapper in this module's shared-protocol tests.
    bsnd_command = list(command)
    bsnd_command[
        bsnd_command.index(str(ROOT / "tests/mla/support/mla_auto_split_tnd_metadata_driver.cpp"))
    ] = str(ROOT / "tests/mla/support/mla_auto_split_bsnd_metadata_driver.cpp")
    for tnd, bsnd in (
        ("mla_auto_split_tnd_plan.cpp", "mla_auto_split_kv_plan.cpp"),
        ("mla_auto_split_tnd_metadata.cpp", "mla_auto_split_bsnd_metadata.cpp"),
    ):
        bsnd_command[bsnd_command.index(str(MLA / tnd))] = str(MLA / bsnd)
    bsnd_binary = directory / "bsnd_metadata"
    bsnd_command[bsnd_command.index("-o") + 1] = str(bsnd_binary)
    (directory / "bsnd-compile.json").write_text(json.dumps(bsnd_command, indent=2) + "\n")
    bsnd_result = subprocess.run(bsnd_command, capture_output=True, text=True, timeout=120)
    (directory / "bsnd-compile.log").write_text(bsnd_result.stdout + bsnd_result.stderr)
    bsnd_result.check_returncode()
    identities[str(bsnd_binary)] = hashlib.sha256(bsnd_binary.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(identities, indent=2) + "\n")
    return binary, native, contracts, bsnd_binary
