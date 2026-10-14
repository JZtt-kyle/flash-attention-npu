"""Fresh TND plan executable and the existing immutable CANN/FA3 CPU oracles."""

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
        ROOT / "tests/mla/support/mla_split_tnd_plan_driver.cpp"
    )
    command.insert(command.index("-o"), str(MLA / "mla_split_tnd_plan.cpp"))
    binary = directory / "plan"
    command[command.index("-o") + 1] = str(binary)
    (directory / "plan-compile.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / "plan-compile.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    manifest = directory / "binaries.json"
    identities = json.loads(manifest.read_text())
    identities[str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(identities, indent=2) + "\n")
    return binary, native, contracts
