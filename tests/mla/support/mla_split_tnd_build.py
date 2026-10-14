"""Fresh CPU protocol bridge and pinned source oracles for TND production split."""

import hashlib
import json
import subprocess
from tests.mla.support.mla_split_tnd_plan_build import compile_drivers as compile_plan
from tests.mla.support.mla_split_kv_build import MLA, ROOT


def compile_drivers(directory):
    _, native, contracts = compile_plan(directory)
    command = json.loads((directory / "plan-compile.json").read_text())
    old = str(ROOT / "tests/mla/support/mla_split_tnd_plan_driver.cpp")
    command[command.index(old)] = str(ROOT / "tests/mla/support/mla_split_tnd_driver.cpp")
    for name in ("mla_split_tnd_metadata.cpp", "mla_used_metadata.cpp", "mla_metadata.cpp"):
        command.insert(command.index("-o"), str(MLA / name))
    binary = directory / "split_tnd"
    command[command.index("-o") + 1] = str(binary)
    (directory / "split-compile.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / "split-compile.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    manifest = directory / "binaries.json"
    identities = json.loads(manifest.read_text())
    identities[str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(identities, indent=2) + "\n")
    return binary, native, contracts
