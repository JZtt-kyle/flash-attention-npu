"""Build the production GQA record harness with the existing fixed-source recipe."""

import json
import subprocess
from tests.mla.support.mla_gqa_schedule_build import compile_drivers as schedule_build
from tests.mla.support.mla_split_kv_build import MLA, ROOT
from tests.mla.support.mla_validation_build import files_identity


def compile_drivers(directory):
    _, native, fa3 = schedule_build(directory)
    command = json.loads((directory / "gqa-command.json").read_text())
    i = command.index(str(ROOT / "tests/mla/support/mla_gqa_schedule_driver.cpp"))
    command[i:] = [
        str(MLA / "mla_gqa_metadata.cpp"),
        str(ROOT / "tests/mla/support/mla_gqa_options_driver.cpp"),
        "-o",
        str(directory / "gqa-options"),
    ]
    (directory / "gqa-options-command.json").write_text(json.dumps(command, indent=2) + "\n")
    run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / "gqa-options-compile.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    binaries = [directory / "gqa-options", native, fa3]
    (directory / "gqa-options-binaries.json").write_text(
        json.dumps(files_identity(binaries), indent=2) + "\n"
    )
    return binaries
