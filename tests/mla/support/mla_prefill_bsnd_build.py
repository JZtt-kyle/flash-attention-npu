"""Reuse pinned CPU source oracles; build only this module's adapter harness."""

import json
import hashlib
import os
from pathlib import Path
import subprocess
from tests.mla.support.mla_split_kv_build import compile_drivers as oracles, MLA, ROOT
from tests.mla.support.mla_split_kv_build import (
    split_toolchain_identity,
    CANN_COMMIT,
    FIA,
    source_function,
)
from tests.mla.support.mla_validation_build import files_identity


def generic_source():
    root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    return subprocess.check_output(
        ["git", "-C", str(root), "show", CANN_COMMIT + ":" + FIA + "fia_tiling_nonquant.cpp"],
        timeout=30,
    )


def toolchain_identity(sanitize):
    return dict(
        **split_toolchain_identity(sanitize),
        generic_tiling_sha256=hashlib.sha256(generic_source()).hexdigest(),
    )


def compile_drivers(directory):
    _, native, _ = oracles(directory)
    command = json.loads((directory / "adapter-compile.json").read_text())
    begin = next(i for i, word in enumerate(command) if word.endswith("mla_split_kv.cpp"))
    command = command[:begin] + [
        str(MLA / "mla_prefill_bsnd_metadata.cpp"),
        str(MLA / "detail/cann_split_core.cpp"),
        str(ROOT / "tests/mla/support/mla_prefill_bsnd_driver.cpp"),
        "-o",
        str(directory / "prefill"),
    ]
    (directory / "prefill-command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / "prefill-compile.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    blob = generic_source()
    (directory / "fia_tiling_nonquant.cpp").write_bytes(blob)
    (directory / "generic_tiling.inc").write_text(
        "\n".join(
            source_function(blob.decode(), "FiaTilingNonQuant::" + name + "(")
            for name in ("CalcInnerSize", "CalcMBaseSize")
        )
    )
    native_command = json.loads((directory / "native-compile.json").read_text())
    begin = next(i for i, word in enumerate(native_command) if word.endswith("split_core.cpp"))
    native_command = native_command[:begin] + [
        str(ROOT / "tests/mla/support/mla_prefill_tiling_source_driver.cpp"),
        "-o",
        str(directory / "generic"),
    ]
    (directory / "generic-command.json").write_text(json.dumps(native_command, indent=2) + "\n")
    result = subprocess.run(native_command, capture_output=True, text=True, timeout=120)
    (directory / "generic-compile.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    binaries = [directory / "prefill", native, directory / "generic"]
    (directory / "prefill-binaries.json").write_text(
        json.dumps(files_identity(binaries), indent=2) + "\n"
    )
    (directory / "prefill-toolchain.json").write_text(
        json.dumps(toolchain_identity(os.environ.get("MLA_SCHEDULE_SANITIZE") == "1"), indent=2)
        + "\n"
    )
    return binaries
