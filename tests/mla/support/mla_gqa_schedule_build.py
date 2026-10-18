"""Reuse fixed source/compiler recipes for the GQA coordinate component."""

import json
import os
import subprocess

from tests.mla.support.mla_split_kv_build import compile_drivers as split_build
from tests.mla.support.mla_split_kv_build import split_toolchain_identity, MLA, ROOT
from tests.mla.support.mla_window_build import fa3_blob
from tests.mla.support.mla_validation_build import files_identity


def fa3_options_source():
    """Keep the explicit-choice and resource-subtraction statements verbatim.

    CUDA's heuristic is deliberately a stub: None is a backend choice. This
    oracle checks explicit values and subtraction, not a CUDA occupancy claim.
    """
    api = fa3_blob().decode()
    pack = next(line for line in api.splitlines() if "params.pack_gqa = pack_gqa_." in line)
    margin = next(line for line in api.splitlines() if "params.num_sm = " in line)
    return (
        """
// Verbatim FA3 d15f153 statements, BSD 3-Clause; see LICENSE.fa3.
#include <iostream>
#include <optional>
struct Params { bool pack_gqa; int num_sm; };
struct Device { int multiProcessorCount; } device;
namespace at::cuda { Device* getCurrentDeviceProperties() { return &device; } }
bool backend_default;
bool get_pack_gqa(const Params&) { return backend_default; }
int main() {
    int pack, physical, sm_margin, automatic;
    while (std::cin >> pack >> physical >> sm_margin >> automatic) {
        Params params{};
        std::optional<bool> pack_gqa_ = pack < 0 ? std::nullopt : std::optional<bool>(pack);
        device.multiProcessorCount = physical;
        backend_default = automatic;
"""
        + pack
        + "\n"
        + margin
        + """
        std::cout << '[' << params.pack_gqa << ',' << params.num_sm << "]\\n" << std::flush;
    }
}
"""
    )


def compile_drivers(directory):
    _, native, _ = split_build(directory)
    command = json.loads((directory / "adapter-compile.json").read_text())
    driver = str(ROOT / "tests/mla/support/mla_split_kv_driver.cpp")
    index = command.index(driver)
    command[index:] = [
        str(MLA / "mla_gqa_schedule.cpp"),
        str(ROOT / "tests/mla/support/mla_gqa_schedule_driver.cpp"),
        "-o",
        str(directory / "gqa"),
    ]
    source = directory / "fa3_gqa_options.cpp"
    source.write_text(fa3_options_source())
    oracle = json.loads((directory / "contracts-compile.json").read_text())
    oracle[-3:] = [str(source), "-o", str(directory / "fa3-gqa")]
    for name, cmd in (("gqa", command), ("fa3-gqa", oracle)):
        (directory / f"{name}-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        (directory / f"{name}-compile.log").write_text(run.stdout + run.stderr)
        run.check_returncode()
    binaries = [directory / "gqa", native, directory / "fa3-gqa"]
    (directory / "gqa-binaries.json").write_text(
        json.dumps(files_identity(binaries), indent=2) + "\n"
    )
    (directory / "gqa-toolchain.json").write_text(
        json.dumps(
            split_toolchain_identity(os.environ.get("MLA_SCHEDULE_SANITIZE") == "1"), indent=2
        )
        + "\n"
    )
    return binaries
