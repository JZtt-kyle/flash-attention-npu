"""Validate the production metadata consumer, including real runtime errors."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata import probe as probe
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata_dense import (
    check_metadata_case,
    configure,
    consumers as _independent_consumers,
    make_metadata_case,
    test_checked_runtime_error as check_runtime_error,
)

independent_consumers = _independent_consumers


@pytest.fixture(scope="module")
def consumers(independent_consumers):
    directory = independent_consumers[1] / "production"
    directory.mkdir()
    extension = Path(api.flash_attn_npu_3.__file__).resolve()
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    source = ROOT / "tests/mla/support/mla_production_probe.cpp"
    library = directory / "bridge.so"
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        f"-I{ROOT}/csrc/ascend910/flash_attn_npu_3",
        f"-I{cann}/include",
        f"-I{cann}/include/experiment/runtime",
        f"-I{cann}/include/experiment/msprof",
        f"-I{cann}/pkg_inc/runtime",
        str(source),
        str(extension),
        f"-L{cann}/lib64",
        "-lruntime",
        "-o",
        str(library),
    ]
    (directory / "build-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    manifest = json.loads((independent_consumers[1] / "build-manifest.json").read_text())
    manifest["libraries"]["metadata_checked"] = dict(path=str(library), sha256=sha256(library))
    manifest["production_extension"] = dict(path=str(extension), sha256=sha256(extension))
    for path in (source, Path(__file__), ROOT / "tests/mla/support/mla_metadata_error_probe.py"):
        manifest["sources"][str(path.relative_to(ROOT))] = sha256(path)
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return configure(
        {name: row["path"] for name, row in manifest["libraries"].items()},
        directory,
        independent_consumers[0]["dtype"],
    )


@pytest.mark.parametrize("shape", [(2, 1, 8, 257), (1, 7, 16, 65), (21, 2, 128, 511)])
@pytest.mark.parametrize("scale", [None, 0.0, -0.23, 0.23])
def test_production_metadata_source_equality(consumers, probe, shape, scale):
    check_metadata_case(consumers, make_metadata_case(consumers, probe, shape, scale=scale))


@pytest.mark.parametrize("fault", ["magic", "schedule_failure", "shape", "workspace_high_word"])
def test_production_runtime_error(consumers, probe, fault):
    check_runtime_error(consumers, probe, fault)
