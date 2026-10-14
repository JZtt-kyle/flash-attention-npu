"""One fresh compilation and driver lifecycle per module acceptance process."""

import contextlib
import os
from pathlib import Path

import pytest

from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_auto_split_mixed_build import compile_drivers


@pytest.fixture(scope="session")
def mixed_drivers(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("mixed-protocol")
    )
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CpuDriver(p)) for p in compile_drivers(directory)]


@pytest.fixture(scope="session")
def drivers(mixed_drivers):
    plan, _, native, contracts = mixed_drivers
    for got in plan.evaluate([[2, 0, 0], [3, 0, 0], [4, 0, 0, 0]]):
        assert got["status"] != 0 and got["unchanged"]
    return plan, native, contracts


@pytest.fixture(scope="session")
def metadata_drivers(mixed_drivers):
    _, metadata, native, contracts = mixed_drivers
    return metadata, native, contracts
