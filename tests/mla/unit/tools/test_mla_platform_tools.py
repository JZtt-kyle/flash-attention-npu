"""Platform checks use CPU execution, tracked CANN inputs and normal failures."""

import argparse
from pathlib import Path

import pytest

from tests.mla.support import mla_validation as driver
from tests.mla.unit.tools.test_mla_validation import (
    test_success_failure_interruption_and_resume as check_lifecycle,
)


@pytest.mark.parametrize("suite", ["unit-platform", "platform-sanitize"])
def test_execution_and_failure(tmp_path, monkeypatch, suite):
    assert suite in driver.CPU_SUITES
    assert not driver.is_e2e(driver.SUITES[suite])
    env = driver.environment(
        argparse.Namespace(device="7", golden="cache", artifacts=tmp_path), tmp_path, suite
    )
    assert env["MLA_VALIDATION_NPU"] == "0"
    assert env["MLA_PLATFORM_SANITIZE"] == str(int(suite.endswith("sanitize")))
    check_lifecycle(tmp_path, monkeypatch, suite)


def test_platform_database_identity():
    identity = driver.suite_dependencies("unit-platform")
    sanitized = driver.suite_dependencies("platform-sanitize")
    assert identity["platform_files"] == sanitized["platform_files"]
    assert not identity["sanitize"] and sanitized["sanitize"]
    assert any("libasan" in p for p in sanitized["binaries"])
    assert any("libubsan" in p for p in sanitized["binaries"])
    names = {Path(p).name for p in identity["platform_files"]}
    assert {"Ascend910B3.ini", "Ascend910_9391.ini", "libplatform.so", "libtiling_api.a"} <= names
    assert all(len(digest) == 64 for digest in identity["platform_files"].values())
