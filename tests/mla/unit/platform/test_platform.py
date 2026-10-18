"""910B/C host compatibility without claiming any 910C device execution."""

import configparser
import json
import os
import subprocess

import pytest

from tests.mla.support.mla_platform_build import compile_platform, platform_inputs


SOCS = (
    "Ascend910B1",
    "Ascend910B2",
    "Ascend910B2C",
    "Ascend910B3",
    "Ascend910B4",
    "Ascend910B4-1",
    "Ascend910_9391",
    "Ascend910_9381",
    "Ascend910_9372",
    "Ascend910_9392",
    "Ascend910_9382",
    "Ascend910_9362",
)


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    return compile_platform(
        tmp_path_factory.mktemp("platform"),
        sanitize=os.environ.get("MLA_PLATFORM_SANITIZE") == "1",
    )


def query(driver, name, full=False, mutation="none", config=None):
    return subprocess.run(
        [str(driver), name, config or name, str(int(full)), mutation],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("soc", SOCS)
@pytest.mark.parametrize("full", [False, True])
def test_cann_platform_resources(driver, soc, full):
    _, configs = platform_inputs()
    config = configparser.ConfigParser()
    config.read(configs / f"{soc}.ini")
    assert config["version"].getint("NpuArch") == 2201
    result = query(driver, soc, full)
    assert result.returncode == 0, result.stderr
    # CANN's logger may emit an initialization diagnostic before the payload.
    payloads = [line for line in result.stdout.splitlines() if line.startswith('{"aic":')]
    assert len(payloads) == 1, result.stdout
    p = json.loads(payloads[0])
    assert p["aic"] == config["SoCInfo"].getint("cube_core_cnt")
    assert p["aiv"] == config["SoCInfo"].getint("vector_core_cnt") == 2 * p["aic"]
    assert 0 < p["aic"] <= 26
    physical = [
        config["AICoreSpec"].getint(name)
        for name in ("l1_size", "ub_size", "l0_a_size", "l0_b_size", "l0_c_size")
    ]
    # GetCoreMemSize reports usable bytes after the API's own reservations.
    # BuildDenseSchedule above independently checks the kernel's required budget.
    assert all(0 < usable <= total for usable, total in zip(p["memory"], physical))
    expected = [p["aic"]] * 26 if full else list(range(1, p["aic"] + 1)) + [0] * (26 - p["aic"])
    assert p["blocks"] == expected
    assert p["lib_api"] > 0


@pytest.mark.parametrize(
    "name",
    [
        "null",
        "",
        "Ascend910A",
        "Ascend910ProB",
        "Ascend950DT_9591",
        "Ascend310P3",
        "Ascend910_9591",
    ],
)
def test_unsupported_soc(driver, name):
    result = query(driver, name, config="Ascend910B3")
    assert result.returncode == 1
    assert "requires an Ascend 910B/C device" in result.stderr


def test_original_910b_alias_is_not_c1v2(driver):
    result = query(driver, "Ascend910B")
    assert result.returncode == 1
    assert "requires validated C1V2 resources" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "zero-cores",
        "too-many-cores",
        "invalid-vectors",
        "l1",
        "ub",
        "l0a",
        "l0b",
        "l0c",
        "block-dim",
    ],
)
def test_invalid_resources(driver, mutation):
    result = query(driver, "Ascend910_9391", mutation=mutation)
    assert result.returncode == 1
    assert "schedule rejected resources" in result.stderr
