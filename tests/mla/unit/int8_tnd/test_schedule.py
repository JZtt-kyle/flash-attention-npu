"""TND production scheduling: fixed native geometry and checked CPU bounds."""

import ctypes as ct
import json
import os
import itertools
from pathlib import Path
import subprocess

import pytest

from tests.mla.paths import ROOT
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_int8_cases import SOURCE_CASES
from tests.mla.support.mla_int8_tiling import Params, parse_tiling
from tests.mla.support.mla_int8_native import native_vendor
from tests.mla.support.mla_validation_build import files_identity


@pytest.fixture(scope="module")
def driver():
    root = Path(os.environ["MLA_VALIDATION_RESULT"]) / "schedule"
    root.mkdir()
    binary = root / "driver"
    source = ROOT / "tests/mla/support/mla_int8_tnd_schedule_driver.cpp"
    mla = ROOT / "csrc/ascend910/flash_attn_npu_3/mla"
    command = [
        "g++",
        "-std=c++17",
        "-O1",
        "-g",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-fno-sanitize-recover=all",
        str(source),
        f"-I{mla.parent}",
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, capture_output=True)
    (root / "build.json").write_text(
        json.dumps(
            dict(
                command=command,
                inputs=files_identity(
                    [source, mla / "mla_int8_schedule.h", mla / "mla_int8_params.h"]
                ),
                outputs=files_identity([binary]),
            ),
            indent=2,
        )
        + "\n"
    )
    with CpuDriver(binary) as process:
        yield process


def encode(
    batch=1,
    query=1,
    heads=8,
    capacity=128,
    cores=20,
    splits=0,
    causal=0,
    reserved=0,
    lengths=(127,),
    prefix=(0, 1),
    total=1,
):
    return [
        batch,
        query,
        heads,
        capacity,
        cores,
        splits,
        int(causal),
        reserved,
        total,
        len(lengths),
        *lengths,
        len(prefix),
        *prefix,
    ]


def test_fixed_native_geometry(driver):
    fixtures = parse_tiling(
        os.environ["MLA_INT8_SOURCE_LOG"], len(SOURCE_CASES), native_vendor(), SOURCE_CASES
    )["cases"]
    for spec, fixture in zip(SOURCE_CASES, fixtures):
        if spec[1] != "TND":
            continue
        expected = Params.from_buffer_copy(
            bytes((ct.c_uint32 * len(fixture["words"]))(*fixture["words"]))
        )
        row = encode(
            len(spec[2]),
            max(spec[2]),
            spec[3],
            expected.baseParams.seqSize,
            causal=bool(spec[5]),
            lengths=spec[4],
            prefix=(0, *itertools.accumulate(spec[2])),
            total=sum(spec[2]),
        )
        actual = driver.evaluate([row])[0]
        expected.baseParams.scaleValue = 0  # invocation value, excluded from metadata
        assert actual[:2] == [1, fixture["mode"]], spec[0]
        assert actual[2:] == list(
            (ct.c_uint32 * (ct.sizeof(Params) // 4)).from_buffer_copy(expected)
        ), spec[0]


@pytest.mark.parametrize(
    "change",
    [
        {"query": 33},
        {"query": 2},
        {"heads": 0},
        {"heads": 3},
        {"heads": 256},
        {"capacity": 129},
        {"capacity": 2**31},
        {"cores": 0},
        {"cores": 26},
        {"splits": 2},
        {"causal": 2},
        {"reserved": 1},
        {"lengths": (-1,)},
        {"lengths": (129,)},
        {"batch": 2},
        {"batch": 2**32 - 1, "lengths": ()},
        {"prefix": (1, 1)},
        {"prefix": (0, -1)},
        {"prefix": (0, 2)},
        {"prefix": (0, 0)},
        {"prefix": (0,)},
        {"total": 1048577},
        {"batch": 2, "lengths": (127, 127), "prefix": (0, 1, 0), "total": 0},
    ],
)
def test_invalid_bounds(driver, change):
    assert driver.evaluate([encode(**change)])[0][0] == 0


@pytest.mark.parametrize("total", [1048576, 1048577])
def test_total_query_allocation_bound(driver, total):
    # Keep each segment valid so the upper-bound rejection cannot be caused
    # by a bad prefix tail, oversized segment or a different shape bound.
    batch = (total + 31) // 32
    result = driver.evaluate(
        [
            encode(
                batch=batch,
                query=32,
                causal=True,
                lengths=(0,) * batch,
                prefix=tuple(min(i * 32, total) for i in range(batch + 1)),
                total=total,
            )
        ]
    )[0]
    assert result[0] == (total == 1048576)


@pytest.mark.parametrize(
    "query,heads,lengths",
    [
        (0, 128, (0, 129)),
        (1, 1, (0,)),
        (1, 128, (0, 0)),
        (4, 8, (4097, 0)),
        (32, 128, (4097, 1, 0)),
        (4, 8, (4096, 5120)),
        (2, 128, (37000,)),
    ],
)
@pytest.mark.parametrize("splits", [0, 1])
def test_empty_fd_and_maximum_query(driver, query, heads, lengths, splits):
    capacity = max(128, (max(lengths) + 127) // 128 * 128)
    result = driver.evaluate(
        [
            encode(
                len(lengths),
                query,
                heads,
                capacity,
                splits=splits,
                causal=query > 1,
                lengths=lengths,
                prefix=tuple(i * query for i in range(len(lengths) + 1)),
                total=len(lengths) * query,
            )
        ]
    )[0]
    assert result[0] == 1
    if splits == 1:
        assert result[1] & 2 == 0
