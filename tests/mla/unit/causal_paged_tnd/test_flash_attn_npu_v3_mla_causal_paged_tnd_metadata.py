"""v11 TND PA source geometry, causal whole-M partitions and release consumer bounds."""

import itertools
import json
import os
import subprocess

import pytest

from tests.mla.support.mla_causal_build import CpuDriver
from tests.mla.support.components.schedule import MLA, ROOT
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_schedule import (
    causal_artifacts as causal_artifacts,
    drivers as drivers,
    evaluate,
    schedule_case,
    check_schedule,
)
from tests.mla.support.components.paged_schedule import (
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    paged_case,
    execute,
)
from tests.mla.support.components.paged_metadata import encode


@pytest.fixture(scope="module")
def causal_paged_tnd_driver(causal_artifacts):
    directory = causal_artifacts / "paged-tnd-metadata"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aicpu_api.h").write_text(
        "#define __global__\n#define __aicpu__\nnamespace AscendC { inline void DataStoreBarrier() {} }\n"
    )
    binary = directory / "v11"
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-g",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-D_GLIBCXX_ASSERTIONS",
        "-DMLA_TEST_TND",
        f"-I{directory}",
        f"-I{MLA}",
        "-x",
        "c++",
        str(MLA / "mla_paged_metadata.aicpu"),
        str(ROOT / "tests/mla/support/mla_causal_paged_metadata_driver.cpp"),
        "-o",
        str(binary),
    ]
    if os.environ.get("MLA_SCHEDULE_SANITIZE") == "1":
        command += [
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer",
        ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    with CpuDriver(binary) as driver:
        yield driver


@pytest.mark.parametrize("dtype,cores,used", itertools.product((1, 2), (1, 20, 26), (False, True)))
def test_all_pages_source_partition_and_consumer(
    causal_paged_tnd_driver, drivers, paged_oracles, dtype, cores, used
):
    items = [
        paged_case(
            2,
            [q, 0, max(q - 2, 0)],
            [k, 0, min(k, 3)],
            [max(q - 1, 0), 0, max(q - 2, 0)] if used else None,
            page=p,
            heads=h,
            aic=cores,
            ratio=2,
        )
        for p in range(16, 1025, 16)
        for q, k, h in ((7, p + 1, 8), (257, 129, 3), (3, 3, 129))
    ]
    rows = evaluate(causal_paged_tnd_driver, [[dtype, *encode(i)] for i in items])
    tiles = execute(paged_oracles[0], items)
    schedules = [
        schedule_case(
            i["q"] if i["uq"] is None else i["uq"], i["uk"], i["heads"], *w[15:17], cores, 2
        )
        for i, w in zip(items, rows)
    ]
    adapted, native = [evaluate(driver, schedules) for driver in drivers[:2]]
    for i, w, tile, case, got, source in zip(items, rows, tiles, schedules, adapted, native):
        assert w[:6] == [0x4D4C4131, 11, 11, 423, dtype, 0]
        assert tile[0] == 0 and w[15:17] == tile[1:3]
        check_schedule(case, got, source)
        assert w[20] == got[1] and w[23:101] == got[6:84]
        assert w[101] + (w[102] << 32) == got[2]
        assert w[401:408] == [1, sum(i["q"]), 0, max(i["q"]), int(used), i["page"], i["sk"]]
        assert w[12] == (max(i["uk"]) + i["page"] - 1) // i["page"] * i["page"]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize(
    "q,k,uq,capacity",
    [
        ([], [], [], 0),
        ([0, 0], [17, 0], None, 224),
        ([7, 7], [0, 0], None, 0),
        ([7, 7], [0, 0], [5, 0], 224),
        ([3, 3], [17, 0], [0, 0], 224),
    ],
)
def test_empty_and_zero_used(causal_paged_tnd_driver, dtype, q, k, uq, capacity):
    item = paged_case(2, q, k, uq, page=112, capacity=capacity, heads=8)
    row = evaluate(causal_paged_tnd_driver, [[dtype, *encode(item)]])[0]
    assert row[5] == 0


@pytest.mark.parametrize(
    "change",
    [
        dict(page=17),
        dict(page=0),
        dict(sk=16385),
        dict(heads=0),
        dict(aic=0),
        dict(layout=0),
        dict(uq=[8, 1]),
        dict(uk=[-1, 1]),
        dict(uk=[225, 0]),
        dict(cq=[0, 8, 14]),
        dict(cq=[1, 7, 14]),
        dict(cq=[0, 7, 6]),
        dict(cq=[0, 7, -1]),
        dict(cq=[0, 7, 2147483647]),
        dict(has_q=False),
        dict(has_uk=False),
        dict(ukcount=1),
        dict(uqcount=1),
        dict(qcount=1),
    ],
)
def test_failure_records(causal_paged_tnd_driver, change):
    item = dict(paged_case(2, [7, 7], [17, 0], [5, 1], page=112, capacity=224, heads=8), **change)
    row = evaluate(causal_paged_tnd_driver, [[1, *encode(item)]])[0]
    assert row[5] != 0 and not any(row[8:])


def test_total_derived_from_prefix(causal_paged_tnd_driver):
    item = paged_case(2, [7, 0, 3], [17, 1, 0], [5, 0, 1], page=112, heads=8)
    rows = evaluate(
        causal_paged_tnd_driver,
        [[1, *encode(dict(item, total_q=total))] for total in (0, -1, 1 << 40)],
    )
    assert rows[0] == rows[1] == rows[2]
    assert rows[0][402:404] == [10, 0]
