"""v16 TND PA source geometry, band whole-M partitions and release consumer bounds."""

import itertools
import json
import os
import subprocess

import pytest

from tests.mla.support.mla_causal_build import CpuDriver
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_schedule import MLA, ROOT
from tests.mla.unit.window_bsnd.test_flash_attn_npu_v3_mla_window_schedule import (
    window_artifacts as window_artifacts,
    drivers as drivers,
    normalized,
    source_case,
    check_schedule,
)
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_schedule import (
    evaluate,
    schedule_case,
)
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_schedule import (
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    paged_case,
    execute,
)
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_metadata import encode


@pytest.fixture(scope="module")
def window_paged_tnd_driver(window_artifacts):
    directory = window_artifacts / "paged-tnd-metadata"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aicpu_api.h").write_text(
        "#define __global__\n#define __aicpu__\nnamespace AscendC { inline void DataStoreBarrier() {} }\n"
    )
    binary = directory / "v16"
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
        f"-I{directory}",
        f"-I{MLA}",
        "-x",
        "c++",
        str(MLA / "mla_paged_metadata.aicpu"),
        str(ROOT / "tests/mla/support/mla_window_paged_tnd_metadata_driver.cpp"),
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
    window_paged_tnd_driver, drivers, paged_oracles, dtype, cores, used
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
        for p in (
            range(16, 1025, 16)
            if os.environ.get("MLA_FULL_MATRIX") == "1"
            else (16, 48, 112, 128, 256, 496, 528, 1024)
        )
        for q, k, h in ((7, p + 1, 8), (257, 129, 3), (3, 3, 129))
    ]
    tiles = execute(paged_oracles[0], items)
    for window in [
        (2, 1, False),
        (513, -1, False),
        (-1, 2, False),
        (2, 7, True),
        (2**63 - 1, 2**63 - 1, False),
    ]:
        rows = evaluate(
            window_paged_tnd_driver, [[dtype, *map(int, window), *encode(i)] for i in items]
        )
        schedules = [
            schedule_case(
                i["q"] if i["uq"] is None else i["uq"], i["uk"], i["heads"], *w[15:17], cores, 2
            )
            for i, w in zip(items, rows)
        ]
        configs = [[max(i["q"], default=0), i["sk"], *map(int, window)] for i in items]
        adapted = evaluate(drivers[0], [[1, *c, *x[1:]] for c, x in zip(configs, schedules)])
        native = evaluate(drivers[1], [source_case(x, c) for c, x in zip(configs, schedules)])
        for i, w, tile, case, config, got, source in zip(
            items, rows, tiles, schedules, configs, adapted, native
        ):
            assert w[:6] == [0x4D4C4131, 16, 16, 426, dtype, 0]
            assert tile[0] == 0 and w[15:17] == tile[1:3]
            check_schedule(case, config, got, source)
            assert w[20] == got[1] and w[23:101] == got[6:84]
            assert w[101] + (w[102] << 32) == got[2]
            assert w[401:408] == [
                1,
                sum(i["q"]),
                0,
                max(i["q"], default=0),
                int(used),
                i["page"],
                i["sk"],
            ]
            assert w[12] == (max(i["uk"]) + i["page"] - 1) // i["page"] * i["page"]
            assert w[-3:] == [v % 2**32 for v in normalized(config)]


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
def test_empty_and_zero_used(window_paged_tnd_driver, dtype, q, k, uq, capacity):
    item = paged_case(2, q, k, uq, page=112, capacity=capacity, heads=8)
    row = evaluate(window_paged_tnd_driver, [[dtype, 2, 1, 0, *encode(item)]])[0]
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
        dict(total_q=15),
        dict(cq=[0, 7, 6]),
        dict(cq=[1, 7, 14]),
        dict(cq=[0, -1, 14]),
        dict(cq=[0, 7, 15]),
        dict(has_uk=False),
        dict(ukcount=1),
        dict(uqcount=1),
        dict(qcount=1),
    ],
)
def test_failure_records(window_paged_tnd_driver, change):
    item = dict(paged_case(2, [7, 7], [17, 0], [5, 1], page=112, capacity=224, heads=8), **change)
    row = evaluate(window_paged_tnd_driver, [[1, 2, 1, 0, *encode(item)]])[0]
    assert row[5] != 0 and not any(row[8:])


@pytest.mark.parametrize("window", [(-(2**63), 1, False), (2, -(2**63), False)])
def test_invalid_window(window_paged_tnd_driver, window):
    item = paged_case(2, [7, 7], [17, 0], [5, 1], page=112, capacity=224, heads=8)
    row = evaluate(window_paged_tnd_driver, [[1, *map(int, window), *encode(item)]])[0]
    assert row[5] != 0 and not any(row[8:])
