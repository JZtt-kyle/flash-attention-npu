"""v8 TND allocation/used identity, pinned-source schedule and release consumer."""

import itertools
import json
import os
import subprocess

import pytest

from tests.mla.support.mla_causal_build import CpuDriver
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_schedule import MLA, ROOT
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_schedule import (
    causal_artifacts as causal_artifacts,
    drivers as drivers,
    evaluate,
    schedule_case,
    check_schedule,
)
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    used_case,
    execute,
)


@pytest.fixture(scope="module")
def tnd_metadata_driver(causal_artifacts):
    directory = causal_artifacts / "tnd-metadata"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aicpu_api.h").write_text(
        "#define __global__\n#define __aicpu__\nnamespace AscendC { inline void DataStoreBarrier() {} }\n"
    )
    binary = directory / "v8"
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
        str(MLA / "mla_causal_metadata.aicpu"),
        str(ROOT / "tests/mla/support/mla_causal_tnd_metadata_driver.cpp"),
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


CASES = [
    used_case(1, q, k, uq, uk, heads=h, **bounds)
    for q, k, uq, uk, h, bounds in (
        ([], [], [], [], 3, dict(sq=7, sk=33)),
        ([0, 0], [33, 0], [0, 0], [0, 0], 3, dict(sq=7)),
        ([7, 3], [0, 0], [5, 3], [0, 0], 3, dict(sk=33)),
        ([7, 0, 3], [33, 0, 17], [5, 0, 3], [3, 0, 0], 73, {}),
        ([513, 0, 257], [257, 33, 129], [513, 0, 129], [257, 0, 129], 3, {}),
        ([257, 1, 7], [257, 513, 0], [257, 0, 7], [257, 513, 0], 1, {}),
        ([3, 7], [33, 513], [1, 2], [17, 257], 16, dict(sq=64, sk=1025)),
        ([1] * 21, [16384] * 21, [1] * 21, [16384] * 21, 8, {}),
        ([3, 2, 7], [33, 0, 65], [0, 0, 0], [0, 0, 65], 129, {}),
    )
]


def request_for(item, dtype=1, cores=20):
    q, k = item["q"], item["kv"]
    uq, uk = item["uq"], item["uk"]
    return [
        len(q),
        item.get("sq", max(q, default=0)),
        item["heads"],
        item.get("sk", max(k, default=0)),
        sum(q),
        sum(k),
        dtype,
        int(uq is not None) + 2 * int(uk is not None),
        cores,
        *itertools.accumulate(q, initial=0),
        *itertools.accumulate(k, initial=0),
        *itertools.chain.from_iterable(zip(q if uq is None else uq, k if uk is None else uk)),
    ]


@pytest.mark.parametrize("dtype,cores", itertools.product((1, 2), (1, 20, 26)))
@pytest.mark.parametrize("q_present,k_present", itertools.product((False, True), repeat=2))
def test_wire_source_and_consumer(
    tnd_metadata_driver, drivers, used_oracle, dtype, cores, q_present, k_present
):
    for spec in CASES:
        item = dict(
            spec,
            uq=spec["uq"] if q_present else None,
            uk=spec["uk"] if k_present else None,
            aic=cores,
            ratio=2,
        )
        q, k = item["q"], item["kv"]
        b = len(q)
        words = evaluate(tnd_metadata_driver, [request_for(item, dtype, cores)])[0]
        assert words[:6] == [0x4D4C4131, 8, 8, 410 + 8 * b, dtype, 0]
        assert words[401:410] == [
            1,
            sum(q),
            0,
            item.get("sq", max(q, default=0)),
            1,
            sum(k),
            0,
            item.get("sk", max(k, default=0)),
            int(q_present) + 2 * int(k_present),
        ]
        qs = item["uq"] if q_present else q
        ks = item["uk"] if k_present else k
        starts_q = list(itertools.accumulate(q, initial=0))
        starts_k = list(itertools.accumulate(k, initial=0))
        assert words[410:] == list(
            itertools.chain.from_iterable(
                (starts_q[i], 0, starts_k[i], 0, q[i], k[i], qs[i], ks[i]) for i in range(b)
            )
        )
        if not b or not sum(q) or not sum(k):
            assert words[20:23] == [1, 1, 1]
            continue
        # Source parser/tiling determines tiles independently of migration.
        tiling = execute(used_oracle, [item])[0]
        assert tiling[0] == 0 and words[15:17] == tiling[1:3]
        case = schedule_case(qs, ks, item["heads"], *words[15:17], cores, 2)
        got, native = [evaluate(driver, [case])[0] for driver in drivers[:2]]
        check_schedule(case, got, native)
        assert words[20] == got[1] and words[23:101] == got[6:84]
        assert words[101] + (words[102] << 32) == got[2]


@pytest.mark.parametrize(
    "index,value",
    [
        (1, 1 << 31),
        (2, 0),
        (3, 16385),
        (4, 1 << 31),
        (5, 1 << 31),
        (6, 3),
        (8, 0),
        (9, 1),
        (10, -1),
        (11, 9),
        (12, 1),
        (13, -1),
        (14, 49),
        (15, -1),
        (15, 8),
        (16, -1),
        (16, 34),
    ],
)
def test_failure_record(tnd_metadata_driver, index, value):
    item = used_case(1, [7, 3], [33, 17], [5, 3], [3, 0], heads=3)
    request = request_for(item)
    request[index] = value
    words = evaluate(tnd_metadata_driver, [request])[0]
    assert words[5] != 0 and not any(words[8:])
