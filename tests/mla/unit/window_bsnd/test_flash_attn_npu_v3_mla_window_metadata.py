"""v12 AICPU serialization and consumer, pinned window schedule and sanitizer."""

import itertools
import json
import os
import subprocess

import pytest
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.components.schedule import MLA, ROOT
from tests.mla.unit.window_bsnd.test_flash_attn_npu_v3_mla_window_schedule import (
    window_artifacts as window_artifacts,
    drivers as drivers,
    source_case,
    check_schedule,
)
from tests.mla.support.components.window_schedule import normalized
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_schedule import (
    evaluate,
    schedule_case,
)


@pytest.fixture(scope="module")
def metadata_driver(window_artifacts):
    directory = window_artifacts / "metadata"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aicpu_api.h").write_text(
        "#define __global__\n#define __aicpu__\nnamespace AscendC { inline void DataStoreBarrier() {} }\n"
    )
    binary = directory / "v12"
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
        str(MLA / "mla_window_metadata.aicpu"),
        str(ROOT / "tests/mla/support/mla_window_metadata_driver.cpp"),
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
    (0, 7, 3, 33, [], []),
    (2, 0, 3, 33, [0, 0], [0, 33]),
    (2, 7, 3, 0, [7, 3], [0, 0]),
    (2, 7, 3, 33, [7, 3], [3, 0]),
    (2, 513, 3, 257, [513, 257], [257, 129]),
    (2, 129, 73, 1025, [7, 129], [1025, 513]),
    (3, 1, 128, 16384, [1, 1, 0], [16384, 513, 0]),
]


@pytest.mark.parametrize("dtype,cores", itertools.product((1, 2), (1, 20, 26)))
@pytest.mark.parametrize("q_present,k_present", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_wire_source_and_consumer(metadata_driver, drivers, dtype, cores, q_present, k_present):
    for b, sq, h, sk, uq, uk in CASES:
        for left, right, causal in [
            (2, 1, 0),
            (513, -1, 0),
            (-1, 2, 0),
            (0, 0, 1),
            (-1, -1, 0),
            (-1, -1, 1),
            (2**63 - 1, 2**63 - 1, 0),
        ]:
            request = [
                b,
                sq,
                h,
                sk,
                dtype,
                q_present,
                k_present,
                cores,
                left,
                right,
                causal,
                *itertools.chain.from_iterable(zip(uq, uk)),
            ]
            words = evaluate(metadata_driver, [request])[0]
            assert words[:6] == [0x4D4C4131, 12, 12, 413 + 8 * b, dtype, 0]
            config = [sq, sk, left, right, causal]
            assert words[-3:] == [v % 2**32 for v in normalized(config)]
            assert words[409] == q_present + 2 * k_present
            q = uq if q_present else [sq] * b
            k = uk if k_present else [sk] * b
            assert words[410:-3] == list(
                itertools.chain.from_iterable(
                    (i * sq, 0, i * sk, 0, sq, sk, x, y) for i, (x, y) in enumerate(zip(q, k))
                )
            )
            if not b or not sq or not sk:
                continue
            item = schedule_case(q, k, h, words[15], words[16], cores, 2)
            got = evaluate(drivers[0], [[1, *config, *item[1:]]])[0]
            native = evaluate(drivers[1], [source_case(item, config)])[0]
            check_schedule(item, config, got, native)
            assert words[20] == got[1] and words[23:101] == got[6:84]
            assert words[101] + (words[102] << 32) == got[2]


@pytest.mark.parametrize(
    "index,value",
    [
        (4, 3),
        (11, -1),
        (11, 8),
        (12, -1),
        (12, 34),
        (3, 16385),
        (2, 0),
        (7, 0),
        (8, -(2**63)),
        (9, -(2**63)),
    ],
)
def test_failure_record(metadata_driver, index, value):
    request = [2, 7, 3, 33, 1, 1, 1, 20, 2, 1, 0, 7, 33, 2, 3]
    request[index] = value
    words = evaluate(metadata_driver, [request])[0]
    assert words[5] != 0 and not any(words[8:])
