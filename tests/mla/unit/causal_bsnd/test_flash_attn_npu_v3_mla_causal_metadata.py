"""v7 serialization/consumer identity and pinned-source whole-M scheduling."""

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


@pytest.fixture(scope="module")
def metadata_driver(causal_artifacts):
    directory = causal_artifacts / "metadata"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aicpu_api.h").write_text(
        "#define __global__\n#define __aicpu__\nnamespace AscendC { inline void DataStoreBarrier() {} }\n"
    )
    binary = directory / "v7"
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
        str(ROOT / "tests/mla/support/mla_causal_metadata_driver.cpp"),
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
    (1, 257, 1, 257, [257], [257]),
    (2, 7, 73, 65, [7, 5], [65, 3]),
    (2, 129, 128, 513, [0, 129], [513, 0]),
    (21, 1, 8, 16384, [1] * 21, [16384] * 21),
    (10, 3, 86, 8193, [3] * 10, [8193] * 10),
    (10, 3, 86, 8193, [0, 1, 2, 3, 0, 1, 2, 3, 1, 3], [0, 1, 3, 33, 128, 512, 513, 8192, 8193, 1]),
]


@pytest.mark.parametrize("dtype,cores", itertools.product((1, 2), (1, 20, 26)))
@pytest.mark.parametrize("q_present,k_present", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_wire_source_and_consumer(metadata_driver, drivers, dtype, cores, q_present, k_present):
    for b, sq, h, sk, uq, uk in CASES:
        request = [
            b,
            sq,
            h,
            sk,
            dtype,
            q_present,
            k_present,
            cores,
            *itertools.chain.from_iterable(zip(uq, uk)),
        ]
        words = evaluate(metadata_driver, [request])[0]
        assert words[:6] == [0x4D4C4131, 7, 7, 410 + 8 * b, dtype, 0]
        assert words[409] == q_present + 2 * k_present
        lengths_q = uq if q_present else [sq] * b
        lengths_k = uk if k_present else [sk] * b
        spans = list(
            itertools.chain.from_iterable(
                (i * sq, 0, i * sk, 0, sq, sk, q, k)
                for i, (q, k) in enumerate(zip(lengths_q, lengths_k))
            )
        )
        assert words[410:] == spans
        if not b or not sq or not sk:
            assert words[20:23] == [1, 1, 1]
            continue
        item = schedule_case(lengths_q, lengths_k, h, words[15], words[16], cores, 2)
        got = evaluate(drivers[0], [item])[0]
        native = evaluate(drivers[1], [item])[0]
        check_schedule(item, got, native)
        assert words[20] == got[1]
        assert words[23:101] == got[6:84]
        assert words[101] + (words[102] << 32) == got[2]


@pytest.mark.parametrize(
    "fault", ["dtype", "negative-q", "long-q", "negative-k", "long-k", "shape", "heads", "platform"]
)
def test_failure_record(metadata_driver, fault):
    request = [2, 7, 3, 33, 1, 1, 1, 20, 7, 33, 2, 3]
    index, value = {
        "dtype": (4, 3),
        "negative-q": (8, -1),
        "long-q": (8, 8),
        "negative-k": (9, -1),
        "long-k": (9, 34),
        "shape": (3, 16385),
        "heads": (2, 0),
        "platform": (7, 0),
    }[fault]
    request[index] = value
    words = evaluate(metadata_driver, [request])[0]
    assert words[5] != 0 and not any(words[8:])
