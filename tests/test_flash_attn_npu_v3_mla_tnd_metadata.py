"""v3 CPU wire checks and pinned-source differential, including full sanitizer runs."""

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla_metadata_build import MLA, ROOT, sha256
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata import projection
from tests.test_flash_attn_npu_v3_mla_tnd_schedule import (
    tnd_artifacts as tnd_artifacts,
    tnd_oracle as tnd_oracle,
    check_tasks,
    evaluate,
    expected_spans,
    prefixes,
    tnd_case,
)

PREFIX = 405
MAGIC = 0x4D4C4131
SENTINEL = 0x12345678


@pytest.fixture(scope="module")
def tnd_builder(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_TND_METADATA_CPU_DIR", tmp_path_factory.mktemp("mla-tnd-wire"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aicpu_api.h").write_text(
        "#define __global__\n#define __aicpu__\nnamespace AscendC { inline void DataStoreBarrier() {} }\n"
    )
    binary = directory / "metadata"
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
        str(MLA / "mla_tnd_metadata.aicpu"),
        str(ROOT / "tests/mla_tnd_metadata_driver.cpp"),
        "-o",
        str(binary),
    ]
    if os.environ.get("MLA_SCHEDULE_SANITIZE") == "1":
        command += [
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer",
        ]
    (directory / "compile.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    sources = [
        *MLA.glob("*.h"),
        *MLA.glob("*.cpp"),
        *MLA.glob("detail/*"),
        MLA / "mla_tnd_metadata.aicpu",
        ROOT / "tests/mla_tnd_metadata_driver.cpp",
        Path(__file__),
    ]
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                binary=str(binary.resolve()),
                binary_sha256=sha256(binary),
                sources={str(p.relative_to(ROOT)): sha256(p) for p in sources},
                sanitized=os.environ.get("MLA_SCHEDULE_SANITIZE") == "1",
            ),
            indent=2,
        )
        + "\n"
    )
    return binary, directory


def build(builder, cases, dtype=1):
    result = subprocess.run(
        [str(builder[0]), str(dtype)],
        input="".join(" ".join(map(str, c)) + "\n" for c in cases),
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert not result.stderr, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == len(cases)
    return rows


def metadata_cases(aic, ratio):
    result = []
    for b, heads, (qs, ks) in itertools.product(
        sorted({1, 2, aic, aic + 1, 2 * aic + 1}),
        (1, 128, 171),
        [
            ([3, 0, 1], [17, 0, 33]),
            ([0, 1, 3], [513, 0, 8193]),
            ([257, 1, 0], [16384, 512, 0]),
            ([2], [511]),
            ([1, 3, 7], [1, 513, 0]),
            ([2, 0], [0, 513]),
        ],
    ):
        q = [qs[i % len(qs)] for i in range(b)]
        k = [ks[i % len(ks)] for i in range(b)]
        if sum(q) and sum(k):
            result.append(tnd_case(q, k, heads=heads, sq=512, sk=16384, aic=aic, ratio=ratio))
    return result


def check_records(rows, cases, oracle, directory, name, dtype):
    directory = Path(directory)
    mapped, native = evaluate(oracle, cases), evaluate(oracle, cases, native=True)
    (directory / f"{name}-inputs.json").write_text(json.dumps(cases) + "\n")
    (directory / f"{name}-outputs.json").write_text(
        json.dumps(
            dict(
                records=rows,
                mapped=mapped,
                native=native,
            )
        )
        + "\n"
    )
    counts = dict(cases=len(cases), native_non_fd=0, fa3_no_split_mapping=0, omitted_suffix=0)
    for row, item, want, source in zip(rows, cases, mapped, native):
        cq, ck = prefixes(item)
        spans = expected_spans(item)
        assert row[:8] == [MAGIC, 3, 3, PREFIX + 4 * item[0], dtype, 0, 0, 0]
        assert len(row) == row[3]
        assert row[401:PREFIX] == [cq[-1], ck[-1], item[1], item[3]]
        assert row[PREFIX:] == spans[2:]
        assert row[8:15] == [item[0], 1, item[2], *spans[:2], 512, 64]
        assert row[17:20] == [item[4], item[5], item[5] // item[4]]

        def wide(words):
            return [lo | hi << 32 for lo, hi in zip(words[::2], words[1::2])]

        workspace = wide(row[367:391])
        cursor = 0
        for offset, size in zip(
            workspace[4:8], [workspace[0] * 4, workspace[0] * 2, workspace[1] * 4, workspace[1] * 4]
        ):
            assert offset == cursor and offset % 32 == 0
            cursor += 2 * row[21] * size
        assert cursor == workspace[8] <= workspace[9]
        assert wide(row[391:401]) == [
            516096,
            174080 if row[19] == 2 else 178176,
            65536,
            65536,
            131072,
        ]
        got = projection(row) + spans
        assert got == want, (item, got, want)
        if source[5] == 0:
            assert got == source
            counts["native_non_fd"] += 1
        else:
            counts["fa3_no_split_mapping"] += 1
        check_tasks(item, got)
        counts["omitted_suffix"] += got[18 + got[3] - 1] < item[0]
    counts["records_sha256"] = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    (directory / f"{name}.json").write_text(json.dumps(counts, indent=2) + "\n")


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("aic,ratio", itertools.product((1, 15, 20, 26), (1, 2)))
def test_source_records(tnd_builder, tnd_oracle, dtype, aic, ratio):
    cases = metadata_cases(aic, ratio)
    check_records(
        build(tnd_builder, cases, dtype),
        cases,
        tnd_oracle,
        tnd_builder[1],
        f"cpu-{dtype}-{aic}-{ratio}",
        dtype,
    )


def test_buffer_guards_and_reuse(tnd_builder):
    subprocess.run([str(tnd_builder[0]), "--check-buffers"], check=True, timeout=120)


@pytest.mark.parametrize("dtype", (1, 2))
def test_t01_tail_equality_and_max_bounds(tnd_builder, tnd_oracle, dtype):
    cases = [tnd_case(), tnd_case([2] * 10, [511] * 10), tnd_case([2] * 10, [511] + [0] * 9)]
    rows = build(tnd_builder, cases, dtype)
    check_records(rows, cases, tnd_oracle, tnd_builder[1], f"t01-{dtype}", dtype)
    assert rows[0][PREFIX:] == [0, 0, 3, 17, 3, 17, 0, 0, 3, 17, 1, 33]
    assert [r[15] for r in rows[1:]] == [256, 128]
    bound = tnd_case(sq=512, sk=16384)
    other = build(tnd_builder, [bound], dtype)[0]
    assert other[:401] == rows[0][:401] and other[PREFIX:] == rows[0][PREFIX:]
    assert other[401:PREFIX] == [4, 50, 512, 16384]


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize(
    "qlens,klens", [([], []), ([0], [0]), ([0, 0], [0, 513]), ([3, 0, 1], [0, 0, 0])]
)
def test_global_empty(tnd_builder, dtype, qlens, klens):
    item = tnd_case(qlens, klens, sq=17, sk=16384)
    row = build(tnd_builder, [item], dtype)[0]
    spans = expected_spans(item)
    assert row[:8] == [MAGIC, 3, 3, PREFIX + len(qlens) * 4, dtype, 0, 0, 0]
    assert row[8:15] == [len(qlens), 1, 128, *spans[:2], 512, 64]
    assert projection(row) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] + [0] * 338
    assert row[401:PREFIX] == [sum(qlens), sum(klens), 17, 16384]
    assert row[PREFIX:] == spans[2:]


FAILURES = [
    ({"has_q": False}, 3, 0),
    ({"has_k": False}, 3, 0),
    ({"qcount": 0}, 3, 0),
    ({"kcount": 3}, 3, 0),
    ({"cq": [1, 3, 3, 4]}, 2, 1),
    ({"ck": [-1, 17, 17, 50]}, 2, 1),
    ({"cq": [0, 3, 2, 4]}, 2, 1),
    ({"ck": [0, 17, 16, 50]}, 2, 1),
    ({"cq": [0, 3, -(2**31), 4]}, 2, 1),
    ({"ck": [0, 17, 17, -1]}, 2, 1),
    ({"total_q": 5}, 2, 1),
    ({"total_k": 51}, 2, 1),
    ({"total_q": -1}, 2, 1),
    ({"total_k": -1}, 2, 1),
    ({"sq": 2}, 2, 1),
    ({"sk": 32}, 2, 1),
    ({"heads": 0}, 2, 1),
    ({"sq": -1}, 2, 1),
    ({"sk": 16385}, 2, 1),
    ({"total_q": 2**31}, 2, 2),
    ({"total_k": 2**31}, 2, 2),
    ({"heads": 2**23}, 2, 2),
    ({"sq": 2**30 + 1}, 2, 2),
    ({"sq": 2**30, "heads": 4}, 2, 2),
    ({"sk": 2**32}, 2, 2),
    ({"aic": 0, "aiv": 0}, 2, 3),
    ({"aic": 27}, 2, 3),
    ({"aiv": 21}, 2, 3),
    ({"block_dim": 21}, 2, 3),
    ({"libapi": 2**64 - 1}, 2, 5),
    ({"l1": 1}, 2, 4),
    ({"qlens": [], "klens": [], "has_q": False}, 3, 0),
    ({"qlens": [], "klens": [], "ck": [1]}, 2, 1),
    ({"qlens": [], "klens": [], "total_q": 1}, 2, 1),
    ({"qlens": [0, 0], "klens": [0, 0], "kcount": 2}, 3, 0),
]


@pytest.mark.parametrize("kwargs,status,schedule_status", FAILURES)
def test_failure_zeroes_payload(tnd_builder, kwargs, status, schedule_status):
    row = build(tnd_builder, [tnd_case(**kwargs)])[0]
    assert row[5:8] == [status, schedule_status, 0]
    assert row[8:] == [0] * (len(row) - 8)


def test_dtype_static_and_physical_tile_bounds(tnd_builder):
    for dtype in (0, 3, 2**32 - 1):
        row = build(tnd_builder, [tnd_case(has_q=False)], dtype)[0]
        assert row[4:8] == [dtype, 1, 0, 0] and not any(row[8:])
    large = tnd_case([2**30, 2**30 - 1], [0, 0], heads=1)
    row = build(tnd_builder, [large])[0]
    assert row[5:8] == [0, 0, 0] and row[401] == 2**31 - 1
    assert row[PREFIX:] == expected_spans(large)[2:]
    tiles = tnd_case([2**27] * 15 + [2**27 - 1], [16384] * 16, heads=16)
    row = build(tnd_builder, [tiles])[0]
    assert row[5:8] == [2, 2, 0] and not any(row[8:])
