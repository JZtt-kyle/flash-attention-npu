"""v4 mixed-layout CPU wire/buffer checks with an independent pinned-source oracle."""

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla_metadata_build import MLA, ROOT, sha256
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata import projection
from tests.test_flash_attn_npu_v3_mla_tnd_bsnd_schedule import (
    mixed_artifacts as mixed_artifacts,
    mixed_oracle as mixed_oracle,
    expected_spans,
    input_lengths,
    mixed_case,
)
from tests.test_flash_attn_npu_v3_mla_tnd_schedule import (
    check_tasks,
    evaluate,
    expected_spans as compact_spans,
    tnd_case,
)

PREFIX = 406
MAGIC = 0x4D4C4131
SENTINEL = 0x12345678


def wide_words(value):
    return [value & 0xFFFFFFFF, value >> 32]


def wire_spans(item):
    cq, q, kv = input_lengths(item)
    return [w for b in range(item[0]) for w in (cq[b], *wide_words(b * item[3]), q[b], kv[b])]


@pytest.fixture(scope="module")
def mixed_builder(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_TND_BSND_METADATA_CPU_DIR", tmp_path_factory.mktemp("mla-mixed-wire"))
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
        str(MLA / "mla_tnd_bsnd_metadata.aicpu"),
        str(ROOT / "tests/mla_tnd_bsnd_metadata_driver.cpp"),
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
        MLA / "mla_tnd_bsnd_metadata.aicpu",
        ROOT / "tests/mla_tnd_bsnd_metadata_driver.cpp",
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
    for b, heads, (qs, ks, capacity) in itertools.product(
        sorted({1, 2, aic, aic + 1, 2 * aic + 1}),
        (1, 128, 171),
        [
            ([3, 0, 1], [17, 0, 33], 64),
            ([0, 1, 3], [513, 0, 8193], 16384),
            ([257, 1, 0], [16384, 512, 0], 16384),
            ([2], [511], 512),
            ([1, 3, 7], [1, 513, 0], 8193),
            ([2, 0], [0, 513], 1024),
            ([3, 0, 1], [0], 513),
        ],
    ):
        q = [qs[i % len(qs)] for i in range(b)]
        k = [ks[i % len(ks)] for i in range(b)]
        if sum(q):
            result.append(
                mixed_case(q, k, heads=heads, sq=512, capacity=capacity, aic=aic, ratio=ratio)
            )
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
        cq, q, kv = input_lengths(item)
        spans = expected_spans(item)
        assert row[:8] == [MAGIC, 4, 4, PREFIX + 5 * item[0], dtype, 0, 0, 0]
        assert len(row) == row[3]
        assert row[401:PREFIX] == [cq[-1], item[1], item[3], *wide_words(item[0] * item[3])]
        assert row[PREFIX:] == wire_spans(item)
        assert row[8:15] == [item[0], 1, item[2], spans[0], item[3], 512, 64]
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
        compact = tnd_case(q, kv, heads=item[2])
        check_tasks(compact, [*got[:356], *compact_spans(compact)])
        counts["omitted_suffix"] += got[18 + got[3] - 1] < item[0]
    counts["records_sha256"] = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    (directory / f"{name}.json").write_text(json.dumps(counts, indent=2) + "\n")


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("aic,ratio", itertools.product((1, 15, 20, 26), (1, 2)))
def test_source_records(mixed_builder, mixed_oracle, dtype, aic, ratio):
    cases = metadata_cases(aic, ratio)
    check_records(
        build(mixed_builder, cases, dtype),
        cases,
        mixed_oracle,
        mixed_builder[1],
        f"cpu-{dtype}-{aic}-{ratio}",
        dtype,
    )


def test_buffer_guards_protocol_isolation_and_reuse(mixed_builder):
    subprocess.run([str(mixed_builder[0]), "--check-buffers"], check=True, timeout=120)


@pytest.mark.parametrize("dtype", (1, 2))
def test_capacity_q_upper_bound_and_noncumulative_kv(mixed_builder, mixed_oracle, dtype):
    cases = [
        mixed_case(),
        mixed_case(klens=[33, 0, 17]),
        mixed_case(capacity=1024),
        mixed_case([2] * 10, [511] * 10, capacity=511),
        mixed_case([2] * 10, [511] + [0] * 9, capacity=511),
    ]
    rows = build(mixed_builder, cases, dtype)
    check_records(rows, cases, mixed_oracle, mixed_builder[1], f"layout-{dtype}", dtype)
    assert rows[0][PREFIX:] == [0, 0, 0, 3, 17, 3, 64, 0, 0, 0, 3, 128, 0, 1, 33]
    assert rows[0][PREFIX + 1 :: 5] == rows[1][PREFIX + 1 :: 5] == [0, 64, 128]
    assert rows[2][PREFIX + 1 :: 5] == [0, 1024, 2048]
    assert [r[15] for r in rows[3:]] == [128, 256]
    other = build(mixed_builder, [mixed_case(sq=512)], dtype)[0]
    assert other[:401] == rows[0][:401] and other[PREFIX:] == rows[0][PREFIX:]
    assert other[401:PREFIX] == [4, 512, 64, 192, 0]


EMPTY_CASES = [
    ([], [], 0),
    ([], [], 16384),
    ([0], [0], 64),
    ([0, 0], [1, 513], 1024),
    ([3, 0, 1], [0, 0, 0], 0),
]


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("q,k,capacity", EMPTY_CASES)
def test_global_empty(mixed_builder, dtype, q, k, capacity):
    item = mixed_case(q, k, sq=17, capacity=capacity, has_k=bool(k))
    row = build(mixed_builder, [item], dtype)[0]
    assert row[:8] == [MAGIC, 4, 4, PREFIX + len(q) * 5, dtype, 0, 0, 0]
    assert row[8:15] == [len(q), 1, 128, max(q, default=0), capacity, 512, 64]
    assert projection(row) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] + [0] * 338
    assert row[401:PREFIX] == [sum(q), 17, capacity, *wide_words(len(q) * capacity)]
    assert row[PREFIX:] == wire_spans(item)


FAILURES = [
    ({"has_q": False}, 3, 0),
    ({"has_k": False}, 3, 0),
    ({"qcount": 0}, 3, 0),
    ({"kcount": 0}, 3, 0),
    ({"kcount": 4}, 3, 0),
    ({"cq": [1, 3, 3, 4]}, 2, 1),
    ({"cq": [-1, 3, 3, 4]}, 2, 1),
    ({"cq": [0, 3, 2, 4]}, 2, 1),
    ({"cq": [0, 3, -(2**31), 4]}, 2, 1),
    ({"klens": [-1, 0, 33]}, 2, 1),
    ({"klens": [17, -(2**31), 33]}, 2, 1),
    ({"klens": [17, 0, 65]}, 2, 1),
    ({"klens": [17, 0, 2**31 - 1]}, 2, 1),
    ({"total_q": 3}, 2, 1),
    ({"total_q": 5}, 2, 1),
    ({"total_q": -1}, 2, 1),
    ({"sq": 2}, 2, 1),
    ({"capacity": 32}, 2, 1),
    ({"heads": 0}, 2, 1),
    ({"sq": -1}, 2, 1),
    ({"capacity": -1}, 2, 1),
    ({"capacity": 16385}, 2, 1),
    ({"total_q": 2**31}, 2, 2),
    ({"heads": 2**23}, 2, 2),
    ({"sq": 2**30 + 1}, 2, 2),
    ({"sq": 2**30, "heads": 4}, 2, 2),
    ({"capacity": 2**32}, 2, 2),
    ({"aic": 0, "aiv": 0}, 2, 3),
    ({"aic": 27}, 2, 3),
    ({"aiv": 21}, 2, 3),
    ({"block_dim": 21}, 2, 3),
    ({"libapi": 2**64 - 1}, 2, 5),
    ({"l1": 1}, 2, 4),
    ({"qlens": [], "klens": [], "has_q": False}, 3, 0),
    ({"qlens": [], "klens": [], "cq": [1]}, 2, 1),
    ({"qlens": [], "klens": [], "total_q": 1}, 2, 1),
    ({"qlens": [], "klens": [], "kcount": 1}, 3, 0),
    ({"qlens": [3, 0, 1], "klens": [0, 0, 0], "capacity": 0, "cq": [0, 3, 2, 4]}, 2, 1),
]


@pytest.mark.parametrize("kwargs,status,schedule_status", FAILURES)
def test_failure_zeroes_payload(mixed_builder, kwargs, status, schedule_status):
    row = build(mixed_builder, [mixed_case(**kwargs)])[0]
    assert row[5:8] == [status, schedule_status, 0]
    assert row[8:] == [0] * (len(row) - 8)


@pytest.mark.parametrize("dtype", (1, 2))
def test_64bit_capacity_and_physical_starts(mixed_builder, dtype):
    b = 2**18 + 2
    item = mixed_case([0] * b, [0] * b, capacity=16384)
    row = build(mixed_builder, [item], dtype)[0]
    assert row[:8] == [MAGIC, 4, 4, PREFIX + 5 * b, dtype, 0, 0, 0]
    assert row[401:PREFIX] == [0, 0, 16384, *wide_words(b * 16384)]
    assert row[PREFIX:] == wire_spans(item)
    assert row[-4:-2] == [16384, 1]
    (mixed_builder[1] / f"wide-{dtype}.json").write_text(
        json.dumps(
            dict(
                batch=b,
                total_kv_capacity=b * 16384,
                last_kv_start=(b - 1) * 16384,
                words=len(row),
                record_sha256=hashlib.sha256(json.dumps(row).encode()).hexdigest(),
            )
        )
        + "\n"
    )


def test_dtype_static_and_physical_tile_bounds(mixed_builder):
    for dtype in (0, 3, 2**32 - 1):
        row = build(mixed_builder, [mixed_case(has_q=False)], dtype)[0]
        assert row[4:8] == [dtype, 1, 0, 0] and not any(row[8:])
    for batch in (-1, (2**32 - 1 - PREFIX) // 5 + 1, 2**32 - 1, 2**63 - 1):
        assert build(mixed_builder, [mixed_case(batch=batch)]) == [[-1]]
    large = mixed_case([2**30, 2**30 - 1], [0, 0], capacity=0, heads=1)
    row = build(mixed_builder, [large])[0]
    assert row[5:8] == [0, 0, 0] and row[401] == 2**31 - 1
    assert row[PREFIX:] == wire_spans(large)
    tiles = mixed_case([2**27] * 15 + [2**27 - 1], [16384] * 16, capacity=16384, heads=16)
    row = build(mixed_builder, [tiles])[0]
    assert row[5:8] == [2, 2, 0] and not any(row[8:])
