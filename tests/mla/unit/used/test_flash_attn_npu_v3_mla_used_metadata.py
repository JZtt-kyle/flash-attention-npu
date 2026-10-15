"""v5 allocation/used wire and buffers, checked against independent pinned CANN source."""

import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess

import pytest

from tests.mla.support.mla_metadata_build import MLA, ROOT, sha256
from tests.mla.support.components.bsnd_metadata import projection
from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata import build as build_encoded
from tests.mla.support.components.used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    encode,
    used_case,
)

from tests.mla.support.components.used_metadata import (
    MAGIC as MAGIC,
    PREFIX as PREFIX,
    SENTINEL as SENTINEL,
    allocation_words as allocation_words,
    check_records as check_records,
    input_arrays as input_arrays,
    wire_spans as wire_spans,
)


@pytest.fixture(scope="module")
def used_builder(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_USED_METADATA_CPU_DIR", tmp_path_factory.mktemp("mla-used-wire"))
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
        str(MLA / "mla_used_metadata.aicpu"),
        str(ROOT / "tests/mla/support/mla_used_metadata_driver.cpp"),
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
        MLA / "mla_used_metadata.aicpu",
        ROOT / "tests/mla/support/mla_used_metadata_driver.cpp",
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


def build(builder, items, dtype=1):
    return build_encoded(builder, [encode(item) for item in items], dtype)


def metadata_cases(layout, aic=20, ratio=2):
    result = []
    for b, heads, (qc, kc), mode in itertools.product(
        sorted({1, 2, aic, aic + 1}), (1, 128, 171), ((7, 33), (2, 511), (257, 16384)), range(4)
    ):
        q = [qc] * b if layout == 0 else [[qc, 0, 1][i % 3] for i in range(b)]
        k = [[kc, 0, 17][i % 3] for i in range(b)] if layout == 1 else [kc] * b
        uq = [min(n, [3, 0, 1][i % 3]) for i, n in enumerate(q)] if mode & 1 else None
        uk = [min(n, [17, 0, kc][i % 3]) for i, n in enumerate(k)] if mode & 2 else None
        result.append(used_case(layout, q, k, uq, uk, heads=heads, aic=aic, ratio=ratio))
    result += [
        used_case(layout, [2] * 10, [511] * 10, uq, uk, aic=aic, ratio=ratio)
        for uq, uk in itertools.product((None, [0] * 10), (None, [511] * 10, [0] * 10))
    ]
    return result


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("layout,aic,ratio", itertools.product(range(3), (1, 15, 20, 26), (1, 2)))
def test_source_records(used_builder, used_oracle, dtype, layout, aic, ratio):
    items = metadata_cases(layout, aic, ratio)
    check_records(
        build(used_builder, items, dtype),
        items,
        used_oracle,
        used_builder[1],
        f"cpu-{dtype}-{layout}-{aic}-{ratio}",
        dtype,
    )


def test_buffers_guards_and_v1_to_v4_isolation(used_builder):
    subprocess.run([str(used_builder[0]), "--check-buffers"], check=True, timeout=120)


@pytest.mark.parametrize("dtype", (1, 2))
def test_u01_t01_geometry_and_presence(used_builder, used_oracle, dtype):
    items = [
        used_case(uq=[3, 1], uk=[17, 0]),
        used_case(1, [3, 0, 1], [17, 0, 33], [2, 0, 1], [7, 0, 17]),
        used_case(2, [3, 0, 1], [64] * 3, [2, 0, 1], [17, 0, 33]),
    ]
    items += [
        used_case(1, [2] * 10, [511] * 10, uq, uk)
        for uq, uk in itertools.product((None, [1] * 10), (None, [511] * 10, [0] * 10))
    ]
    rows = build(used_builder, items, dtype)
    check_records(rows, items, used_oracle, used_builder[1], f"u01-t01-{dtype}", dtype)
    assert rows[0][PREFIX:] == [0, 0, 0, 0, 7, 33, 3, 17, 7, 0, 33, 0, 7, 33, 1, 0]
    assert [r[15] for r in rows[3:]] == [256, 128, 128, 256, 128, 128]
    for item, row in zip(items[1:3], rows[1:3]):
        higher = dict(item, sq=512, **({"sk": 513} if item["layout"] == 1 else {}))
        other = build(used_builder, [higher], dtype)[0]
        assert other[:401] == row[:401] and other[PREFIX:] == row[PREFIX:]
    frozen = rows[0]
    changed = dict(items[0], uq=[0, 0], uk=[0, 0])
    updated = build(used_builder, [changed], dtype)[0]
    for i in range(6):
        assert updated[PREFIX + i :: 8] == frozen[PREFIX + i :: 8]
    assert projection(updated)[4] == 0 and projection(updated)[13] > 0


EMPTY_CASES = [
    used_case(layout, q, k, [0] * len(q) if explicit else None, [0] * len(k) if explicit else None)
    for layout, (q, k), explicit in itertools.product(
        range(3), (([], []), ([0, 0], [33, 33]), ([7, 7], [0, 0])), (False, True)
    )
]


@pytest.mark.parametrize("dtype", (1, 2))
def test_empty_and_invalid_empty(used_builder, dtype):
    rows = build(used_builder, EMPTY_CASES, dtype)
    for item, row in zip(EMPTY_CASES, rows):
        assert row[:8] == [MAGIC, 5, 5, PREFIX + 8 * len(item["q"]), dtype, 0, 0, 0]
        assert row[401:PREFIX] == allocation_words(item) and row[PREFIX:] == wire_spans(item)
        assert projection(row) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] + [0] * 338
        failed = build(used_builder, [dict(item, uqcount=2**64 - 1, has_uq=False)], dtype)[0]
        assert failed[5:8] == [3, 0, 0] and not any(failed[8:])


FAILURES = []
for layout in range(3):
    for changes, status, schedule in [
        ({"uq": [-1, 0]}, 2, 1),
        ({"uk": [0, -(2**31)]}, 2, 1),
        ({"uq": [8, 0]}, 2, 1),
        ({"uk": [0, 34]}, 2, 1),
        ({"uq": [2**31 - 1, 0]}, 2, 1),
        ({"uk": [0, 2**31 - 1]}, 2, 1),
        ({"uq": [1]}, 3, 0),
        ({"uk": [1]}, 3, 0),
        ({"uq": [1, 1], "uqcount": 0}, 3, 0),
        ({"uk": [1, 1], "ukcount": 0}, 3, 0),
        ({"uqcount": 1}, 3, 0),
        ({"ukcount": 2**64 - 1}, 3, 0),
        ({"qcount": 2**64 - 1, "has_q": False}, 3, 0),
        ({"kcount": 2**64 - 1, "has_k": False}, 3, 0),
        ({"sq": -1}, 2, 1),
        ({"sk": -1}, 2, 1),
        ({"heads": 0}, 2, 1),
        ({"total_q": -1}, 2, 1),
        ({"total_k": -1}, 2, 1),
        ({"total_q": 13}, 2, 1),
        ({"total_k": 65}, 2, 1),
        ({"sk": 16385}, 2, 1),
        ({"sq": 2**32}, 2, 2),
        ({"heads": 2**23}, 2, 2),
        ({"sq": 2**30 + 1}, 2, 2),
        ({"sq": 2**30, "heads": 4}, 2, 2),
        ({"total_q": 2**63 - 1}, 2, 2),
        ({"aic": 0, "aiv": 0}, 2, 3),
        ({"aic": 27}, 2, 3),
        ({"ratio": 3}, 2, 3),
        ({"aiv": 21}, 2, 3),
        ({"block_dim": 21}, 2, 3),
        ({"libapi": 2**64 - 1}, 2, 5),
        ({"l1": 1}, 2, 4),
    ]:
        FAILURES.append((used_case(layout, **changes), status, schedule))
for field in ("cq", "ck"):
    for prefix in ([1, 7, 14], [-1, 7, 14], [0, 8, 7], [0, -(2**31), 14], [0, 7, 13]):
        FAILURES.append((used_case(1, [7, 7], [7, 7], [0, 0], [0, 0], **{field: prefix}), 2, 1))
FAILURES += [
    (item, 3, 0)
    for item in (
        used_case(1, has_q=False),
        used_case(1, has_k=False),
        used_case(2, has_q=False),
        used_case(0, has_q=True),
        used_case(0, has_k=True),
        used_case(2, has_k=True),
        used_case(1, [], [], has_q=False),
        used_case(1, [], [], has_k=False),
    )
]
FAILURES += [
    (item, 2, status)
    for item, status in (
        (used_case(1, [3, 0, 1], [17, 0, 33], uq=[0, 1, 0]), 1),
        (used_case(1, [3, 0, 1], [17, 0, 33], uk=[0, 1, 0]), 1),
        (used_case(2, [3, 0, 1], [33] * 3, uq=[0, 0, 2]), 1),
        (used_case(1, total_q=2**31), 2),
        (used_case(1, total_k=2**31), 2),
    )
]


@pytest.mark.parametrize("item,status,schedule", FAILURES)
def test_failure_zeroes_payload(used_builder, item, status, schedule):
    row = build(used_builder, [item])[0]
    assert row[5:8] == [status, schedule, 0] and not any(row[8:])


def test_dtype_static_and_active_tile_bounds(used_builder):
    for dtype in (0, 3, 2**32 - 1):
        row = build(used_builder, [used_case(1, has_q=False)], dtype)[0]
        assert row[4:8] == [dtype, 1, 0, 0] and not any(row[8:])
    for batch in (-1, (2**32 - 1 - PREFIX) // 8 + 1, 2**32 - 1, 2**63 - 1):
        assert build(used_builder, [used_case(batch=batch)]) == [[-1]]
    full = used_case(1, [2**27] * 15 + [2**27 - 1], [16384] * 16, heads=16)
    row = build(used_builder, [full])[0]
    assert row[5:8] == [2, 2, 0] and not any(row[8:])
    zero = dict(full, uq=[0] * 16)
    row = build(used_builder, [zero])[0]
    assert row[5:8] == [0, 0, 0] and row[PREFIX:] == wire_spans(zero)


def wide_case(layout):
    b = 2**18 + 2
    return used_case(
        layout, [65536 if layout == 0 else 0] * b, [16384] * b, [0] * b, [0] * b, heads=1
    )


@pytest.mark.parametrize("layout", (0, 2))
def test_64bit_physical_spans(used_builder, layout):
    item = wide_case(layout)
    row = build(used_builder, [item])[0]
    assert row[5:8] == [0, 0, 0]
    assert row[401:PREFIX] == allocation_words(item) and row[PREFIX:] == wire_spans(item)
    assert row[-5] == 1  # High word of the last physical KV start.
    if layout == 0:
        assert row[-7] > 0
    (used_builder[1] / f"wide-{layout}.json").write_text(
        json.dumps(
            dict(
                batch=len(item["q"]),
                totals=[sum(item["q"]), sum(item["kv"])],
                last_span=row[-8:],
                words=len(row),
                record_sha256=hashlib.sha256(json.dumps(row).encode()).hexdigest(),
            )
        )
        + "\n"
    )


def test_random_updates_and_snapshot_reuse(used_builder, used_oracle):
    rng = random.Random(60652)
    items = []
    for _ in range(120):
        layout, b = rng.randrange(3), rng.randrange(1, 28)
        qc, kc = rng.choice((1, 7, 257)), rng.choice((33, 511, 513, 8193, 16384))
        q = [qc] * b if layout == 0 else [qc] + [rng.randrange(qc + 1) for _ in range(b - 1)]
        k = [kc] + [rng.randrange(kc + 1) for _ in range(b - 1)] if layout == 1 else [kc] * b
        uq = None if rng.randrange(2) else [rng.randrange(n + 1) for n in q]
        uk = None if rng.randrange(2) else [rng.randrange(n + 1) for n in k]
        items.append(used_case(layout, q, k, uq, uk))
    rows = build(used_builder, items + items[::-1])
    assert rows[: len(items)] == rows[len(items) :][::-1]
    check_records(rows[: len(items)], items, used_oracle, used_builder[1], "random", 1)
