"""v5 allocation/used decoder, shared by CPU sanitizers and release device probes."""

import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla_metadata_build import MLA, ROOT, sha256
from tests.test_flash_attn_npu_v3_mla_metadata_consumer import CORRUPTIONS, Memory
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata_consumer import (
    expected_schedule as bsnd_schedule,
)
from tests.test_flash_attn_npu_v3_mla_used_metadata import (
    used_builder as used_builder,
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    PREFIX,
    SENTINEL,
    build,
    check_records,
    metadata_cases,
    input_arrays,
    allocation_words,
    EMPTY_CASES,
    used_case,
    wide_case,
    encode,
)


class Allocation(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("layout", "low", "high", "maximum")]


class Context(ct.Structure):
    _fields_ = (
        [(name, ct.c_uint32) for name in ("dtype", "batch", "heads", "flags")]
        + [
            ("query", Allocation),
            ("kv", Allocation),
        ]
        + [(name, ct.c_uint32) for name in ("aic", "aiv", "block_dim")]
        + [
            ("workspace", ct.c_uint64),
            ("libapi", ct.c_uint64),
            ("memory", Memory),
        ]
    )


class Array(ct.Structure):
    _fields_ = [("address", ct.c_uint64), ("count", ct.c_uint64)]


ARRAY_NAMES = ("cu_q", "cu_k", "used_q", "used_k")


class Inputs(ct.Structure):
    _fields_ = [(name, Array) for name in ARRAY_NAMES]


USED_CORRUPTIONS = CORRUPTIONS + [
    *[(field, version, 2) for field in (1, 2) for version in (1, 3, 4)],
    *[(i, 0xFFFFFFFF, 5) for i in range(401, 410)],
    *[(PREFIX + 8 * b + i, 0xFFFFFFFF, 12 if i < 6 else 13) for b in range(2) for i in range(8)],
    (106, 0, 7),
    (107, 1, 7),
    (366, 1, 7),
    *[(i, 0xFFFFFFFF, 9) for i in range(371, 375)],
]


@pytest.fixture(scope="module")
def used_consumer(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_USED_CONSUMER_CPU_DIR", tmp_path_factory.mktemp("mla-used-consumer"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "tests/mla_used_metadata_consumer_driver.cpp"] + [
        MLA / name
        for name in (
            "mla_used_metadata.cpp",
            "mla_tnd_schedule.cpp",
            "mla_metadata.cpp",
            "mla_schedule.cpp",
            "detail/cann_split_core.cpp",
        )
    ]
    common = ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{MLA}", *map(str, sources)]
    library = directory / "consumer.so"
    commands = [[*common, "-O2", "-shared", "-fPIC", "-o", str(library)]]
    if os.environ.get("MLA_CONSUMER_SANITIZE") == "1":
        binary = directory / "consumer-sanitize"
        commands += [
            [
                *common,
                "-O1",
                "-g",
                "-DMLA_CONSUMER_SANITIZE",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                "-fno-omit-frame-pointer",
                "-o",
                str(binary),
            ],
            [str(binary)],
        ]
    (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with (directory / "build.log").open("w") as log:
        for command in commands:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sources += list(MLA.glob("*.h")) + [MLA / "detail/cann_split_core.h", Path(__file__)]
    (directory / "build-manifest.json").write_text(
        json.dumps(
            dict(
                library=str(library.resolve()),
                library_sha256=sha256(library),
                sources={str(p.relative_to(ROOT)): sha256(p) for p in sources},
                sanitizer=os.environ.get("MLA_CONSUMER_SANITIZE") == "1",
            ),
            indent=2,
        )
        + "\n"
    )
    lib = ct.CDLL(str(library.resolve()))
    lib.consume_used_metadata.argtypes = [
        ct.c_void_p,
        ct.c_uint32,
        ct.POINTER(Context),
        ct.POINTER(Inputs),
        ct.c_void_p,
        ct.POINTER(ct.c_uint32),
    ]
    lib.consume_used_metadata.restype = ct.c_uint32
    return lib, directory


def context_for(row, item, dtype):
    values = encode(item)
    alloc = allocation_words(item)
    return Context(
        dtype,
        len(item["q"]),
        values[2],
        alloc[8],
        Allocation(*alloc[:4]),
        Allocation(*alloc[4:8]),
        values[4],
        values[5],
        row[21],
        row[385] | row[386] << 32,
        values[11] if row[15] else 0,
        Memory(*values[6:11]),
    )


def expected_schedule(row):
    return bsnd_schedule(row) + row[401:410]


def decode(consumer, row, context, item, *, words=None, address=None, descriptors=None):
    storage = (ct.c_uint32 * (len(row) + 2))(SENTINEL, *row, SENTINEL)
    output = (ct.c_uint32 * 100)(*([SENTINEL] * 100))
    reads = (ct.c_uint32 * 5)()
    inputs, keep = Inputs(), []
    for name, (values, present, count) in zip(ARRAY_NAMES, input_arrays(item)):
        array = (ct.c_uint32 * (len(values) + 2))(SENTINEL, *values, SENTINEL)
        setattr(inputs, name, Array(ct.addressof(array) + 4 if present else 0, count))
        keep.append((array, list(array)))
    for name, changes in (descriptors or {}).items():
        for field, value in changes.items():
            setattr(getattr(inputs, name), field, value)
    status = consumer[0].consume_used_metadata(
        ct.addressof(storage) + 4 if address is None else address,
        len(row) if words is None else words,
        ct.byref(context),
        ct.byref(inputs),
        ct.addressof(output) + 4,
        reads,
    )
    assert list(storage) == [SENTINEL, *row, SENTINEL]
    assert all(list(array) == before for array, before in keep)
    assert output[0] == output[-1] == SENTINEL
    if status:
        assert list(output) == [SENTINEL] * 100
    return status, list(output)[1:-1], list(reads)


@pytest.mark.parametrize("dtype,layout,aic", itertools.product((1, 2), range(3), (1, 15, 20, 26)))
def test_source_records(used_consumer, used_builder, used_oracle, dtype, layout, aic):
    items = metadata_cases(layout, aic, 2)
    rows = build(used_builder, items, dtype)
    for item, row in zip(items, rows):
        status, schedule, _ = decode(used_consumer, row, context_for(row, item, dtype), item)
        assert status == 0, (item, status)
        assert schedule == expected_schedule(row)
    check_records(
        rows, items, used_oracle, used_consumer[1], f"source-{dtype}-{layout}-{aic}", dtype
    )


@pytest.mark.parametrize("word,value,expected", USED_CORRUPTIONS)
def test_corrupt_record(used_consumer, used_builder, word, value, expected):
    item = used_case(1, [1, 1], [257, 257], [1, 1], [257, 257], heads=8)
    row = build(used_builder, [item])[0]
    c = context_for(row, item, 1)
    assert row[word] != value
    row[word] = value
    status, _, reads = decode(used_consumer, row, c, item)
    assert status == expected
    if word < 8:
        assert reads[0] <= 8 and reads[1:] == [0] * 4


@pytest.mark.parametrize("dtype,layout", itertools.product((1, 2), range(3)))
def test_every_current_prefix_and_used(used_consumer, used_builder, dtype, layout):
    item = used_case(layout, [7] * 42, [513] * 42, [3, 0, 1] * 14, [0, 513, 17] * 14)
    row = build(used_builder, [item], dtype)[0]
    c = context_for(row, item, dtype)
    for axis, (values, present, _) in enumerate(input_arrays(item)):
        if not present:
            continue
        for b in range(len(values)):
            for bad in (-1, -(2**31), 2**31 - 1, values[b] + 1):
                altered = values.copy()
                altered[b] = bad
                current = dict(item, **{("cq", "ck", "uq", "uk")[axis]: altered})
                assert decode(used_consumer, row, c, current)[0] == (12 if axis < 2 else 13)
    # Valid changed lengths must be rejected until the same buffer is regenerated.
    changed = dict(item, uq=[0] * 42, uk=[0] * 42)
    updated = build(used_builder, [changed], dtype)[0]
    assert decode(used_consumer, updated, context_for(updated, changed, dtype), changed)[0] == 0
    assert updated[PREFIX : PREFIX + 6] == row[PREFIX : PREFIX + 6]
    assert updated[15] and updated[385]


@pytest.mark.parametrize("dtype", (1, 2))
def test_u01_t01_empty_and_presence(used_consumer, used_builder, dtype):
    items = (
        EMPTY_CASES
        + [
            used_case(uq=[3, 1], uk=[17, 0]),
            used_case(1, [3, 0, 1], [17, 0, 33], [2, 0, 1], [7, 0, 17]),
            used_case(2, [3, 0, 1], [64] * 3, [2, 0, 1], [17, 0, 33]),
        ]
        + [
            used_case(layout, [2] * 10, [511] * 10, uq, uk)
            for layout, uq, uk in itertools.product(
                range(3), (None, [1] * 10, [0] * 10), (None, [511] * 10, [0] * 10)
            )
        ]
    )
    rows = build(used_builder, items, dtype)
    for item, row in zip(items, rows):
        c = context_for(row, item, dtype)
        status, schedule, _ = decode(used_consumer, row, c, item)
        assert status == 0 and schedule == expected_schedule(row)
        changed = dict(item, uq=None if item["uq"] is not None else item["q"])
        assert decode(used_consumer, row, c, changed)[0] == 13
        if item["q"]:
            changed = dict(item, uq=[-1] * len(item["q"]))
            assert decode(used_consumer, row, c, changed)[0] == 13
        elif item["layout"]:
            assert decode(used_consumer, row, c, dict(item, cq=[1]))[0] == 12


def test_descriptors_headers_and_context(used_consumer, used_builder):
    item = used_case(1, [3, 0, 1], [17, 0, 33], [2, 0, 1], [7, 0, 17])
    row = build(used_builder, [item])[0]
    c = context_for(row, item, 1)
    for name in ARRAY_NAMES:
        for fields in (
            {"address": 0},
            {"address": 1},
            {"address": 2**64 - 4},
            {"count": 0},
            {"count": 2**64 - 1},
            {"address": 0, "count": 0},
        ):
            status, _, reads = decode(used_consumer, row, c, item, descriptors={name: fields})
            assert status == (12 if name.startswith("cu_") else 13)
            assert reads[1:] == [0] * 4
    for kwargs in [dict(words=w) for w in (0, 8, len(row) - 1, len(row) + 1, 2**32 - 1)] + [
        dict(address=a) for a in (0, 1, 2**64 - 4)
    ]:
        assert decode(used_consumer, row, c, item, **kwargs)[::2] == (1, [0] * 5)
    for field, value, expected in (
        ("batch", 2**32 - 1, 1),
        ("dtype", 3, 4),
        ("heads", 0, 5),
        ("heads", 2**32 - 1, 5),
        ("flags", 4, 5),
        ("aic", 0, 6),
        ("aiv", 20, 6),
        ("block_dim", 0, 6),
        ("workspace", 0, 9),
        ("libapi", 2**64 - 1, 9),
    ):
        changed = Context.from_buffer_copy(c)
        setattr(changed, field, value)
        assert decode(used_consumer, row, changed, item)[0] == expected
    for axis in ("query", "kv"):
        for field in ("layout", "low", "high", "maximum"):
            changed = Context.from_buffer_copy(c)
            setattr(getattr(changed, axis), field, 2**32 - 1)
            assert decode(used_consumer, row, changed, item)[0] == 5
    for field in ("l1", "ub", "l0a", "l0b", "l0c"):
        changed = Context.from_buffer_copy(c)
        setattr(changed.memory, field, 0)
        assert decode(used_consumer, row, changed, item)[0] == 10


def test_endpoint_holes_and_variable_m(used_consumer, used_builder):
    item = used_case(
        1,
        [7, 0, 3, 7, 0],
        [513, 7, 33, 17, 16384],
        [3, 0, 1, 0, 0],
        [513, 7, 1, 0, 16384],
        heads=171,
    )
    good = build(used_builder, [item])[0]
    c = context_for(good, item, 1)
    for b, m in ((1, 0), (2, 1), (5, 1), (0, 0)):
        row = good.copy()
        row[20] = 1
        row[23:101] = [0] * 78
        row[23], row[49] = b, m
        assert decode(used_consumer, row, c, item)[0] == 8


@pytest.mark.parametrize("axis", ("query", "kv"))
def test_wide_offsets(used_consumer, used_builder, axis):
    item = wide_case(0 if axis == "query" else 2)
    row = build(used_builder, [item])[0]
    c = context_for(row, item, 1)
    status, schedule, _ = decode(used_consumer, row, c, item)
    assert status == 0 and schedule == expected_schedule(row)
    offset = -7 if axis == "query" else -5
    assert row[offset] > 0
    row[offset] = 0
    assert decode(used_consumer, row, c, item)[0] == 12


def test_allocation_bounds_and_used_tile_overflow(used_consumer, used_builder):
    # Tight TND totals fit while B*maxLength is deliberately much larger.
    # Metadata-only fixtures allocate no attention tensors at these limits.
    items = [
        used_case(1, [2**30, 2**30 - 1], [16384, 1], [0, 0], [0, 0], heads=1),
        used_case(1, [3, 0, 1], [17, 0, 33], [2, 0, 1], [7, 0, 17], sq=2**30, sk=16384, heads=1),
        used_case(0, [65536] * 9, [16384] * 9, [0] * 9, [0] * 9, heads=65535),
    ]
    rows = build(used_builder, items)
    for item, row in zip(items, rows):
        assert row[5:8] == [0, 0, 0]
        c = context_for(row, item, 1)
        assert decode(used_consumer, row, c, item)[0] == 0
    item, row = items[-1], rows[-1]
    c = context_for(row, item, 1)
    changed = dict(item, uq=item["q"], uk=item["kv"])
    for b in range(len(item["q"])):
        row[PREFIX + 8 * b + 6 : PREFIX + 8 * b + 8] = [item["q"][b], item["kv"][b]]
    assert decode(used_consumer, row, c, changed)[0] == 8
