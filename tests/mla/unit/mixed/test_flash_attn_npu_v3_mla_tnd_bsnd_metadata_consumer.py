"""v4 shared decoder: source schedules, physical KV spans and rejection boundaries."""

import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla.support.mla_metadata_build import MLA, ROOT, sha256
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata_consumer import CORRUPTIONS, Memory
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_metadata_consumer import (
    expected_schedule as bsnd_schedule,
)
from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata import (
    mixed_builder as mixed_builder,
    mixed_artifacts as mixed_artifacts,
    mixed_oracle as mixed_oracle,
    PREFIX,
    SENTINEL,
    build,
    check_records,
    metadata_cases,
    input_lengths,
    wire_spans,
    wide_words,
    EMPTY_CASES,
    mixed_case,
)


class Context(ct.Structure):
    _fields_ = [
        (n, ct.c_uint32)
        for n in (
            "dtype",
            "batch",
            "total_q",
            "heads",
            "max_q",
            "capacity",
            "aic",
            "aiv",
            "block_dim",
        )
    ] + [("workspace", ct.c_uint64), ("libapi", ct.c_uint64), ("memory", Memory)]


MIXED_CORRUPTIONS = CORRUPTIONS + [
    *[(field, version, 2) for field in (1, 2) for version in (1, 3)],
    *[(i, 0xFFFFFFFF, 5) for i in range(401, 406)],
    *[
        (PREFIX + 5 * b + i, 0xFFFFFFFF, 12 if i in (0, 3) else 11)
        for b in range(2)
        for i in range(5)
    ],
]


def prefixes(values):
    q, _, k = input_lengths(values)
    return q, k


@pytest.fixture(scope="module")
def mixed_consumer(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_TND_BSND_CONSUMER_CPU_DIR", tmp_path_factory.mktemp("mla-tnd-consumer"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "tests/mla/support/mla_tnd_bsnd_metadata_consumer_driver.cpp"] + [
        MLA / name
        for name in (
            "mla_tnd_bsnd_metadata.cpp",
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
    lib.consume_mixed_metadata.argtypes = [
        ct.c_void_p,
        ct.c_uint32,
        ct.POINTER(Context),
        ct.c_void_p,
        ct.c_uint64,
        ct.c_void_p,
        ct.c_uint64,
        ct.c_void_p,
        ct.POINTER(ct.c_uint32),
    ]
    lib.consume_mixed_metadata.restype = ct.c_uint32
    return lib, directory


def context_for(row, values, dtype):
    cq, _ = prefixes(values)
    return Context(
        dtype,
        values[0],
        cq[-1],
        values[2],
        values[1],
        values[3],
        values[4],
        values[5],
        row[21],
        row[385] | row[386] << 32,
        values[11] if cq[-1] and values[3] else 0,
        Memory(*values[6:11]),
    )


def expected_schedule(row):
    return bsnd_schedule(row) + [row[401], *row[404:406]]


def decode(
    consumer,
    row,
    context,
    q,
    k,
    *,
    words=None,
    address=None,
    qaddress=None,
    kaddress=None,
    qcount=None,
    kcount=None,
):
    storage = (ct.c_uint32 * (len(row) + 2))(SENTINEL, *row, SENTINEL)
    query = (ct.c_int32 * (len(q) + 2))(SENTINEL, *q, SENTINEL)
    kv = (ct.c_int32 * (len(k) + 2))(SENTINEL, *k, SENTINEL)
    output = (ct.c_uint32 * 92)(*([SENTINEL] * 92))
    reads = (ct.c_uint32 * 3)()
    status = consumer[0].consume_mixed_metadata(
        ct.addressof(storage) + 4 if address is None else address,
        len(row) if words is None else words,
        ct.byref(context),
        ct.addressof(query) + 4 if qaddress is None else qaddress,
        len(q) if qcount is None else qcount,
        ct.addressof(kv) + 4 if kaddress is None else kaddress,
        len(k) if kcount is None else kcount,
        output,
        reads,
    )
    assert list(storage) == [SENTINEL, *row, SENTINEL]
    assert list(query) == [SENTINEL, *q, SENTINEL] and list(kv) == [SENTINEL, *k, SENTINEL]
    if status:
        assert list(output) == [SENTINEL] * 92
    return status, list(output), list(reads)


@pytest.mark.parametrize("dtype,aic", itertools.product((1, 2), (1, 15, 20, 26)))
def test_source_records(mixed_consumer, mixed_builder, mixed_oracle, dtype, aic):
    cases = metadata_cases(aic, 2)
    rows = build(mixed_builder, cases, dtype)
    for values, row in zip(cases, rows):
        status, schedule, _ = decode(
            mixed_consumer, row, context_for(row, values, dtype), *prefixes(values)
        )
        assert status == 0, (values, status)
        assert schedule == expected_schedule(row)
    check_records(rows, cases, mixed_oracle, mixed_consumer[1], f"source-{dtype}-{aic}", dtype)


@pytest.mark.parametrize("word,value,expected", MIXED_CORRUPTIONS)
def test_corrupt_record(mixed_consumer, mixed_builder, word, value, expected):
    values = mixed_case([1, 1], [257, 257], heads=8, capacity=257)
    row = build(mixed_builder, [values])[0]
    c = context_for(row, values, 1)
    assert row[word] != value
    row[word] = value
    status, _, reads = decode(mixed_consumer, row, c, *prefixes(values))
    assert status == expected
    if word < 8:
        assert reads[0] <= 8 and reads[1:] == [0, 0]


@pytest.mark.parametrize("dtype", (1, 2))
def test_every_current_prefix_and_span(mixed_consumer, mixed_builder, dtype):
    values = mixed_case([3, 0, 1] * 14, [0, 8193, 513] * 14, sq=512, capacity=16384)
    row = build(mixed_builder, [values], dtype)[0]
    c = context_for(row, values, dtype)
    q, k = prefixes(values)
    for axis in range(2):
        for b in range(len((q, k)[axis])):
            for bad in (-1, -(2**31), 2**31 - 1, (q, k)[axis][b] + 1):
                current = [q.copy(), k.copy()]
                current[axis][b] = bad
                assert decode(mixed_consumer, row, c, *current)[0] == (12 if axis == 0 else 11)
    for i in range(PREFIX, len(row)):
        altered = row.copy()
        altered[i] += 1
        assert decode(mixed_consumer, altered, c, q, k)[0] == (
            12 if (i - PREFIX) % 5 in (0, 3) else 11
        )


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("q,k,capacity", EMPTY_CASES)
def test_empty_still_validates(mixed_consumer, mixed_builder, dtype, q, k, capacity):
    values = mixed_case(q, k, sq=17, capacity=capacity)
    row = build(mixed_builder, [values], dtype)[0]
    c = context_for(row, values, dtype)
    cq, ck = prefixes(values)
    assert decode(mixed_consumer, row, c, cq, ck)[1] == expected_schedule(row)
    for field in (1, 2):
        for version in (1, 2, 3):
            altered = row.copy()
            altered[field] = version
            assert decode(mixed_consumer, altered, c, cq, ck)[0] == 2
    for axis in range(2):
        current = [cq.copy(), ck.copy()]
        if not current[axis]:
            continue
        current[axis][-1] += 1
        assert decode(mixed_consumer, row, c, *current)[0] == (12 if axis == 0 else 11)
    for word in (15, 16, *range(23, 401)):
        altered = row.copy()
        altered[word] = 1
        assert decode(mixed_consumer, altered, c, cq, ck)[0] == 7
    assert decode(mixed_consumer, row, c, cq, ck, kaddress=0)[0] == (11 if q else 0)
    assert decode(mixed_consumer, row, c, cq, ck, qaddress=0)[0] == 12


def test_variable_m_endpoints_and_zero_cost_suffix(mixed_consumer, mixed_builder):
    values = mixed_case([3, 0, 1, 7, 0], [513, 7, 1, 0, 16384], heads=171, capacity=16384)
    row = build(mixed_builder, [values])[0]
    c = context_for(row, values, 1)
    # Hand-crafted partitions validate the addressing contract independently
    # of the producer's choices. Batch 1 has no M blocks; batch 2 has only one.
    for endpoints, expected in [
        ([(1, 0), (2, 0), (3, 0)], 0),
        ([(1, 1), (3, 0)], 8),
        ([(2, 1), (3, 0)], 8),
        ([(1, 0)], 8),
        ([(1, 0), (1, 0), (3, 0)], 8),
        ([(3, 0), (2, 0)], 8),
        ([(3, 1)], 0),  # Remaining partial sample and suffix all have zero cost.
        ([(5, 0)], 0),
        ([(5, 1)], 8),
    ]:
        changed = row.copy()
        changed[20] = len(endpoints)
        for core, (batch, m) in enumerate(endpoints):
            changed[23 + core], changed[49 + core], changed[75 + core] = batch, m, 0
        assert decode(mixed_consumer, changed, c, *prefixes(values))[0] == expected


def test_context_descriptors_and_overflow(mixed_consumer, mixed_builder):
    values = mixed_case()
    row = build(mixed_builder, [values])[0]
    c = context_for(row, values, 1)
    q, k = prefixes(values)
    for options, expected in [
        *[(dict(words=w), 1) for w in (0, 8, 401, len(row) - 1, len(row) + 1, 2**32 - 1)],
        *[(dict(address=a), 1) for a in (0, 2, 2**64 - 4)],
        *[(dict(qaddress=a), 12) for a in (0, 2, 2**64 - 4)],
        *[(dict(kaddress=a), 11) for a in (0, 2, 2**64 - 4)],
        *[(dict(qcount=n), 12) for n in (0, c.batch, 2**64 - 1)],
        *[(dict(kcount=n), 11) for n in (0, c.batch + 1, 2**64 - 1)],
    ]:
        assert decode(mixed_consumer, row, c, q, k, **options)[0] == expected
    for field, value, expected in [
        ("batch", 2**32 - 1, 1),
        ("dtype", 2, 4),
        ("total_q", 2**31, 5),
        ("max_q", 2**30 + 1, 5),
        ("capacity", 16385, 5),
        ("heads", 0, 5),
        ("heads", 2**23, 5),
        ("aic", 0, 6),
        ("aic", 27, 6),
        ("aiv", c.aic, 6),
        ("block_dim", c.block_dim + 1, 6),
        ("workspace", c.workspace - 1, 9),
        ("libapi", 2**64 - 1, 9),
    ]:
        changed = Context.from_buffer_copy(c)
        setattr(changed, field, value)
        assert decode(mixed_consumer, row, changed, q, k)[0] == expected
    for name, _ in c.memory._fields_:
        changed = Context.from_buffer_copy(c)
        setattr(changed.memory, name, 0)
        assert decode(mixed_consumer, row, changed, q, k)[0] == 10


def test_max_bounds_and_nonrectangular_capacity(mixed_consumer, mixed_builder):
    cases = [
        mixed_case(sq=512, capacity=16384),
        mixed_case([2] * 10, [511] * 10, capacity=511),
        mixed_case([2] * 10, [511] + [0] * 9, capacity=511),
        mixed_case([2**30, 2**30 - 1], [0, 0], capacity=0, heads=1),
        mixed_case([3] * 16, [16384] * 16, capacity=16384, heads=16),
    ]
    rows = build(mixed_builder, cases)
    assert [row[15] for row in rows[1:3]] == [128, 256]
    for row, values in zip(rows, cases):
        assert decode(mixed_consumer, row, context_for(row, values, 1), *prefixes(values))[0] == 0
    # Actual M/S2 tiles overflow even though Q addressing and static fields fit.
    values = mixed_case([2**27] * 15 + [2**27 - 1], [16384] * 16, capacity=16384, heads=16)
    q, k = prefixes(values)
    row = rows[-1]
    row[11], row[401], row[402] = 2**27, q[-1], 2**27
    row[PREFIX:] = wire_spans(values)
    assert decode(mixed_consumer, row, context_for(row, values, 1), q, k)[0] == 8


@pytest.mark.parametrize("dtype", (1, 2))
def test_zero_used_kv_is_not_empty_capacity(mixed_consumer, mixed_builder, dtype):
    values = mixed_case([3, 0, 1], [0, 0, 0], capacity=16384)
    row = build(mixed_builder, [values], dtype)[0]
    c = context_for(row, values, dtype)
    assert row[16] == 512 and c.workspace > 0
    assert decode(mixed_consumer, row, c, *prefixes(values))[0] == 0
    altered = row.copy()
    altered[15:17] = [0, 0]
    altered[23:401] = [0] * 378
    assert decode(mixed_consumer, altered, c, *prefixes(values))[0] == 7
    c.workspace = 0
    assert decode(mixed_consumer, row, c, *prefixes(values))[0] == 9


@pytest.mark.parametrize("dtype", (1, 2))
def test_wide_physical_capacity_and_offsets(mixed_consumer, mixed_builder, dtype):
    batch = 2**18 + 2
    values = mixed_case([0] * batch, [0] * batch, capacity=16384)
    row = build(mixed_builder, [values], dtype)[0]
    c = context_for(row, values, dtype)
    status, decoded, reads = decode(mixed_consumer, row, c, *prefixes(values))
    assert status == 0 and decoded == expected_schedule(row)
    assert decoded[-2:] == wide_words(batch * 16384)
    assert reads[1:] == [batch + 1, batch]
    for word, expected in ((405, 5), (len(row) - 3, 11)):
        altered = row.copy()
        altered[word] = 0
        assert decode(mixed_consumer, altered, c, *prefixes(values))[0] == expected
    (mixed_consumer[1] / f"wide-{dtype}.json").write_text(
        json.dumps(
            dict(
                batch=batch,
                total_kv_capacity=batch * 16384,
                last_kv_start=(batch - 1) * 16384,
                decoded=decoded,
                reads=reads,
            )
        )
        + "\n"
    )
