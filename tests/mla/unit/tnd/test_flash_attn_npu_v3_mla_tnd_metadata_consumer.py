"""v3 shared decoder: source schedules, compact spans and rejection boundaries."""

import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla.support.mla_metadata_build import MLA, ROOT, sha256
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata_consumer import CORRUPTIONS
from tests.mla.unit.tnd.test_flash_attn_npu_v3_mla_tnd_metadata import (
    tnd_builder as tnd_builder,
    build,
)
from tests.mla.support.components.tnd_metadata import (
    tnd_artifacts as tnd_artifacts,
    tnd_oracle as tnd_oracle,
    PREFIX,
    SENTINEL,
    check_records,
    metadata_cases,
    prefixes,
    tnd_case,
)

from tests.mla.support.components.tnd_metadata_consumer import (
    Context as Context,
    context_for as context_for,
    expected_schedule as expected_schedule,
)


TND_CORRUPTIONS = CORRUPTIONS + [
    (1, 1, 2),
    (2, 1, 2),
    *[(i, 0xFFFFFFFF, 5) for i in range(401, 405)],
    *[(i, 0xFFFFFFFF, 12) for i in range(405, 413)],
]


@pytest.fixture(scope="module")
def tnd_consumer(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_TND_CONSUMER_CPU_DIR", tmp_path_factory.mktemp("mla-tnd-consumer"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "tests/mla/support/mla_tnd_metadata_consumer_driver.cpp"] + [
        MLA / name
        for name in (
            "mla_tnd_metadata.cpp",
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
    lib.consume_tnd_metadata.argtypes = [
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
    lib.consume_tnd_metadata.restype = ct.c_uint32
    return lib, directory


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
    output = (ct.c_uint32 * 91)(*([SENTINEL] * 91))
    reads = (ct.c_uint32 * 3)()
    status = consumer[0].consume_tnd_metadata(
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
        assert list(output) == [SENTINEL] * 91
    return status, list(output), list(reads)


@pytest.mark.parametrize("dtype,aic", itertools.product((1, 2), (1, 15, 20, 26)))
def test_source_records(tnd_consumer, tnd_builder, tnd_oracle, dtype, aic):
    cases = metadata_cases(aic, 2)
    rows = build(tnd_builder, cases, dtype)
    for values, row in zip(cases, rows):
        status, schedule, _ = decode(
            tnd_consumer, row, context_for(row, values, dtype), *prefixes(values)
        )
        assert status == 0, (values, status)
        assert schedule == expected_schedule(row)
    check_records(rows, cases, tnd_oracle, tnd_consumer[1], f"source-{dtype}-{aic}", dtype)


@pytest.mark.parametrize("word,value,expected", TND_CORRUPTIONS)
def test_corrupt_record(tnd_consumer, tnd_builder, word, value, expected):
    values = tnd_case([1, 1], [257, 257], heads=8)
    row = build(tnd_builder, [values])[0]
    c = context_for(row, values, 1)
    assert row[word] != value
    row[word] = value
    status, _, reads = decode(tnd_consumer, row, c, *prefixes(values))
    assert status == expected
    if word < 8:
        assert reads[0] <= 8 and reads[1:] == [0, 0]


@pytest.mark.parametrize("dtype", (1, 2))
def test_every_current_prefix_and_span(tnd_consumer, tnd_builder, dtype):
    values = tnd_case([3, 0, 1] * 14, [0, 8193, 513] * 14, sq=512, sk=16384)
    row = build(tnd_builder, [values], dtype)[0]
    c = context_for(row, values, dtype)
    q, k = prefixes(values)
    for axis in range(2):
        for b in range(c.batch + 1):
            for bad in (-1, -(2**31), 2**31 - 1, (q, k)[axis][b] + 1):
                current = [q.copy(), k.copy()]
                current[axis][b] = bad
                assert decode(tnd_consumer, row, c, *current)[0] == 12
    for i in range(PREFIX, len(row)):
        altered = row.copy()
        altered[i] += 1
        assert decode(tnd_consumer, altered, c, q, k)[0] == 12


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("q,k", [([], []), ([0], [0]), ([0, 0], [0, 513]), ([3, 0, 1], [0, 0, 0])])
def test_empty_still_validates(tnd_consumer, tnd_builder, dtype, q, k):
    values = tnd_case(q, k, sq=17, sk=16384)
    row = build(tnd_builder, [values], dtype)[0]
    c = context_for(row, values, dtype)
    cq, ck = prefixes(values)
    assert decode(tnd_consumer, row, c, cq, ck)[1] == expected_schedule(row)
    for field in (1, 2):
        for version in (1, 2):
            altered = row.copy()
            altered[field] = version
            assert decode(tnd_consumer, altered, c, cq, ck)[0] == 2
    for axis in range(2):
        current = [cq.copy(), ck.copy()]
        current[axis][-1] += 1
        assert decode(tnd_consumer, row, c, *current)[0] == 12
    for word in (15, 16, *range(23, 401)):
        altered = row.copy()
        altered[word] = 1
        assert decode(tnd_consumer, altered, c, cq, ck)[0] == 7


def test_variable_m_endpoints_and_zero_cost_suffix(tnd_consumer, tnd_builder):
    values = tnd_case([3, 0, 1, 7, 0], [513, 7, 1, 0, 16384], heads=171)
    row = build(tnd_builder, [values])[0]
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
        assert decode(tnd_consumer, changed, c, *prefixes(values))[0] == expected


def test_context_descriptors_and_overflow(tnd_consumer, tnd_builder):
    values = tnd_case()
    row = build(tnd_builder, [values])[0]
    c = context_for(row, values, 1)
    q, k = prefixes(values)
    for options, expected in [
        *[(dict(words=w), 1) for w in (0, 8, 401, len(row) - 1, len(row) + 1, 2**32 - 1)],
        *[(dict(address=a), 1) for a in (0, 2, 2**64 - 4)],
        *[(dict(qaddress=a), 12) for a in (0, 2, 2**64 - 4)],
        *[(dict(kaddress=a), 12) for a in (0, 2, 2**64 - 4)],
        *[(dict(qcount=n), 12) for n in (0, c.batch, 2**64 - 1)],
        *[(dict(kcount=n), 12) for n in (0, c.batch, 2**64 - 1)],
    ]:
        assert decode(tnd_consumer, row, c, q, k, **options)[0] == expected
    for field, value, expected in [
        ("batch", 2**32 - 1, 1),
        ("dtype", 2, 4),
        ("total_q", 2**31, 5),
        ("total_k", 2**31, 5),
        ("max_q", 2**30 + 1, 5),
        ("max_k", 16385, 5),
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
        assert decode(tnd_consumer, row, changed, q, k)[0] == expected
    for name, _ in c.memory._fields_:
        changed = Context.from_buffer_copy(c)
        setattr(changed.memory, name, 0)
        assert decode(tnd_consumer, row, changed, q, k)[0] == 10


def test_max_bounds_and_nonrectangular_capacity(tnd_consumer, tnd_builder):
    cases = [
        tnd_case(sq=512, sk=16384),
        tnd_case([2] * 10, [511] * 10),
        tnd_case([2] * 10, [511] + [0] * 9),
        tnd_case([2**30, 2**30 - 1], [0, 0], heads=1),
        tnd_case([3] * 16, [16384] * 16, heads=16),
    ]
    rows = build(tnd_builder, cases)
    assert [row[15] for row in rows[1:3]] == [256, 128]
    for row, values in zip(rows, cases):
        assert decode(tnd_consumer, row, context_for(row, values, 1), *prefixes(values))[0] == 0
    # Legal fields but too many *actual* physical tiles. Reject before endpoints
    # or workspace, without allocating the huge Q/KV tensors on CPU or device.
    values = tnd_case([2**27] * 15 + [2**27 - 1], [16384] * 16, heads=16)
    q, k = prefixes(values)
    row = rows[-1]
    row[11], row[401], row[403] = 2**27, q[-1], 2**27
    row[PREFIX:] = [v for b in range(16) for v in (q[b], k[b], q[b + 1] - q[b], k[b + 1] - k[b])]
    assert decode(tnd_consumer, row, context_for(row, values, 1), q, k)[0] == 8
