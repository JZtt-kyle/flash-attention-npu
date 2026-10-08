"""Shared CPU/device v2 decoder: source records, stale lengths and unsafe holes."""

import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla_metadata_build import MLA, ROOT, sha256
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata import (
    PREFIX,
    SENTINEL,
    arguments,
    bsnd_builder as bsnd_builder,
    build,
    check_source,
    metadata_cases,
)
from tests.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    length_artifacts as length_artifacts,
    length_case,
    length_oracle as length_oracle,
)
from tests.test_flash_attn_npu_v3_mla_metadata_consumer import CORRUPTIONS, Context


BSND_CORRUPTIONS = [
    (word, 1 if word in (1, 2) else value, status) for word, value, status in CORRUPTIONS
] + [(i, value, 11) for i in (401, 402) for value in (0, 256, 258, 0xFFFFFFFF)]


@pytest.fixture(scope="module")
def bsnd_consumer(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_BSND_CONSUMER_CPU_DIR", tmp_path_factory.mktemp("mla-bsnd-consumer"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "tests/mla_bsnd_metadata_consumer_driver.cpp"] + [
        MLA / name
        for name in (
            "mla_bsnd_metadata.cpp",
            "mla_bsnd_schedule.cpp",
            "mla_metadata.cpp",
            "mla_schedule.cpp",
            "detail/cann_split_core.cpp",
        )
    ]
    common = ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{MLA}"]
    common += [str(p) for p in sources]
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
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
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
    lib.consume_bsnd_metadata.argtypes = [
        ct.c_void_p,
        ct.c_uint32,
        ct.POINTER(Context),
        ct.c_void_p,
        ct.c_uint64,
        ct.c_void_p,
        ct.POINTER(ct.c_uint32),
    ]
    lib.consume_bsnd_metadata.restype = ct.c_uint32
    return lib, directory


def context_for(row, values, dtype):
    # Independent trusted generation supplies workspace/blockDim. Mutations
    # never change this caller context, and shape/platform come from the call.
    p = arguments(values, dtype).platform
    return Context(
        dtype,
        *values[:4],
        p.aic,
        p.aiv,
        row[21],
        row[385] | row[386] << 32,
        0 if not all(values[i] for i in (0, 1, 3)) else p.libapi,
        p.memory,
    )


def decode(
    consumer, row, context, lengths, *, words=None, count=None, address=None, length_address=None
):
    storage = (ct.c_uint32 * (len(row) + 2))(SENTINEL, *row, SENTINEL)
    lens = (ct.c_int32 * max(1, len(lengths or [])))(*(lengths or []))
    output = (ct.c_uint32 * 89)(*([SENTINEL] * 89))
    reads = (ct.c_uint32 * 2)()
    result = consumer[0].consume_bsnd_metadata(
        ct.addressof(storage) + 4 if address is None else address,
        len(row) if words is None else words,
        ct.byref(context),
        (ct.addressof(lens) if lengths is not None else 0)
        if length_address is None
        else length_address,
        (len(lengths) if lengths is not None else 0) if count is None else count,
        output,
        reads,
    )
    assert list(storage) == [SENTINEL, *row, SENTINEL]
    assert list(lens)[: len(lengths or [])] == (lengths or [])
    if result:
        assert list(output) == [SENTINEL] * 89
    return result, list(output), list(reads)


def expected_schedule(row):
    used = row[20]
    return [row[i] for i in (8, 11, 10, 12, 15, 16, 20, 21, 19, 367, 369)] + [
        row[23 + axis * 26 + core] if core < used else 0 for axis in range(3) for core in range(26)
    ]


@pytest.mark.parametrize("dtype,aic", itertools.product((1, 2), (1, 15, 20, 26)))
def test_source_records(bsnd_consumer, bsnd_builder, length_oracle, dtype, aic):
    cases = metadata_cases(aic, 2)
    rows = [build(bsnd_builder, values, dtype) for values in cases]
    for values, row in zip(cases, rows):
        current = values[16:] if values[13] else None
        status, schedule, _ = decode(bsnd_consumer, row, context_for(row, values, dtype), current)
        assert status == 0, (values, status)
        assert schedule == expected_schedule(row)
    check_source(rows, cases, length_oracle, bsnd_consumer[1], f"source-{dtype}-{aic}")


@pytest.mark.parametrize("word,value,expected", BSND_CORRUPTIONS)
def test_corrupt_record(bsnd_consumer, bsnd_builder, word, value, expected):
    values = length_case([257, 257], batch=2, sq=1, heads=8, sk=257)
    row = build(bsnd_builder, values)
    context = context_for(row, values, 1)
    row[word] = value
    status, _, reads = decode(bsnd_consumer, row, context, [257, 257])
    assert status == expected
    if word < 8:
        assert reads[0] <= 8 and reads[1] == 0


@pytest.mark.parametrize("dtype", (1, 2))
def test_stale_lengths_and_implicit_capacity(bsnd_consumer, bsnd_builder, dtype):
    lengths = [0, 1, 511, 512, 513, 8193, 16384] * 6
    values = length_case(lengths, batch=len(lengths), sq=3, heads=171, sk=16384)
    row = build(bsnd_builder, values, dtype)
    c = context_for(row, values, dtype)
    assert decode(bsnd_consumer, row, c, lengths)[0] == 0
    assert decode(bsnd_consumer, row, c, None)[0] == 11
    for b in range(len(lengths)):
        for value in (-1, 16385, (lengths[b] + 1) % 16385):
            current = lengths.copy()
            current[b] = value
            assert decode(bsnd_consumer, row, c, current)[0] == 11
            altered = row.copy()
            altered[PREFIX + b] = value & 0xFFFFFFFF
            assert decode(bsnd_consumer, altered, c, lengths)[0] == 11


@pytest.mark.parametrize("dtype", (1, 2))
def test_empty_capacities_still_check_lengths(bsnd_consumer, bsnd_builder, dtype):
    for b, sq, sk in itertools.product((0, 2), (0, 3), (0, 16384)):
        if b * sq * sk:
            continue
        for lengths in (None, [sk] * b, [0] * b):
            values = length_case(lengths, batch=b, sq=sq, sk=sk)
            row = build(bsnd_builder, values, dtype)
            c = context_for(row, values, dtype)
            assert decode(bsnd_consumer, row, c, lengths)[0] == 0
            if b:
                current = row[PREFIX:]
                current[-1] = -1
                assert decode(bsnd_consumer, row, c, current)[0] == 11
            row[1:3] = [1, 1]  # B=0 must still distinguish v1 from v2.
            assert decode(bsnd_consumer, row, c, lengths)[0] == 2


def test_zero_suffix_and_partial_sample_holes(bsnd_consumer, bsnd_builder):
    values = length_case([0, 1, 511, 512, 513] * 3 + [0], batch=16, sq=1, heads=1, sk=513, aic=15)
    row = build(bsnd_builder, values)
    assert row[23 + row[20] - 1] == 15
    assert decode(bsnd_consumer, row, context_for(row, values, 1), values[16:])[0] == 0
    # A hole containing nonzero work is unsafe even if the last sample is zero.
    values = length_case([1, 0, 7, 0, 0], batch=5, sq=3, heads=171, sk=513)
    row = build(bsnd_builder, values)
    c = context_for(row, values, 1)
    row[20], row[23], row[49], row[75] = 1, 1, 0, 0
    assert decode(bsnd_consumer, row, c, values[16:])[0] == 8
    row[23] = 3
    assert decode(bsnd_consumer, row, c, values[16:])[0] == 0
    row[49] = 1
    assert decode(bsnd_consumer, row, c, values[16:])[0] == 8


def test_buffer_descriptor_and_wide_bounds(bsnd_consumer, bsnd_builder):
    values = length_case([0, 513], batch=2, sq=3, heads=171, sk=16384)
    row = build(bsnd_builder, values)
    c = context_for(row, values, 1)
    for words in (0, 8, 401, 402, 404, 2**32 - 1):
        status, _, reads = decode(bsnd_consumer, row[:8], c, [0, 513], words=words)
        assert status == 1 and reads == [0, 0]
    for options in (dict(address=0), dict(address=2), dict(address=2**64 - 4)):
        assert decode(bsnd_consumer, row, c, [0, 513], **options)[0] == 1
    for options in (
        dict(count=0),
        dict(count=1),
        dict(count=3),
        dict(count=2**64 - 1),
        dict(length_address=2),
        dict(length_address=0),
        dict(length_address=2**64 - 4),
    ):
        assert decode(bsnd_consumer, row, c, [0, 513], **options)[0] == 11
    c.batch = 2**32 - 1
    assert decode(bsnd_consumer, row[:8], c, None, words=2**32 - 1)[::2] == (1, [0, 0])
    c = context_for(row, values, 1)
    for field, value, expected in (
        ("aiv", 20, 6),
        ("workspace", c.workspace - 1, 9),
        ("libapi", 2**64 - 1, 9),
        ("sq", 2**32 - 1, 5),
    ):
        altered = Context.from_buffer_copy(c)
        setattr(altered, field, value)
        assert decode(bsnd_consumer, row, altered, [0, 513])[0] == expected
