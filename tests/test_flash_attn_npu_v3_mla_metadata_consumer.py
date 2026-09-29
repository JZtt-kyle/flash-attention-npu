"""CPU checks of the device decoder, including truncated and corrupt records."""

import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla_metadata_build import MLA, ROOT, sha256


WORDS = 401
SENTINEL = 0x12345678


class Memory(ct.Structure):
    _fields_ = [(n, ct.c_uint64) for n in ("l1", "ub", "l0a", "l0b", "l0c")]


class Context(ct.Structure):
    _fields_ = [
        (n, ct.c_uint32) for n in ("dtype", "batch", "sq", "heads", "sk", "aic", "aiv", "block_dim")
    ] + [("workspace", ct.c_uint64), ("libapi", ct.c_uint64), ("memory", Memory)]


class Shape(ct.Structure):
    _fields_ = [(n, ct.c_int64) for n in ("batch", "sq", "heads", "sk")]


class Platform(ct.Structure):
    _fields_ = [
        ("aic", ct.c_uint32),
        ("aiv", ct.c_uint32),
        ("memory", Memory),
        ("libapi", ct.c_uint64),
        ("block_dims", ct.c_uint32 * 26),
    ]


class Args(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", Shape),
        ("platform", Platform),
        ("cache_seqlens", ct.c_uint64),
        ("cache_seqlens_count", ct.c_uint64),
    ]


# Wire v1 offsets, independently fixed by its documented serialized layout.
# Include valid-header failures, both halves of 64-bit fields and partition holes.
CORRUPTIONS = [
    (0, 0, 2),
    (1, 2, 2),
    (2, 2, 2),
    (3, WORDS - 1, 2),
    (7, 1, 2),
    (5, 1, 3),
    (6, 1, 3),
    (4, 3, 4),
    *[(i, 0, 5) for i in range(8, 15)],
    (17, 0, 6),
    (18, 20, 6),
    (19, 1, 6),
    (20, 0, 6),
    (20, 27, 6),
    (21, 0, 6),
    (21, 21, 6),
    (22, 0, 6),
    (15, 0, 7),
    (15, 128, 7),
    (15, 257, 7),
    (16, 512, 7),
    (103, 1, 7),
    (104, 1, 7),
    (105, 1, 7),
    (23, 0, 8),
    (23, 3, 8),
    (24, 1, 8),
    (49, 1, 8),
    (75, 1, 8),
    *[(i, 0xFFFFFFFF, 9) for i in (367, 368, 369, 370, *range(375, 391))],
    *[(i, 0xFFFFFFFF, 10) for i in range(391, 401)],
]


@pytest.fixture(scope="module")
def consumer(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_CONSUMER_ARTIFACT_DIR", tmp_path_factory.mktemp("mla-consumer"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / "tests/mla_metadata_consumer_driver.cpp"
    sources = [
        source,
        MLA / "mla_metadata.cpp",
        MLA / "mla_schedule.cpp",
        MLA / "detail/cann_split_core.cpp",
    ]
    common = ["g++", "-std=c++17", f"-I{MLA}", *(str(p) for p in sources)]
    library = directory / "consumer.so"
    commands = [[*common, "-O2", "-fPIC", "-shared", "-o", str(library)]]
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
        for cmd in commands:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    sources += [
        MLA / n
        for n in (
            "mla_metadata.h",
            "mla_metadata_args.h",
            "mla_metadata_consumer.h",
            "mla_dense_params.h",
            "mla_schedule.h",
            "detail/cann_split_core.h",
        )
    ]
    manifest = dict(
        library=str(library.resolve()),
        library_sha256=sha256(library),
        sources={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        sanitizer=os.environ.get("MLA_CONSUMER_SANITIZE") == "1",
    )
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lib = ct.CDLL(str(library.resolve()))
    lib.build_metadata.argtypes = [ct.POINTER(Args), ct.c_void_p, ct.POINTER(Context)]
    lib.consume_metadata.argtypes = [
        ct.c_void_p,
        ct.c_uint32,
        ct.POINTER(Context),
        ct.c_void_p,
        ct.POINTER(ct.c_uint32),
    ]
    lib.consume_metadata.restype = ct.c_uint32
    return lib, directory


def make_record(consumer, shape=(2, 1, 8, 257), dtype=1, aic=20, block_dim=0, libapi=16777216):
    args = Args(
        0,
        WORDS,
        dtype,
        Shape(*shape),
        Platform(
            aic,
            aic * 2,
            Memory(524032, 196352, 65536, 65536, 131072),
            libapi,
            (ct.c_uint32 * 26)(*[block_dim or i for i in range(1, 27)]),
        ),
    )
    record, context = (ct.c_uint32 * WORDS)(), Context()
    consumer[0].build_metadata(ct.byref(args), record, ct.byref(context))
    assert record[5] == record[6] == 0
    return record, context


def consume(consumer, record, context, words=WORDS):
    # Exact Kernel schedule size, 11 scalar words and 3 arrays of 26 endpoints.
    output = (ct.c_uint32 * 89)(*[SENTINEL] * 89)
    reads = ct.c_uint32()
    result = consumer[0].consume_metadata(record, words, ct.byref(context), output, ct.byref(reads))
    if result:
        assert list(output) == [SENTINEL] * 89
    return result, list(output), reads.value


@pytest.mark.parametrize("dtype", (1, 2))
def test_valid_partition_matrix(consumer, dtype):
    count = 0
    for aic in (1, 2, 3, 20, 24, 26):
        shapes = itertools.product(
            sorted({1, 2, max(1, aic // 2), aic - 1 or 1, aic, aic + 1, 2 * aic + 1}),
            ((1, 1), (1, 17), (1, 127), (7, 16), (1, 128), (1, 129), (1, 255), (2, 128)),
            (1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 257, 511, 512),
        )
        for b, (sq, h), sk in shapes:
            record, context = make_record(consumer, (b, sq, h, sk), dtype, aic)
            status, output, _ = consume(consumer, record, context)
            assert status == 0, (b, sq, h, sk, aic, status)
            assert output[:9] == [b, sq, h, sk, record[15], record[16], record[20], record[21], 2]
            assert output[9:11] == [record[367], record[369]]
            for axis in range(3):
                used = record[20]
                assert (
                    output[11 + axis * 26 : 11 + axis * 26 + used]
                    == list(record)[23 + axis * 26 : 23 + axis * 26 + used]
                )
            count += 1
    (consumer[1] / f"matrix-{dtype}.json").write_text(json.dumps(dict(cases=count)) + "\n")


@pytest.mark.parametrize("word,value,expected", CORRUPTIONS)
def test_corrupt_record(consumer, word, value, expected):
    record, context = make_record(consumer)
    record[word] = value
    status, _, reads = consume(consumer, record, context)
    assert status == expected
    if word < 8:
        assert reads <= 8


@pytest.mark.parametrize("words", (0, 8, WORDS - 1, WORDS + 1, 2**32 - 1))
def test_buffer_size_before_header(consumer, words):
    record, context = make_record(consumer)
    assert consume(consumer, record, context, words)[::2] == (1, 0)


def test_context_and_wide_bounds(consumer):
    record, context = make_record(consumer)
    # Use the largest exact total representable, with the actual reserve.
    reserve = context.workspace
    record, context = make_record(consumer, libapi=2**64 - 1 - reserve)
    assert record[389] == record[390] == 0xFFFFFFFF
    assert consume(consumer, record, context)[0] == 0
    context.libapi += 1
    assert consume(consumer, record, context)[0] == 9
    record, context = make_record(consumer)
    context.workspace -= 1
    assert consume(consumer, record, context)[0] == 9
    for name, _ in Memory._fields_:
        record, context = make_record(consumer)
        setattr(context.memory, name, 0)
        assert consume(consumer, record, context)[0] == 10
    record, context = make_record(consumer)
    context.sq = context.heads = 2**32 - 1
    assert consume(consumer, record, context)[0] == 5


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("dtype", 2, 4),
        ("batch", 1, 5),
        ("sq", 2, 5),
        ("heads", 16, 5),
        ("sk", 65, 5),
        ("aic", 21, 6),
        ("aiv", 20, 6),
        ("block_dim", 20, 6),
        ("workspace", 0, 9),
        ("libapi", 0, 9),
    ],
)
def test_context_mismatch(consumer, name, value, expected):
    record, context = make_record(consumer)
    setattr(context, name, value)
    assert consume(consumer, record, context)[0] == expected


def test_extra_launch_blocks(consumer):
    record, context = make_record(consumer, block_dim=20)
    status, output, _ = consume(consumer, record, context)
    assert status == 0 and output[6:8] == [2, 20]
