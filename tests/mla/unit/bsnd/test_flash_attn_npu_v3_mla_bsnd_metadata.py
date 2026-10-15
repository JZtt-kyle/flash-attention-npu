"""CPU wire/entry checks for BSND length snapshots, independently source checked."""

import ctypes as ct
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla.support.mla_metadata_build import MLA, ROOT, sha256
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    length_artifacts as length_artifacts,
    length_oracle as length_oracle,
    check_tasks,
    evaluate,
)
from tests.mla.support.components.bsnd_schedule import length_case
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata_consumer import (
    consumer as consumer,
    consume,
    make_record,
)
from tests.mla.support.components.metadata_consumer import Args

from tests.mla.support.components.bsnd_metadata import (
    MAGIC as MAGIC,
    PREFIX as PREFIX,
    SENTINEL as SENTINEL,
    arguments as arguments,
    projection as projection,
)


def metadata_cases(aic, ratio):
    result = []
    for b, (sq, h), sk in itertools.product(
        sorted({1, 2, aic - 1 or 1, aic + 1, 2 * aic + 1}),
        ((1, 17), (2, 128), (3, 171)),
        (1, 512, 513, 8193, 16384),
    ):
        patterns = [
            None,
            [sk] * b,
            [0] * b,
            [min(sk, 511)] * b,
            [[0, 1, min(sk, 512), min(sk, 513), sk][i % 5] for i in range(b)],
            [sk] + [0] * (b - 1),
        ]
        for lengths in patterns:
            result.append(
                length_case(lengths, batch=b, sq=sq, heads=h, sk=sk, aic=aic, ratio=ratio)
            )
    return result


def check_source(rows, cases, oracle, directory, name):
    directory = Path(directory)
    (directory / f"{name}-inputs.json").write_text(json.dumps(cases) + "\n")
    mapped = evaluate(oracle, cases)
    native = evaluate(oracle, cases, automatic_fd=True)
    counts = dict(
        cases=len(cases), native_non_fd=0, fa3_no_split_mapping=0, zero_kv_suffix_outside_end=0
    )
    for row, case, want, original in zip(rows, cases, mapped, native):
        assert row[:4] == [MAGIC, 2, 2, PREFIX + case[0]]
        assert row[5:8] == [0, 0, 0]
        lengths = case[16:] if case[13] else [case[3]] * case[0]
        assert row[PREFIX:] == lengths
        s = row[8:PREFIX]
        assert s[:7] == [case[0], 1, case[2], case[1], case[3], 512, 64]
        assert s[9:12] == [case[4], case[5], case[5] // case[4]]
        wide = [lo | hi << 32 for lo, hi in zip(row[367:401:2], row[368:401:2])]
        r1, r2 = wide[:2]
        cursor = 0
        for offset, size in zip(wide[4:8], (r1 * 4, r1 * 2, r2 * 4, r2 * 4)):
            assert offset == cursor and offset % 32 == 0
            cursor += 2 * s[13] * size
        assert cursor == wide[8] <= wide[9]
        assert wide[12:] == [516096, 174080 if s[11] == 2 else 178176, 65536, 65536, 131072]
        got = projection(row)
        assert got == want, (case, got, want)
        if original[5] == 0:
            assert got == original
            counts["native_non_fd"] += 1
        else:
            blocks = (case[1] * case[2] + original[1] - 1) // original[1]
            ends = sorted(
                {
                    original[18 + c] * blocks + original[44 + c] + bool(original[70 + c])
                    for c in range(original[3])
                }
            )
            assert [got[18 + c] * blocks + got[44 + c] for c in range(got[3])] == ends
            counts["fa3_no_split_mapping"] += 1
        check_tasks(case, got)
        counts["zero_kv_suffix_outside_end"] += got[18 + got[3] - 1] < case[0]
    counts["output_sha256"] = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    (directory / f"{name}.json").write_text(json.dumps(counts, indent=2) + "\n")


@pytest.fixture(scope="module")
def bsnd_builder(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_BSND_METADATA_CPU_DIR", tmp_path_factory.mktemp("mla-bsnd-metadata"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "aicpu_api.h").write_text(
        "#define __global__\n#define __aicpu__\nnamespace AscendC { inline void DataStoreBarrier() {} }\n"
    )
    common = [
        "g++",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        f"-I{directory}",
        f"-I{MLA}",
        "-x",
        "c++",
        str(MLA / "mla_bsnd_metadata.aicpu"),
        str(ROOT / "tests/mla/support/mla_bsnd_metadata_driver.cpp"),
    ]
    library = directory / "bsnd.so"
    commands = [[*common, "-O2", "-fPIC", "-shared", "-o", str(library)]]
    if os.environ.get("MLA_SCHEDULE_SANITIZE") == "1":
        binary = directory / "bsnd-sanitize"
        commands += [
            [
                *common,
                "-O1",
                "-g",
                "-DMLA_BSND_METADATA_SANITIZE",
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
    (directory / "build-manifest.json").write_text(
        json.dumps(
            dict(
                library=str(library.resolve()),
                library_sha256=sha256(library),
                sources={
                    str(p.relative_to(ROOT)): sha256(p)
                    for p in [
                        *MLA.glob("*.h"),
                        *MLA.glob("*.cpp"),
                        MLA / "mla_bsnd_metadata.aicpu",
                        ROOT / "tests/mla/support/mla_bsnd_metadata_driver.cpp",
                    ]
                },
                sanitizer=os.environ.get("MLA_SCHEDULE_SANITIZE") == "1",
            ),
            indent=2,
        )
        + "\n"
    )
    lib = ct.CDLL(str(library.resolve()))
    lib.bsnd_entry.argtypes = [ct.POINTER(Args)]
    lib.bsnd_entry.restype = ct.c_uint32
    lib.bsnd_words.argtypes = [ct.c_int64, ct.POINTER(ct.c_uint32)]
    lib.bsnd_words.restype = ct.c_bool
    return lib, directory


def build(bsnd_builder, values, dtype=1):
    args = arguments(values, dtype)
    storage = (ct.c_uint32 * (args.words + 2))(*([SENTINEL] * (args.words + 2)))
    args.output = ct.addressof(storage) + 4
    lengths = (ct.c_int32 * max(1, len(values[16:])))(*values[16:])
    args.cache_seqlens = ct.addressof(lengths) if values[13] else 0
    args.cache_seqlens_count = values[14]
    before = bytes(lengths)
    assert bsnd_builder[0].bsnd_entry(ct.byref(args)) == 0
    assert bytes(lengths) == before and storage[0] == storage[-1] == SENTINEL
    assert list(storage)[1:6] == [MAGIC, 2, 2, args.words, dtype]
    return list(storage)[1:-1]


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("aic,ratio", itertools.product((1, 15, 20, 26), (1, 2)))
def test_cpu_source_snapshots(bsnd_builder, length_oracle, dtype, aic, ratio):
    cases = metadata_cases(aic, ratio)
    rows = [build(bsnd_builder, values, dtype) for values in cases]
    check_source(rows, cases, length_oracle, bsnd_builder[1], f"cpu-{dtype}-{aic}-{ratio}")


@pytest.mark.parametrize("dtype", (1, 2))
def test_dense_and_empty_equivalence(bsnd_builder, consumer, dtype):
    for b, sq, sk in itertools.product((0, 1, 3), (0, 1, 171), (0, 1, 513, 16384)):
        dense, context = make_record(consumer, (b, sq, 3, sk), dtype)
        for lengths in (None, [sk] * b):
            row = build(
                bsnd_builder,
                length_case(
                    lengths, batch=b, sq=sq, heads=3, sk=sk, libapi=16777216, l1=524032, ub=196352
                ),
                dtype,
            )
            assert row[8:PREFIX] == list(dense)[8:] and row[PREFIX:] == [sk] * b
            record = (ct.c_uint32 * len(row))(*row)
            assert consume(consumer, record, context, words=PREFIX)[0] == 2
            if b:
                assert consume(consumer, record, context, words=len(row))[0] == 1


@pytest.mark.parametrize(
    "lengths,count,shape,status",
    [
        ([-1, 0], None, {}, [2, 1]),
        ([0, 16385], None, {}, [2, 1]),
        ([0, 2**31 - 1], None, {}, [2, 1]),
        ([1, 1], 1, {}, [3, 0]),
        (None, 2, {}, [3, 0]),
        ([0, 0], None, {"sq": -1}, [2, 1]),
        ([0, 0], None, {"heads": 0}, [2, 1]),
        ([0, 0], None, {"aic": 0}, [2, 3]),
        ([0, 0], None, {"ub": 0}, [2, 4]),
        ([0, 0], None, {"libapi": 2**64 - 1}, [2, 5]),
        ([0, 1], None, {"sk": 0}, [2, 1]),
        ([0, -1], None, {"sq": 0}, [2, 1]),
    ],
)
def test_failure_record_clears_snapshot(bsnd_builder, lengths, count, shape, status):
    shape = dict(batch=2, sk=16384) | shape
    row = build(bsnd_builder, length_case(lengths, count=count, **shape))
    assert row[5:7] == status and not any(row[8:])


@pytest.mark.parametrize(
    "batch,expected",
    [
        (-1, None),
        (0, 401),
        (1, 402),
        (2**32 - 402, 2**32 - 1),
        (2**32 - 401, None),
        (2**63 - 1, None),
    ],
)
def test_static_word_count_bounds(bsnd_builder, batch, expected):
    words = ct.c_uint32(SENTINEL)
    assert bsnd_builder[0].bsnd_words(batch, ct.byref(words)) == (expected is not None)
    assert words.value == (SENTINEL if expected is None else expected)


@pytest.mark.parametrize("dtype,ratio", itertools.product((1, 2), (1, 2)))
def test_zero_suffix_preserves_source_endpoint(bsnd_builder, length_oracle, dtype, ratio):
    values = length_case(
        [0, 1, 511, 512, 513] * 3 + [0], batch=16, sq=1, heads=1, sk=513, aic=15, ratio=ratio
    )
    row = build(bsnd_builder, values, dtype)
    check_source([row], [values], length_oracle, bsnd_builder[1], f"suffix-{dtype}-{ratio}")
    assert row[23 + row[20] - 1] == 15 and row[-1] == 0
