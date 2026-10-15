"""v4 mixed-layout CPU wire/buffer checks with an independent pinned-source oracle."""

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla.support.mla_cpu_driver import evaluate_cpu
from tests.mla.support.mla_metadata_build import MLA, ROOT, sha256
from tests.mla.support.components.bsnd_metadata import projection
from tests.mla.support.components.tnd_bsnd_schedule import (
    mixed_artifacts as mixed_artifacts,
    mixed_oracle as mixed_oracle,
    mixed_case,
)

from tests.mla.support.components.tnd_bsnd_metadata import (
    EMPTY_CASES as EMPTY_CASES,
    FAILURES as FAILURES,
    MAGIC as MAGIC,
    PREFIX as PREFIX,
    SENTINEL as SENTINEL,
    check_records as check_records,
    metadata_cases as metadata_cases,
    wide_words as wide_words,
    wire_spans as wire_spans,
)


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
        str(ROOT / "tests/mla/support/mla_tnd_bsnd_metadata_driver.cpp"),
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
        ROOT / "tests/mla/support/mla_tnd_bsnd_metadata_driver.cpp",
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


def build(builder, cases, dtype=1, *, from_prefix=False):
    rows = evaluate_cpu(
        builder[0], cases, args=[str(dtype), *(["--from-prefix"] if from_prefix else [])]
    )
    assert len(rows) == len(cases)
    return rows


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


@pytest.mark.parametrize("dtype", (1, 2))
def test_total_from_prefix_matches_explicit_and_source(mixed_builder, mixed_oracle, dtype):
    cases = metadata_cases(20, 2)
    for c in cases:
        c[12] = 20  # Public nonempty records bind a full-core launch.
    rows = build(mixed_builder, cases, dtype, from_prefix=True)
    assert rows == build(mixed_builder, cases, dtype)
    check_records(rows, cases, mixed_oracle, mixed_builder[1], f"inferred-{dtype}", dtype)
    empty = [mixed_case(q, k, sq=17, capacity=sk) for q, k, sk in EMPTY_CASES]
    assert build(mixed_builder, empty, dtype, from_prefix=True) == build(
        mixed_builder, empty, dtype
    )
    # totalQuery in the private ABI is unused only by the new public producer.
    c = mixed_case(total_q=-1)
    assert build(mixed_builder, [c], dtype, from_prefix=True) == build(
        mixed_builder, [mixed_case(block_dim=20)], dtype
    )


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
