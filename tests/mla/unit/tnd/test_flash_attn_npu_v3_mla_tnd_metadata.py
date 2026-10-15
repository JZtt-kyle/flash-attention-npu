"""v3 CPU wire checks and pinned-source differential, including full sanitizer runs."""

import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla.support.mla_metadata_build import MLA, ROOT, sha256
from tests.mla.support.components.bsnd_metadata import projection
from tests.mla.support.components.tnd_schedule import (
    tnd_artifacts as tnd_artifacts,
    tnd_oracle as tnd_oracle,
    expected_spans,
    tnd_case,
)

from tests.mla.support.components.tnd_metadata import (
    FAILURES as FAILURES,
    MAGIC as MAGIC,
    PREFIX as PREFIX,
    SENTINEL as SENTINEL,
    check_records as check_records,
    metadata_cases as metadata_cases,
)


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
        str(ROOT / "tests/mla/support/mla_tnd_metadata_driver.cpp"),
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
        ROOT / "tests/mla/support/mla_tnd_metadata_driver.cpp",
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
