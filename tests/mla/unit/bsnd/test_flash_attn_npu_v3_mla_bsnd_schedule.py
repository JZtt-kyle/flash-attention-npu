"""Independent CPU checks for BSND actual KV lengths; no production/NPU entry.

The oracle compiles pinned CANN SplitCore plus verbatim host length/base/workspace
functions. Native source output and the FA3 num_splits=1 mapping are counted
separately. MLA_LENGTH_SCHEDULE_ARTIFACT_DIR preserves inputs/digests/manifests;
MLA_SCHEDULE_SANITIZE=1 enables ASan/UBSan for both executables.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla.support.mla_cpu_driver import evaluate_cpu
from tests.mla.support.components.schedule import (
    CANN_COMMIT,
    FIA,
    FUNCTIONS,
    MLA,
    ROOT,
    SOURCE_FILES,
    run,
    source_function,
)

from tests.mla.support.components.bsnd_schedule import (
    length_case as length_case,
)


@pytest.fixture(scope="session")
def length_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_LENGTH_SCHEDULE_ARTIFACT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-bsnd-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def compile_driver(directory, oracle):
    name = "oracle" if oracle else "schedule"
    binary = directory / name
    command = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O2",
        "-g",
        "-D_GLIBCXX_ASSERTIONS",
        "-DBSND_KV_LENGTHS",
    ]
    if os.environ.get("MLA_SCHEDULE_SANITIZE") == "1":
        command += [
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
        ]
    if oracle:
        command += ["-DSOURCE_ORACLE", "-I" + str(directory), str(directory / "split_core.cpp")]
    else:
        command += ["-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(MLA)]
        command += [
            str(MLA / name)
            for name in ("mla_bsnd_schedule.cpp", "mla_schedule.cpp", "detail/cann_split_core.cpp")
        ]
    command += [str(ROOT / "tests/mla/support/mla_schedule_driver.cpp"), "-o", str(binary)]
    (directory / f"{name}-compile.json").write_text(json.dumps(command, indent=2) + "\n")
    result = run(command)
    (directory / f"{name}-compile.log").write_text(result.stdout + result.stderr)
    return binary


@pytest.fixture(scope="session")
def length_scheduler(length_artifacts):
    return compile_driver(length_artifacts, oracle=False)


@pytest.fixture(scope="session")
def length_oracle(length_artifacts):
    source_root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    # Missing pinned source is an acceptance failure for this migration step.
    blobs = {
        name: subprocess.check_output(
            ["git", "-C", str(source_root), "show", CANN_COMMIT + ":" + name], timeout=30
        )
        for name in SOURCE_FILES
    }
    functions = FUNCTIONS + ["DealSameSeqEachBatch", "CreateSplitInput"]
    manifest = {
        "commit": CANN_COMMIT,
        "functions": functions,
        "split_core_functions": ["GetS1SeqSize", "GetS2SeqSize", "SplitCore"],
        "files": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
    }
    for name, blob in blobs.items():
        (length_artifacts / Path(name).name).write_bytes(blob)
    source = blobs[FIA + "fia_tiling_nonquant_mla.cpp"].decode()
    header = blobs[FIA + "fia_tiling_nonquant_mla_arch22.h"].decode()
    for default in ("mBaseSize_ = 256", "mFdBaseSize_ = 8", "splitKVFlag_ = false"):
        assert default in header
    bodies = ["template <typename T>\n" + source_function(source, "inline auto Align(")]
    bodies += [source_function(source, "FiaTilingNonQuantMla::" + name + "(") for name in functions]
    (length_artifacts / "source_functions.inc").write_text("\n".join(bodies))
    (length_artifacts / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (length_artifacts / "log").mkdir(exist_ok=True)
    (length_artifacts / "log/log.h").write_text(
        "#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n"
    )
    return compile_driver(length_artifacts, oracle=True)


def evaluate(binary, cases, *, automatic_fd=False, dense=False):
    outputs = evaluate_cpu(
        binary,
        cases,
        options={
            "MLA_ORACLE_AUTOMATIC_FD": "1" if automatic_fd else None,
            "MLA_BSND_USE_DENSE": "1" if dense else None,
        },
    )
    assert len(outputs) == len(cases)
    return outputs


def check_tasks(shape, result):
    batch, sq, heads, capacity = shape[:4]
    lengths = shape[16:] if shape[13] else [capacity] * batch
    mbase, nbase, cores = result[1:4]
    blocks = (sq * heads + mbase - 1) // mbase
    ends = [result[18 + c] * blocks + result[44 + c] for c in range(cores)]
    assert all(result[44 + c] < blocks for c in range(cores))
    assert ends[-1] <= batch * blocks
    if ends[-1] < batch * blocks:
        assert ends[-1] % blocks == 0
        assert all(lengths[b] == 0 for b in range(ends[-1] // blocks, batch))
    assert all(b > a for a, b in zip([0, *ends[:-1]], ends))
    assert result[70:96] == [0] * 26
    assert result[5:8] == [0, 0, 0]
    # Independently enumerate compute tiles. The source initializes empty O/LSE
    # separately; trailing zero-KV samples can be outside the compute endpoints.
    visited = []
    costs = []
    begin = 0
    for end in ends:
        cost = 0
        for row in range(begin, end):
            b, m = divmod(row, blocks)
            visited.append((b, m))
            msize = min(mbase, sq * heads - m * mbase)
            for start in range(0, lengths[b], nbase):
                nsize = min(nbase, lengths[b] - start)
                cost += 6 * ((msize + 15) // 16) + 10 * ((nsize + 63) // 64)
        costs.append(cost)
        begin = end
    assert [row for row in visited if lengths[row[0]] > 0] == [
        (b, m) for b, m in itertools.product(range(batch), range(blocks)) if lengths[b] > 0
    ]
    assert result[4] == max(costs)
    assert result[9] > 0 and result[13] > 0  # Capacity-based workspace, even for all-zero lengths.


def compare(scheduler, oracle, cases, directory, name):
    # Retain exact inputs even if a differential or sanitizer check fails.
    (directory / f"{name}-inputs.json").write_text(json.dumps(cases) + "\n")
    actual = evaluate(scheduler, cases)
    mapped = evaluate(oracle, cases)
    native = evaluate(oracle, cases, automatic_fd=True)
    counts = dict(native_non_fd=0, fa3_no_split_mapping=0, zero_kv_suffix_outside_end=0)
    for shape, got, want, source in zip(cases, actual, mapped, native):
        assert got == want, (shape, got, want)
        assert got[0] == 0 and len(got) == 356
        if source[5] == 0:
            assert got == source, (shape, got, source)
            counts["native_non_fd"] += 1
        else:
            blocks = (shape[1] * shape[2] + source[1] - 1) // source[1]
            expected_ends = sorted(
                {
                    source[18 + c] * blocks + source[44 + c] + bool(source[70 + c])
                    for c in range(source[3])
                }
            )
            assert [got[18 + c] * blocks + got[44 + c] for c in range(got[3])] == expected_ends
            counts["fa3_no_split_mapping"] += 1
        check_tasks(shape, got)
        if got[18 + got[3] - 1] < shape[0]:
            counts["zero_kv_suffix_outside_end"] += 1
    report = dict(
        cases=len(cases),
        **counts,
        all_fields_equal=True,
        tasks_and_costs_checked=True,
        input_sha256=hashlib.sha256(json.dumps(cases).encode()).hexdigest(),
        output_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest(),
        native_output_sha256=hashlib.sha256(json.dumps(native).encode()).hexdigest(),
    )
    (directory / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return actual


@pytest.mark.parametrize("aic,ratio", itertools.product(range(1, 27), (1, 2)))
def test_actual_kv_source_boundaries(length_scheduler, length_oracle, length_artifacts, aic, ratio):
    cases = []
    for batch in sorted({1, max(1, aic - 1), aic, aic + 1, 2 * aic + 1}):
        for sq, heads in ((1, 1), (1, 128), (2, 128), (3, 171)):
            for capacity in (1, 512, 513, 8193, 16384):
                edges = [n for n in (0, 1, 511, 512, 513, 8193, 16384) if n <= capacity]
                patterns = [[value] * batch for value in edges]
                patterns += [[edges[b % len(edges)] for b in range(batch)]]
                patterns += [[capacity if b % 2 else 0 for b in range(batch)]]
                patterns += [patterns[-1][::-1]]
                # Deduplicate the one-batch cases without dropping ordered mixed lengths.
                for lengths in dict.fromkeys(map(tuple, patterns)):
                    cases.append(
                        length_case(
                            lengths,
                            batch=batch,
                            sq=sq,
                            heads=heads,
                            sk=capacity,
                            aic=aic,
                            ratio=ratio,
                        )
                    )
    compare(length_scheduler, length_oracle, cases, length_artifacts, f"boundary-{aic}-{ratio}")


@pytest.mark.parametrize("ratio", (1, 2))
def test_equal_and_unequal_m_adjustment(length_scheduler, length_oracle, length_artifacts, ratio):
    cases = [
        length_case(lengths, batch=10, sq=2, heads=m, sk=16384, ratio=ratio)
        for m in range(128, 257)
        for lengths in ([511] * 10, [511] * 9 + [512])
    ]
    result = compare(
        length_scheduler, length_oracle, cases, length_artifacts, f"adjusted-m-{ratio}"
    )
    assert [r[1] for r in result[::2]] == list(range(128, 257))
    assert all(r[1] == 256 for r in result[1::2])


def test_multi_query_tail_and_reuse(length_scheduler, length_oracle, length_artifacts):
    cases = [
        length_case(lengths, batch=len(lengths), sq=sq, heads=heads, sk=16384)
        for sq, heads in ((257, 1), (7, 73), (17, 257), (129, 128))
        for lengths in ([0, 1, 16384], [8193, 0, 511], [0, 0, 0], [513, 512, 0], [0, 0, 1])
    ]
    result = compare(
        length_scheduler, length_oracle, cases + cases[::-1], length_artifacts, "reuse"
    )
    assert result[: len(cases)] == result[len(cases) :][::-1]


@pytest.mark.parametrize("ratio", (1, 2))
def test_trailing_zero_cost_samples(length_scheduler, length_oracle, length_artifacts, ratio):
    # Regression: the source exhausts compute cost at batch 15, leaving the last
    # empty sample to InitOutputSingleCore. A dense-style final-end==B guard
    # wrongly rejected this valid num_splits=1 mapping.
    lengths = [0, 1, 511, 512, 513] * 3 + [0]
    shape = length_case(lengths, batch=16, sq=1, heads=1, sk=513, aic=15, ratio=ratio)
    result = compare(
        length_scheduler, length_oracle, [shape], length_artifacts, f"trailing-zero-{ratio}"
    )[0]
    assert result[18 + result[3] - 1] == 15
    assert result[44 + result[3] - 1] == 0


def test_full_capacity_and_global_empty(length_scheduler, length_artifacts):
    cases = [
        length_case(lengths, batch=b, sq=sq, heads=heads, sk=sk, aic=aic, ratio=ratio)
        for b, (sq, heads), sk, aic, ratio in itertools.product(
            (0, 1, 3, 10, 21),
            ((0, 128), (1, 1), (2, 128), (3, 171)),
            (0, 1, 512, 513, 8193, 16384),
            (1, 20, 26),
            (1, 2),
        )
        for lengths in (None, [sk] * b)
    ]
    # An empty query still validates supplied short KV lengths, then uses the
    # existing global-empty validation schedule rather than calling SplitCore.
    cases += [length_case([0, 1, 513], batch=3, sq=0, sk=513)]
    actual = evaluate(length_scheduler, cases)
    assert actual == evaluate(length_scheduler, cases, dense=True)
    assert all(row[0] == 0 for row in actual)
    (length_artifacts / "dense-equivalence.json").write_text(
        json.dumps(dict(cases=len(cases), all_fields_equal=True)) + "\n"
    )


@pytest.mark.parametrize(
    "lengths,count,shape,status",
    [
        (None, 1, {}, 1),
        ([], 0, {}, 1),
        ([0], 0, {}, 1),
        ([0], 2, {}, 1),
        ([0], 2**64 - 1, {}, 1),
        ([0, 0], 2, {}, 1),
        ([-1], None, {}, 1),
        ([-(2**31)], None, {}, 1),
        ([513], None, {}, 1),
        ([2**31 - 1], None, {}, 1),
        ([0, 1, -1], None, {"batch": 3}, 1),
        ([0, 1, 513], None, {"batch": 3}, 1),
        ([0, 1, 513], None, {"batch": 3, "sq": 0}, 1),
        ([1], None, {"sk": 0}, 1),
        ([0], None, {"batch": 0}, 1),
        (None, 1, {"batch": 0}, 1),
        ([0], None, {"batch": -1}, 1),
        ([0], None, {"sq": -1}, 1),
        ([0], None, {"heads": 0}, 1),
        ([0], None, {"sk": 16385}, 1),
        ([0], None, {"batch": 2**32}, 2),
        ([0], None, {"sq": 2**63 - 1}, 2),
        ([0], None, {"heads": 2**23}, 2),
        ([0], None, {"sq": 2**30 + 1, "heads": 1}, 2),
        ([0], None, {"sk": 2**32}, 2),
        ([0], None, {"batch": 2**24, "sq": 2**30, "heads": 1}, 2),
        ([1] * 11, None, {"batch": 11, "sq": 2**30, "heads": 3, "sk": 16384, "aic": 1}, 2),
        ([0], None, {"aic": 0, "aiv": 0}, 3),
        ([0], None, {"aic": 27, "aiv": 54}, 3),
        ([0], None, {"aiv": 21}, 3),
        ([0], None, {"block_dim": 21}, 3),
        ([0], None, {"libapi": 2**64 - 1}, 5),
    ],
)
def test_invalid_lengths_and_fields(length_scheduler, lengths, count, shape, status):
    # Driver byte-compares a sentinel-filled output on every failure.
    assert evaluate(length_scheduler, [length_case(lengths, count=count, **shape)]) == [[status]]


@pytest.mark.parametrize("ratio", (1, 2))
def test_platform_workspace_boundaries(length_scheduler, length_oracle, length_artifacts, ratio):
    cases = [length_case([0, 513, 8193], batch=3, sk=16384, ratio=ratio, block_dim=20)]
    result = compare(length_scheduler, length_oracle, cases, length_artifacts, f"platform-{ratio}")[
        0
    ]
    assert result[16] == 20
    budget = result[13]
    limit = length_case([0, 513, 8193], batch=3, sk=16384, ratio=ratio, libapi=2**64 - 1 - budget)
    assert evaluate(length_scheduler, [limit])[0][15] == 2**64 - 1
    limit[11] += 1
    assert evaluate(length_scheduler, [limit]) == [[5]]
    for index in range(6, 11):
        bad = cases[0].copy()
        bad[index] -= 1
        assert evaluate(length_scheduler, [bad]) == [[4]]
    # Used cores must all be launched, even when the table entry is otherwise valid.
    assert evaluate(length_scheduler, [length_case([511] * 20, batch=20, block_dim=1)]) == [[3]]
