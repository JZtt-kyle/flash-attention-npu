"""Compact TND CPU parser/schedule, independently checked against pinned CANN.

No public/device TND entry is enabled. MLA_TND_SCHEDULE_ARTIFACT_DIR retains
inputs, outputs, source blobs and manifests; MLA_SCHEDULE_SANITIZE=1 instruments
both drivers. The source executable never includes/links the migrated parser.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess

import pytest

from tests.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    compile_driver as compile_bsnd_driver,
    evaluate as evaluate_bsnd,
    length_case,
)
from tests.test_flash_attn_npu_v3_mla_schedule import (
    CANN_COMMIT,
    FIA,
    FUNCTIONS,
    MLA,
    ROOT,
    SOURCE_FILES,
    case,
    run,
    source_function,
)

PARSER = "attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp"


@pytest.fixture(scope="session")
def tnd_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_TND_SCHEDULE_ARTIFACT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-tnd-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def compile_driver(
    directory, oracle, *, mixed=False, used=False, paged=False, migrated_split=False
):
    name = "oracle" if oracle else "schedule"
    if mixed:
        name += "-mixed"
    if used:
        name += "-used"
    if paged:
        name += "-paged"
    if migrated_split:
        name += "-migrated-split"
    binary = directory / name
    command = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O2",
        "-g",
        "-D_GLIBCXX_ASSERTIONS",
        "-DBSND_KV_LENGTHS",
        "-DTND_LENGTHS",
    ]
    if mixed:
        command += ["-DTND_BSND_KV"]
    if used:
        command += ["-DUSED_LENGTHS"]
    if paged:
        command += ["-DPAGED_KV"]
    if os.environ.get("MLA_SCHEDULE_SANITIZE") == "1":
        command += [
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
        ]
    if oracle:
        command += ["-DSOURCE_ORACLE", "-I" + str(directory)]
        if migrated_split:
            command += [
                "-DMLA_MIGRATED_SPLIT",
                "-I" + str(MLA),
                str(MLA / "detail/cann_split_core.cpp"),
            ]
        else:
            command += [str(directory / "split_core.cpp")]
    else:
        command += ["-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(MLA)]
        command += [
            str(MLA / f)
            for f in ("mla_tnd_schedule.cpp", "mla_schedule.cpp", "detail/cann_split_core.cpp")
        ]
    command += [str(ROOT / "tests/mla_schedule_driver.cpp"), "-o", str(binary)]
    (directory / f"{name}-compile.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / f"{name}-compile.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.fixture(scope="session")
def tnd_scheduler(tnd_artifacts):
    return compile_driver(tnd_artifacts, False)


@pytest.fixture(scope="session")
def bsnd_scheduler(tnd_artifacts):
    directory = tnd_artifacts / "bsnd"
    directory.mkdir(exist_ok=True)
    return compile_bsnd_driver(directory, False)


@pytest.fixture(scope="session")
def tnd_oracle(tnd_artifacts):
    return build_oracle(tnd_artifacts)


def build_oracle(tnd_artifacts, *, mixed=False, used=False, paged=False):
    source_root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    blobs = {
        name: subprocess.check_output(
            ["git", "-C", str(source_root), "show", CANN_COMMIT + ":" + name], timeout=30
        )
        for name in [*SOURCE_FILES, PARSER]
    }
    for name, blob in blobs.items():
        (tnd_artifacts / Path(name).name).write_bytes(blob)
    functions = [*FUNCTIONS, "DealSameSeqEachBatch", "CreateSplitInput"]
    source = blobs[FIA + "fia_tiling_nonquant_mla.cpp"].decode()
    bodies = ["template <typename T>\n" + source_function(source, "inline auto Align(")]
    bodies += [source_function(source, "FiaTilingNonQuantMla::" + f + "(") for f in functions]
    (tnd_artifacts / "source_functions.inc").write_text("\n".join(bodies))
    parser_functions = ["ProcessKVActualSeqLoop"]
    if paged:
        parser_functions += ["GetS2SizeForPageAttention"]
    (tnd_artifacts / "source_parser.inc").write_text(
        "\n".join(
            source_function(blobs[PARSER].decode(), "FiaInfoParser::" + name + "(")
            for name in parser_functions
        )
    )
    (tnd_artifacts / "log").mkdir(exist_ok=True)
    (tnd_artifacts / "log/log.h").write_text(
        "#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n"
    )
    manifest = dict(
        commit=CANN_COMMIT,
        kv_layout="per input" if used else "BSND" if mixed else "TND",
        used_adapter=used,
        paged_adapter=paged,
        functions=functions,
        parser_functions=parser_functions,
        split_core_functions=["GetS1SeqSize", "GetS2SeqSize", "SplitCore"],
        files={name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
    )
    (tnd_artifacts / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return compile_driver(tnd_artifacts, True, mixed=mixed, used=used, paged=paged)


def tnd_case(
    qlens=(3, 0, 1),
    klens=(17, 0, 33),
    *,
    cq=None,
    ck=None,
    qcount=None,
    kcount=None,
    has_q=True,
    has_k=True,
    total_q=None,
    total_k=None,
    **shape,
):
    cq = list(itertools.accumulate(qlens, initial=0)) if cq is None else list(cq)
    ck = list(itertools.accumulate(klens, initial=0)) if ck is None else list(ck)
    defaults = dict(batch=len(qlens), sq=max(qlens, default=0), sk=max(klens, default=0))
    defaults.update(shape)
    return case(**defaults) + [
        int(has_k),
        len(ck) if kcount is None else kcount,
        len(ck),
        *ck,
        cq[-1] if total_q is None else total_q,
        ck[-1] if total_k is None else total_k,
        int(has_q),
        len(cq) if qcount is None else qcount,
        len(cq),
        *cq,
    ]


def prefixes(item):
    ck = item[16 : 16 + item[15]]
    start = 16 + item[15]
    cq = item[start + 5 :]
    return cq, ck


def evaluate(binary, cases, *, native=False, parse=False):
    env = os.environ.copy()
    for name, enabled in (("MLA_ORACLE_AUTOMATIC_FD", native), ("MLA_TND_PARSE_ONLY", parse)):
        env.pop(name, None)
        if enabled:
            env[name] = "1"
    inputs = "".join(" ".join(map(str, item)) + "\n" for item in cases)
    result = run([str(binary)], input=inputs, env=env)
    assert not result.stderr, result.stderr
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(outputs) == len(cases)
    return outputs


def expected_spans(item):
    cq, ck = prefixes(item)
    spans = [(q, k, qend - q, kend - k) for q, k, qend, kend in zip(cq, ck, cq[1:], ck[1:])]
    return [
        max((s[2] for s in spans), default=0),
        max((s[3] for s in spans), default=0),
        *itertools.chain.from_iterable(spans),
    ]


def check_tasks(item, result):
    # Independent scalar enumeration, with no SplitCore/parser helpers.
    cq, ck = prefixes(item)
    mbase, nbase, cores = result[1:4]
    coords, costs = [], []
    for b in range(item[0]):
        rows = (cq[b + 1] - cq[b]) * item[2]
        kv = ck[b + 1] - ck[b]
        for m, start in enumerate(range(0, rows, mbase)):
            coords.append((b, m))
            costs.append(
                sum(
                    6 * ((min(mbase, rows - start) + 15) // 16)
                    + 10 * ((min(nbase, kv - k) + 63) // 64)
                    for k in range(0, kv, nbase)
                )
            )
    ends = [(result[18 + i], result[44 + i]) for i in range(cores)]
    assert all(a < b for a, b in zip([(0, 0), *ends[:-1]], ends))
    assert ends[-1] <= (item[0], 0)
    assert result[5:8] == [0, 0, 0] and result[70:96] == [0] * 26
    begin, loads, visited = (0, 0), [], []
    for end in ends:
        selected = [i for i, coord in enumerate(coords) if begin <= coord < end]
        visited += selected
        loads.append(sum(costs[i] for i in selected))
        begin = end
    assert len(visited) == len(set(visited))
    assert {i for i in visited if costs[i]} == {i for i, cost in enumerate(costs) if cost}
    assert result[4] == max(loads)
    assert result[356:] == expected_spans(item)


def compare(scheduler, oracle, cases, directory, name):
    (directory / f"{name}-inputs.json").write_text(json.dumps(cases) + "\n")
    actual, mapped, native = (
        evaluate(scheduler, cases),
        evaluate(oracle, cases),
        evaluate(oracle, cases, native=True),
    )
    # Keep complete outputs even for failed comparisons.
    (directory / f"{name}-outputs.json").write_text(
        json.dumps(dict(actual=actual, mapped=mapped, native=native)) + "\n"
    )
    counts = dict(native_non_fd=0, fa3_no_split_mapping=0, omitted_suffix=0)
    for item, got, want, source in zip(cases, actual, mapped, native):
        assert got == want, (item, got, want)
        assert got[0] == 0 and len(got) == 358 + item[0] * 4
        counts["native_non_fd" if source[5] == 0 else "fa3_no_split_mapping"] += 1
        if source[5] == 0:
            assert got == source
        check_tasks(item, got)
        counts["omitted_suffix"] += got[18 + got[3] - 1] < item[0]
    (directory / f"{name}.json").write_text(
        json.dumps(
            dict(
                cases=len(cases),
                **counts,
                all_fields_equal=True,
                independent_tasks_costs_spans=True,
            ),
            indent=2,
        )
        + "\n"
    )
    return actual


@pytest.mark.parametrize("aic,ratio", itertools.product(range(1, 27), (1, 2)))
def test_tnd_source_boundaries(tnd_scheduler, tnd_oracle, tnd_artifacts, aic, ratio):
    cases = []
    for batch in sorted({1, max(1, aic - 1), aic, aic + 1}):
        for heads in (1, 128, 171):
            for qpattern, kpattern in (
                ([3, 0, 1], [17, 0, 33]),
                ([0, 1, 3], [513, 0, 8193]),
                ([257, 1, 0], [16384, 512, 0]),
                ([2], [511]),
                ([2, 0], [0, 513]),
                ([1, 3, 7], [1, 513, 0]),
                ([1], [16384]),
            ):
                q = [qpattern[b % len(qpattern)] for b in range(batch)]
                k = [kpattern[b % len(kpattern)] for b in range(batch)]
                if sum(q) and sum(k):
                    cases.append(tnd_case(q, k, heads=heads, aic=aic, ratio=ratio))
    compare(tnd_scheduler, tnd_oracle, cases, tnd_artifacts, f"boundary-{aic}-{ratio}")


def test_random_tnd_reuse_and_upper_bounds(tnd_scheduler, tnd_oracle, tnd_artifacts):
    rng = random.Random(60601)
    cases = []
    for _ in range(200):
        b = rng.randrange(1, 35)
        q = [rng.choice((0, 1, 2, 3, 7, 17, 257)) for _ in range(b)]
        k = [rng.choice((0, 1, 31, 512, 513, 8193, 16384)) for _ in range(b)]
        if sum(q) and sum(k):
            cases.append(
                tnd_case(
                    q,
                    k,
                    heads=rng.choice((1, 8, 73, 128, 257)),
                    sq=512,
                    sk=16384,
                    aic=rng.randrange(1, 27),
                    ratio=rng.choice((1, 2)),
                )
            )
    outputs = compare(tnd_scheduler, tnd_oracle, cases + cases[::-1], tnd_artifacts, "random-reuse")
    assert outputs[: len(cases)] == outputs[len(cases) :][::-1]
    exact = []
    for item in cases:
        item = item.copy()
        item[1], item[3] = expected_spans(item)[:2]
        exact.append(item)
    assert evaluate(tnd_scheduler, exact) == outputs[: len(cases)]


def test_t01_and_source_cumulative_tail_equality(
    tnd_scheduler, tnd_oracle, tnd_artifacts, bsnd_scheduler
):
    cases = [tnd_case(), tnd_case([2] * 10, [511] * 10), tnd_case([2] * 10, [511] + [0] * 9)]
    got = compare(tnd_scheduler, tnd_oracle, cases, tnd_artifacts, "t01-and-tail-equality")
    assert got[0][356:] == [3, 33, 0, 0, 3, 17, 3, 17, 0, 0, 3, 17, 1, 33]
    assert [row[1] for row in got[1:]] == [256, 128]
    bsnd = evaluate_bsnd(bsnd_scheduler, [length_case([511] * 10, batch=10, sq=2, sk=511)])[0]
    assert bsnd[1] == 128  # Preserve the pinned source's layout-specific tiling difference.


def test_bsnd_equivalence(tnd_scheduler, bsnd_scheduler, tnd_artifacts):
    pairs = []
    for b, sq, h, kv, aic, ratio in itertools.product(
        (1, 3, 20), (1, 3, 17), (1, 73), (1, 513, 16384), (1, 20), (1, 2)
    ):
        klens = [kv if i % 2 == 0 else 0 for i in range(b)]
        pairs.append(
            (
                tnd_case([sq] * b, klens, heads=h, aic=aic, ratio=ratio),
                length_case(klens, batch=b, sq=sq, heads=h, sk=kv, aic=aic, ratio=ratio),
            )
        )
    tnd = evaluate(tnd_scheduler, [x for x, _ in pairs])
    bsnd = evaluate_bsnd(bsnd_scheduler, [x for _, x in pairs])
    for (item, _), a, b in zip(pairs, tnd, bsnd):
        assert a[:356] == b
        check_tasks(item, a)
    (tnd_artifacts / "bsnd-equivalence.json").write_text(
        json.dumps(dict(cases=len(pairs), all_fields_equal=True)) + "\n"
    )


@pytest.mark.parametrize("ratio", (1, 2))
def test_m_base_thresholds(tnd_scheduler, tnd_oracle, tnd_artifacts, ratio):
    cases = [
        tnd_case([2] * 10, kv, heads=heads, ratio=ratio)
        for heads in range(127, 258)
        for kv in ([511] + [0] * 9, [511] * 10)
    ]
    got = compare(tnd_scheduler, tnd_oracle, cases, tnd_artifacts, f"m-base-{ratio}")
    assert [x[1] for x in got[::2]] == [256, *range(128, 257), 256]
    assert all(x[1] == 256 for x in got[1::2])


@pytest.mark.parametrize(
    "qlens,klens",
    [([], []), ([0], [0]), ([0, 0], [0, 513]), ([3, 0, 1], [0, 0, 0]), ([0, 0, 0], [0, 0, 0])],
)
def test_global_empty_validates_and_keeps_spans(tnd_scheduler, qlens, klens):
    item = tnd_case(qlens, klens, sq=17, sk=16384)
    got = evaluate(tnd_scheduler, [item])[0]
    assert got[:18] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert got[18:356] == [0] * 338
    assert got[356:] == expected_spans(item)
    assert evaluate(tnd_scheduler, [item], parse=True) == [[0, *expected_spans(item)]]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(qlens=[], klens=[], has_q=False),
        dict(qlens=[], klens=[], qcount=0),
        dict(qlens=[], klens=[], ck=[1]),
        dict(qlens=[], klens=[], total_q=1),
        dict(qlens=[0, 0], klens=[1, 2], ck=[0, 1, 0]),
        dict(qlens=[3, 0, 1], klens=[0, 0, 0], cq=[0, 3, 2, 4]),
        dict(qlens=[0, 0], klens=[0, 0], kcount=2),
        dict(batch=2**32 - 1, qcount=0, kcount=0),
    ],
)
def test_empty_still_rejects_bad_prefixes(tnd_scheduler, kwargs):
    item = tnd_case(**kwargs)
    assert evaluate(tnd_scheduler, [item]) == [[1]]
    assert evaluate(tnd_scheduler, [item], parse=True) == [[1]]


@pytest.mark.parametrize(
    "kwargs,status",
    [
        ({"has_q": False}, 1),
        ({"has_k": False}, 1),
        ({"qcount": 0}, 1),
        ({"kcount": 3}, 1),
        ({"qcount": 2**64 - 1}, 1),
        ({"kcount": 2**64 - 1}, 1),
        ({"cq": [1, 3, 3, 4]}, 1),
        ({"ck": [-1, 17, 17, 50]}, 1),
        ({"cq": [0, 3, 2, 4]}, 1),
        ({"ck": [0, 17, 16, 50]}, 1),
        ({"cq": [0, 3, -(2**31), 4]}, 1),
        ({"ck": [0, 17, 17, -1]}, 1),
        ({"total_q": 3}, 1),
        ({"total_q": 5}, 1),
        ({"total_k": 49}, 1),
        ({"total_k": 51}, 1),
        ({"sq": 2}, 1),
        ({"sk": 32}, 1),
        ({"heads": 0}, 1),
        ({"batch": -1}, 1),
        ({"total_q": -1}, 1),
        ({"total_k": -1}, 1),
        ({"sq": -1}, 1),
        ({"sk": -1}, 1),
        ({"sk": 16385}, 1),
        ({"total_q": 2**31}, 2),
        ({"total_k": 2**31}, 2),
        ({"batch": 2**32}, 2),
        ({"heads": 2**23}, 2),
        ({"sq": 2**30 + 1}, 2),
        ({"sq": 2**30, "heads": 4}, 2),
        ({"total_q": 2**63 - 1}, 2),
        ({"total_k": 2**63 - 1}, 2),
        ({"sk": 2**32}, 2),
        ({"aic": 0, "aiv": 0}, 3),
        ({"aic": 27}, 3),
        ({"aiv": 21}, 3),
        ({"ratio": 3}, 3),
        ({"block_dim": 21}, 3),
        ({"libapi": 2**64 - 1}, 5),
    ],
)
def test_rejections_leave_output_unchanged(tnd_scheduler, kwargs, status):
    item = tnd_case(**kwargs)
    assert evaluate(tnd_scheduler, [item]) == [[status]]
    if status in (1, 2):
        assert evaluate(tnd_scheduler, [item], parse=True) == [[status]]


def test_capacity_field_and_workspace_edges(tnd_scheduler, tnd_oracle, tnd_artifacts):
    # Prefix totals can reach INT32_MAX without allocating tensor storage.
    large = tnd_case([2**30, 2**30 - 1], [0, 0], heads=1)
    assert evaluate(tnd_scheduler, [large], parse=True) == [[0, *expected_spans(large)]]
    assert evaluate(tnd_scheduler, [large])[0][356:] == expected_spans(large)
    # Many empty samples keep their starts after a large first query.
    compact = tnd_case([2**29] + [0] * 19999, [0] * 20000, heads=7)
    assert evaluate(tnd_scheduler, [compact])[0][0] == 0
    overflow = tnd_case([2**29] * 3, [0] * 3, heads=8388607, sq=1)
    assert evaluate(tnd_scheduler, [overflow]) == [[2]]
    # Sum of physical tiles exceeds uint32 before entering the source loops.
    # Each segment fits uint32 S1G, but 16 * 2**28 physical tiles does not.
    tiles = tnd_case([2**27] * 15 + [2**27 - 1], [16384] * 16, heads=16)
    assert evaluate(tnd_scheduler, [tiles], parse=True) == [[0, *expected_spans(tiles)]]
    assert evaluate(tnd_scheduler, [tiles]) == [[2]]
    for ratio in (1, 2):
        item = tnd_case(ratio=ratio, block_dim=20)
        got = compare(tnd_scheduler, tnd_oracle, [item], tnd_artifacts, f"workspace-{ratio}")[0]
        limit = item.copy()
        limit[11] = 2**64 - 1 - got[13]
        assert evaluate(tnd_scheduler, [limit])[0][15] == 2**64 - 1
        limit[11] += 1
        assert evaluate(tnd_scheduler, [limit]) == [[5]]
        for index in range(6, 11):
            bad = item.copy()
            bad[index] -= 1
            assert evaluate(tnd_scheduler, [bad]) == [[4]]
    assert evaluate(tnd_scheduler, [tnd_case([1] * 20, [513] * 20, block_dim=1)]) == [[3]]
