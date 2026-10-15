"""Compact TND CPU parser/schedule, independently checked against pinned CANN.

No public/device TND entry is enabled. MLA_TND_SCHEDULE_ARTIFACT_DIR retains
inputs, outputs, source blobs and manifests; MLA_SCHEDULE_SANITIZE=1 instruments
both drivers. The source executable never includes/links the migrated parser.
"""

import itertools
import json
import random

import pytest

from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    compile_driver as compile_bsnd_driver,
    evaluate as evaluate_bsnd,
)
from tests.mla.support.components.bsnd_schedule import length_case

from tests.mla.support.components.tnd_schedule import (
    PARSER as PARSER,
    build_oracle as build_oracle,
    check_tasks as check_tasks,
    compile_driver as compile_driver,
    evaluate as evaluate,
    expected_spans as expected_spans,
    prefixes as prefixes,
    tnd_artifacts as tnd_artifacts,
    tnd_case as tnd_case,
    tnd_oracle as tnd_oracle,
)


@pytest.fixture(scope="session")
def tnd_scheduler(tnd_artifacts):
    return compile_driver(tnd_artifacts, False)


@pytest.fixture(scope="session")
def bsnd_scheduler(tnd_artifacts):
    directory = tnd_artifacts / "bsnd"
    directory.mkdir(exist_ok=True)
    return compile_bsnd_driver(directory, False)


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
