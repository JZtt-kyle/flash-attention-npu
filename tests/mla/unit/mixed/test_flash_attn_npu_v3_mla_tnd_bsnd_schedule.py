"""Independent CPU checks for compact TND Q with a physical BSND KV cache.

The source executable uses pinned CANN functions and numerical int64 Q tails /
noncumulative KV lengths, never the migrated parser. No mixed metadata, device
attention or public entry is enabled. All tests also support full ASan/UBSan.
"""

import itertools
import json
import os
from pathlib import Path
import random

import pytest

from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    compile_driver as compile_bsnd_driver,
    evaluate as evaluate_bsnd,
    length_case,
)
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_schedule import case
from tests.mla.unit.tnd.test_flash_attn_npu_v3_mla_tnd_schedule import (
    build_oracle,
    check_tasks,
    compile_driver,
    evaluate,
    expected_spans as compact_spans,
    tnd_case,
)


@pytest.fixture(scope="session")
def mixed_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_TND_BSND_SCHEDULE_ARTIFACT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-tnd-bsnd-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def mixed_scheduler(mixed_artifacts):
    return compile_driver(mixed_artifacts, False, mixed=True)


@pytest.fixture(scope="session")
def mixed_oracle(mixed_artifacts):
    return build_oracle(mixed_artifacts, mixed=True)


@pytest.fixture(scope="session")
def bsnd_scheduler(mixed_artifacts):
    directory = mixed_artifacts / "bsnd"
    directory.mkdir(exist_ok=True)
    return compile_bsnd_driver(directory, False)


@pytest.fixture(scope="session")
def compact_scheduler(mixed_artifacts):
    directory = mixed_artifacts / "compact"
    directory.mkdir(exist_ok=True)
    return compile_driver(directory, False)


def mixed_case(
    qlens=(3, 0, 1),
    klens=(17, 0, 33),
    *,
    capacity=64,
    cq=None,
    qcount=None,
    kcount=None,
    has_q=True,
    has_k=True,
    total_q=None,
    **shape,
):
    cq = list(itertools.accumulate(qlens, initial=0)) if cq is None else list(cq)
    defaults = dict(batch=len(qlens), sq=max(qlens, default=0), sk=capacity)
    defaults.update(shape)
    # The common harness's totalKv field is unused in the mixed API: physical
    # cache capacity is B*Sk, with no int32 cumulative KV array or total bound.
    return case(**defaults) + [
        int(has_k),
        len(klens) if kcount is None else kcount,
        len(klens),
        *klens,
        cq[-1] if total_q is None else total_q,
        0,
        int(has_q),
        len(cq) if qcount is None else qcount,
        len(cq),
        *cq,
    ]


def input_lengths(item):
    kv = item[16 : 16 + item[15]]
    cq = item[21 + item[15] :]
    return cq, [b - a for a, b in zip(cq, cq[1:])], kv


def expected_spans(item):
    cq, q, kv = input_lengths(item)
    return [
        max(q, default=0),
        max(kv, default=0),
        *itertools.chain.from_iterable((cq[b], b * item[3], q[b], kv[b]) for b in range(item[0])),
    ]


def compare(scheduler, oracle, cases, directory, name):
    (directory / f"{name}-inputs.json").write_text(json.dumps(cases) + "\n")
    actual, mapped, native = (
        evaluate(scheduler, cases),
        evaluate(oracle, cases),
        evaluate(oracle, cases, native=True),
    )
    (directory / f"{name}-outputs.json").write_text(
        json.dumps(dict(actual=actual, mapped=mapped, native=native)) + "\n"
    )
    counts = dict(native_non_fd=0, fa3_no_split_mapping=0, omitted_suffix=0)
    for item, got, want, source in zip(cases, actual, mapped, native):
        assert got == want, (item, got, want)
        assert got[0] == 0 and len(got) == 358 + item[0] * 4
        assert got[356:] == expected_spans(item)
        counts["native_non_fd" if source[5] == 0 else "fa3_no_split_mapping"] += 1
        if source[5] == 0:
            assert got == source
        # Reuse scalar task/cost enumeration: mathematical lengths are shared
        # across layouts. Physical mixed starts were independently checked above.
        _, q, kv = input_lengths(item)
        compact = tnd_case(q, kv, heads=item[2])
        check_tasks(compact, [*got[:356], *compact_spans(compact)])
        counts["omitted_suffix"] += got[18 + got[3] - 1] < item[0]
    (directory / f"{name}.json").write_text(
        json.dumps(dict(cases=len(cases), **counts, all_fields_tasks_spans_equal=True), indent=2)
        + "\n"
    )
    return actual


@pytest.mark.parametrize("aic,ratio", itertools.product(range(1, 27), (1, 2)))
def test_mixed_source_boundaries(mixed_scheduler, mixed_oracle, mixed_artifacts, aic, ratio):
    cases = []
    for batch, heads, (qpattern, kpattern, capacity) in itertools.product(
        sorted({1, max(1, aic - 1), aic, aic + 1}),
        (1, 128, 171),
        (
            ([3, 0, 1], [17, 0, 33], 64),
            ([0, 1, 3], [513, 0, 8193], 16384),
            ([257, 1, 0], [16384, 512, 0], 16384),
            ([2], [511], 512),
            ([2, 0], [0, 513], 1024),
            ([1, 3, 7], [1, 513, 0], 8193),
            ([1], [16384], 16384),
            ([3, 0, 1], [0], 513),
        ),
    ):
        q = [qpattern[b % len(qpattern)] for b in range(batch)]
        k = [kpattern[b % len(kpattern)] for b in range(batch)]
        if sum(q):
            cases.append(mixed_case(q, k, capacity=capacity, heads=heads, aic=aic, ratio=ratio))
    compare(mixed_scheduler, mixed_oracle, cases, mixed_artifacts, f"boundary-{aic}-{ratio}")


def test_random_reuse_and_q_upper_bound(mixed_scheduler, mixed_oracle, mixed_artifacts):
    rng = random.Random(60622)
    cases = []
    for _ in range(200):
        batch = rng.randrange(1, 35)
        capacity = rng.choice((1, 31, 511, 512, 513, 8193, 16384))
        q = [rng.choice((0, 1, 2, 3, 7, 17, 257)) for _ in range(batch)]
        kv = [rng.choice((0, 1, capacity // 2, capacity)) for _ in range(batch)]
        if sum(q):
            cases.append(
                mixed_case(
                    q,
                    kv,
                    capacity=capacity,
                    heads=rng.choice((1, 8, 73, 128, 257)),
                    sq=512,
                    aic=rng.randrange(1, 27),
                    ratio=rng.choice((1, 2)),
                )
            )
    got = compare(mixed_scheduler, mixed_oracle, cases + cases[::-1], mixed_artifacts, "reuse")
    assert got[: len(cases)] == got[len(cases) :][::-1]
    exact = [c.copy() for c in cases]
    for c in exact:
        c[1] = max(input_lengths(c)[1])
    assert evaluate(mixed_scheduler, exact) == got[: len(cases)]


def test_capacity_and_layout_semantics(
    mixed_scheduler, mixed_oracle, mixed_artifacts, compact_scheduler
):
    cases = [
        mixed_case(),
        mixed_case(klens=[33, 0, 17]),
        mixed_case(capacity=1024),
        mixed_case([2] * 10, [511] * 10, capacity=511),
        mixed_case([2] * 10, [511] + [0] * 9, capacity=511),
    ]
    got = compare(mixed_scheduler, mixed_oracle, cases, mixed_artifacts, "capacity-layout")
    assert got[0][356:] == [3, 33, 0, 0, 3, 17, 3, 64, 0, 0, 3, 128, 1, 33]
    assert got[0][359::4] == got[1][359::4] == [0, 64, 128]
    assert got[2][359::4] == [0, 1024, 2048]
    assert [g[2] for g in got[:3]] == [64, 64, 512]  # Capacity, not observed max=33.
    assert [g[1] for g in got[3:]] == [128, 256]  # Noncumulative KV equality.
    compact = evaluate(
        compact_scheduler,
        [tnd_case([2] * 10, [511] * 10), tnd_case([2] * 10, [511] + [0] * 9)],
    )
    assert [g[1] for g in compact] == [256, 128]


@pytest.mark.parametrize("ratio", (1, 2))
def test_m_base_thresholds(mixed_scheduler, mixed_oracle, mixed_artifacts, ratio):
    cases = [
        mixed_case([2] * 10, kv, capacity=1024, heads=heads, ratio=ratio)
        for heads in range(127, 258)
        for kv in ([511] * 10, [511] + [0] * 9)
    ]
    got = compare(mixed_scheduler, mixed_oracle, cases, mixed_artifacts, f"m-base-{ratio}")
    assert [g[1] for g in got[::2]] == [256, *range(128, 257), 256]
    assert all(g[1] == 256 for g in got[1::2])


def test_bsnd_equivalence(mixed_scheduler, bsnd_scheduler, mixed_artifacts):
    pairs = []
    for b, sq, heads, capacity, aic, ratio in itertools.product(
        (1, 3, 10, 20), (1, 2, 17), (1, 73, 128), (1, 513, 16384), (1, 20), (1, 2)
    ):
        for kv in ([capacity] * b, [capacity if i % 2 == 0 else 0 for i in range(b)], [0] * b):
            pairs.append(
                (
                    mixed_case([sq] * b, kv, capacity=capacity, heads=heads, aic=aic, ratio=ratio),
                    length_case(kv, batch=b, sq=sq, heads=heads, sk=capacity, aic=aic, ratio=ratio),
                )
            )
    mixed = evaluate(mixed_scheduler, [a for a, _ in pairs])
    bsnd = evaluate_bsnd(bsnd_scheduler, [b for _, b in pairs])
    for (item, _), got, want in zip(pairs, mixed, bsnd):
        assert got[:356] == want
        assert got[356:] == expected_spans(item)
    (mixed_artifacts / "bsnd-equivalence.json").write_text(
        json.dumps(dict(cases=len(pairs), all_fields_equal=True)) + "\n"
    )


@pytest.mark.parametrize(
    "q,kv,capacity",
    [([], [], 0), ([], [], 16384), ([0, 0], [1, 513], 1024), ([3, 0, 1], [0, 0, 0], 0)],
)
def test_global_empty_validates_and_keeps_physical_spans(mixed_scheduler, q, kv, capacity):
    item = mixed_case(q, kv, capacity=capacity, sq=17)
    got = evaluate(mixed_scheduler, [item])[0]
    assert got[:18] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert got[18:356] == [0] * 338 and got[356:] == expected_spans(item)
    assert evaluate(mixed_scheduler, [item], parse=True) == [[0, *expected_spans(item)]]


def test_zero_lengths_are_not_empty_capacity(mixed_scheduler, mixed_oracle, mixed_artifacts):
    item = mixed_case([3, 0, 1], [0, 0, 0], capacity=513)
    got = compare(mixed_scheduler, mixed_oracle, [item], mixed_artifacts, "zero-cost")[0]
    assert got[1:5] == [256, 512, 1, 0]
    assert got[13] > 0  # Nonempty physical capacity retains workspace.
    assert evaluate(mixed_scheduler, [mixed_case([], [], has_k=False)])[0][0] == 0


@pytest.mark.parametrize(
    "kwargs,status",
    [
        ({"has_q": False}, 1),
        ({"has_k": False}, 1),
        ({"qcount": 0}, 1),
        ({"qcount": 2**64 - 1}, 1),
        ({"kcount": 0}, 1),
        ({"kcount": 4}, 1),
        ({"kcount": 2**64 - 1}, 1),
        ({"cq": [1, 3, 3, 4]}, 1),
        ({"cq": [-1, 3, 3, 4]}, 1),
        ({"cq": [0, 3, 2, 4]}, 1),
        ({"cq": [0, 3, -(2**31), 4]}, 1),
        ({"klens": [-1, 0, 33]}, 1),
        ({"klens": [17, -(2**31), 33]}, 1),
        ({"klens": [17, 0, 65]}, 1),
        ({"klens": [17, 0, 2**31 - 1]}, 1),
        ({"total_q": 3}, 1),
        ({"total_q": 5}, 1),
        ({"sq": 2}, 1),
        ({"capacity": 32}, 1),
        ({"heads": 0}, 1),
        ({"batch": -1}, 1),
        ({"total_q": -1}, 1),
        ({"sq": -1}, 1),
        ({"capacity": -1}, 1),
        ({"capacity": 16385}, 1),
        ({"total_q": 2**31}, 2),
        ({"batch": 2**32}, 2),
        ({"heads": 2**23}, 2),
        ({"sq": 2**30 + 1}, 2),
        ({"sq": 2**30, "heads": 4}, 2),
        ({"total_q": 2**63 - 1}, 2),
        ({"capacity": 2**32}, 2),
        ({"aic": 0, "aiv": 0}, 3),
        ({"aic": 27}, 3),
        ({"aiv": 21}, 3),
        ({"ratio": 3}, 3),
        ({"block_dim": 21}, 3),
        ({"libapi": 2**64 - 1}, 5),
        ({"qlens": [], "klens": [], "has_q": False}, 1),
        ({"qlens": [], "klens": [], "qcount": 0}, 1),
        ({"qlens": [], "klens": [], "total_q": 1}, 1),
        ({"qlens": [], "klens": [], "kcount": 1}, 1),
        ({"qlens": [0, 0, 0], "klens": [0, 0, 65]}, 1),
        ({"qlens": [3, 0, 1], "klens": [0, 0, 0], "capacity": 0, "cq": [0, 3, 2, 4]}, 1),
        ({"batch": 2**32 - 1, "qcount": 0, "kcount": 0}, 1),
    ],
)
def test_rejections_leave_output_unchanged(mixed_scheduler, kwargs, status):
    item = mixed_case(**kwargs)
    assert evaluate(mixed_scheduler, [item]) == [[status]]
    if status in (1, 2):
        assert evaluate(mixed_scheduler, [item], parse=True) == [[status]]


def test_large_physical_starts_and_query_bound(mixed_scheduler, mixed_artifacts):
    # No tensor allocation. A physical BSND token start legitimately exceeds
    # uint32, unlike the compact int32 KV prefix protocol.
    b = 2**18 + 2
    item = mixed_case([0] * b, [0] * b, capacity=16384)
    parsed = evaluate(mixed_scheduler, [item], parse=True)[0]
    scheduled = evaluate(mixed_scheduler, [item])[0]
    assert parsed[0] == 0 and parsed[1:] == expected_spans(item) == scheduled[356:]
    assert parsed[-3] == (b - 1) * 16384 > 2**32
    large_q = mixed_case([2**30, 2**30 - 1], [0, 0], capacity=0, heads=1)
    assert evaluate(mixed_scheduler, [large_q])[0][356:] == expected_spans(large_q)
    (mixed_artifacts / "large-addresses.json").write_text(
        json.dumps(dict(batch=b, last_kv_start=parsed[-3], total_query=2**31 - 1)) + "\n"
    )


def test_tile_address_and_workspace_overflow(mixed_scheduler, mixed_oracle, mixed_artifacts):
    address = mixed_case([2**29] * 3, [0] * 3, heads=8388607, sq=1)
    assert evaluate(mixed_scheduler, [address]) == [[2]]
    tiles = mixed_case([2**27] * 15 + [2**27 - 1], [16384] * 16, capacity=16384, heads=16)
    assert evaluate(mixed_scheduler, [tiles], parse=True) == [[0, *expected_spans(tiles)]]
    assert evaluate(mixed_scheduler, [tiles]) == [[2]]
    for ratio in (1, 2):
        item = mixed_case(ratio=ratio, block_dim=20)
        got = compare(mixed_scheduler, mixed_oracle, [item], mixed_artifacts, f"workspace-{ratio}")[
            0
        ]
        limit = item.copy()
        limit[11] = 2**64 - 1 - got[13]
        assert evaluate(mixed_scheduler, [limit])[0][15] == 2**64 - 1
        limit[11] += 1
        assert evaluate(mixed_scheduler, [limit]) == [[5]]
        for index in range(6, 11):
            bad = item.copy()
            bad[index] -= 1
            assert evaluate(mixed_scheduler, [bad]) == [[4]]
    bad = mixed_case([1] * 20, [513] * 20, capacity=513, block_dim=1)
    assert evaluate(mixed_scheduler, [bad]) == [[3]]
