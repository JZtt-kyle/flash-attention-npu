"""Independent FA3 normalization, CANN band candidates and row/task ownership."""

import contextlib
import itertools
import os
from pathlib import Path
import random

import pytest

from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_window_build import compile_drivers
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_schedule import (
    cost,
    evaluate,
    save,
    schedule_case,
)


@pytest.fixture(scope="session")
def window_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR")
    return Path(target) if target else tmp_path_factory.mktemp("mla-window")


@pytest.fixture(scope="session")
def drivers(window_artifacts):
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CpuDriver(p)) for p in compile_drivers(window_artifacts)]


def normalized(config, *, forward=False):
    q, k, left, right, causal = config
    if left >= k - 1:
        left = -1
    if right >= q - 1:
        right = -1
    if q == 1 and left == right == -1:
        causal = False
    if causal:
        right = 0
    if forward:
        left, right = ((v + 2**31) % 2**32 - 2**31 for v in (left, right))
    kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
    return [kind, k - 1 if left < 0 else left, q - 1 if right < 0 else right]


@pytest.mark.parametrize(
    "q,k", [(0, 0), (0, 7), (7, 0), (1, 7), (7, 1), (3, 9), (9, 3), (2**30, 16384)]
)
def test_pinned_fa3_normalization(drivers, window_artifacts, q, k):
    windows = sorted({-(2**63), -(2**31), -2, -1, 0, 1, q - 2, q - 1, k - 2, k - 1, 2**63 - 1})
    configs = [
        [q, k, left, right, causal]
        for left, right, causal in itertools.product(windows, windows, (0, 1))
    ]
    got = evaluate(drivers[0], [[3, *c] for c in configs])
    source = evaluate(drivers[3], configs)
    save(window_artifacts, f"normalize-{q}-{k}", configs=configs, actual=got, source=source)
    for config, actual, pinned in zip(configs, got, source):
        expected = normalized(config)
        overflow = config[2] < -(2**31) or (config[3] < -(2**31) and not config[4])
        assert actual == ([2] if overflow else [0, *expected])
        assert pinned == normalized(config, forward=True) + expected
        if not overflow:
            assert pinned[:3] == pinned[3:]


def block_oracle(item, config):
    q, k, heads, m, n, index = item
    kind, left, right = normalized(config)
    pre = 2**31 - 1 if kind == 1 else q - k + left if kind == 2 else left
    nxt = k - q if kind == 1 else k - q + right if kind == 2 else right
    rows = list(range(index * m, min((index + 1) * m, q * heads)))
    intervals = []
    for row in rows:
        i = row // heads
        if k <= 32:
            keys = [
                j for j in range(k) if kind == 0 or (j <= i + nxt and (kind == 1 or j >= i - pre))
            ]
            intervals.append((keys[0], keys[-1] + 1) if keys else (0, 0))
        else:
            lo = 0 if kind != 2 else max(0, i - pre)
            hi = k if kind == 0 else min(k, i + nxt + 1)
            intervals.append((lo, hi) if lo < hi else (0, 0))
    nonempty = [(a, b) for a, b in intervals if b > a]
    begin = min((a for a, _ in nonempty), default=0)
    end = max((b for _, b in nonempty), default=0)
    start_block, end_block = begin // n, (end + n - 1) // n
    count = end_block - start_block
    normal = cost(len(rows), n) if count else 0
    last = cost(len(rows), min(n, k - (end_block - 1) * n)) if count else 0
    empty = sum(a == b for a, b in intervals)
    assert all(a == b for a, b in intervals[:empty])
    return [
        0,
        pre,
        nxt,
        index * m,
        index * m + len(rows),
        rows[0] // heads if rows else 0,
        rows[-1] // heads + 1 if rows else 0,
        begin,
        end,
        start_block,
        end_block,
        empty,
        (count - 1) * normal + last if count else 0,
        normal,
        last,
    ]


def source_case(item, config):
    mode, *fields = item
    kind, left, right = normalized(config)
    return [
        mode,
        (0, 3, 4)[kind],
        2**31 - 1 if kind == 1 else left,
        0 if kind == 1 else right,
        *fields,
    ]


def compare_blocks(drivers, cases, configs, directory, name):
    got = evaluate(drivers[0], [[0, *c, *item] for item, c in zip(cases, configs)])
    source = evaluate(
        drivers[1],
        [
            source_case([0, h, m, n, 1, 2, 1, q, k, b], c)
            for (q, k, h, m, n, b), c in zip(cases, configs)
        ],
    )
    save(directory, name, inputs=cases, configs=configs, actual=got, source=source)
    for item, config, actual, pinned in zip(cases, configs, got, source):
        assert actual == block_oracle(item, config), (item, config, actual)
        assert [actual[i] for i in (1, 2, 9, 10, 12, 13, 14)] == pinned, (
            item,
            config,
            actual,
            pinned,
        )


@pytest.mark.parametrize(
    "heads,m,n", [(1, 1, 1), (5, 4, 3), (129, 128, 112), (257, 256, 448), (8, 256, 512)]
)
def test_exhaustive_small_geometry(drivers, window_artifacts, heads, m, n):
    cases, configs = [], []
    for q, k, slack, causal, (left, right) in itertools.product(
        range(5),
        range(5),
        (0, 2),
        (0, 1),
        [(-1, -1), (-1, 0), (0, -1), (0, 0), (1, 2), (2, 1), (-2, 2**63 - 1)],
    ):
        for b in range(max(1, (q * heads + m - 1) // m)):
            cases.append((q, k, heads, m, n, b))
            configs.append([q + slack, k + slack, left, right, causal])
    compare_blocks(drivers, cases, configs, window_artifacts, f"small-{heads}-{m}-{n}")


def test_tile_kv_and_wide_boundaries(drivers, window_artifacts):
    cases, configs = [], []
    shapes = list(
        itertools.product(
            (1, 513),
            (0, 1, 447, 448, 449, 511, 512, 513, 16384),
            (1, 129),
            (128, 256),
            (112, 448, 512),
        )
    )
    shapes += [(2**30, 16384, 1, 256, 512), (512, 0, (2**32 - 1) // 512, 256, 512)]
    for q, k, h, m, n in shapes:
        count = (q * h + m - 1) // m
        for left, right, causal in [
            (0, 0, 0),
            (1, 1, 0),
            (112, -1, 0),
            (-1, 448, 0),
            (448, 512, 1),
            (-(2**31), 2**63 - 1, 0),
        ]:
            edge = max(0, q - k - max(0, right)) * h // m
            for b in sorted({0, count - 1, min(edge, count - 1), max(0, min(edge - 1, count - 1))}):
                cases.append((q, k, h, m, n, b))
                configs.append([q, k, left, right, causal])
    compare_blocks(drivers, cases, configs, window_artifacts, "boundaries")


def check_schedule(item, config, got, native):
    _, heads, m, n, cores, _, batch, *lengths = item
    sequences = list(zip(lengths[::2], lengths[1::2]))
    tasks, prefix = [], [0]
    for b, (q, k) in enumerate(sequences):
        tasks += [
            (b, block_oracle((q, k, heads, m, n, index), config))
            for index in range((q * heads + m - 1) // m)
        ]
        prefix.append(len(tasks))
    endpoints = []
    for c in range(native[0]):
        b, row, s2 = native[7 + c], native[7 + cores + c], native[7 + 2 * cores + c]
        end = prefix[b] + row + bool(s2)
        if c == native[0] - 1:
            end = len(tasks)
        if (not endpoints or end > endpoints[-1]) and (end or not tasks):
            endpoints.append(end)
    fields = [[0] * 26 for _ in range(4)]
    previous, visited = 0, []
    for c, end in enumerate(endpoints):
        b = next((b for b in range(batch) if prefix[b + 1] > end), batch)
        fields[0][c], fields[1][c] = b, end - prefix[b]
        assigned = tasks[previous:end]
        fields[3][c] = sum(block[12] for _, block in assigned)
        for sample, block in assigned:
            visited.extend((sample, row) for row in range(block[3], block[4]))
        previous = end
    expected = [
        0,
        len(endpoints),
        max(fields[3]),
        sum(t[1][12] for t in tasks),
        len(tasks),
        sum(t[1][10] - t[1][9] for t in tasks),
        *itertools.chain.from_iterable(fields),
    ]
    assert got[:110] == expected, (item, config, got, expected, native)
    assert native[5:7] == [expected[3], expected[5]]
    assert visited == [(b, row) for b, (q, _) in enumerate(sequences) for row in range(q * heads)]
    assert len(set(visited)) == len(visited)
    assert got[6 + got[1] - 1] == batch
    return bool(native[2])


def compare_schedules(drivers, items, configs, directory, name, raw=None):
    raw = items if raw is None else raw
    got = evaluate(drivers[0], [[item[0], *c, *item[1:]] for item, c in zip(raw, configs)])
    native_inputs = [source_case(item, c) for item, c in zip(items, configs)]
    source = evaluate(drivers[1], native_inputs)
    migrated = evaluate(drivers[2], native_inputs)
    save(
        directory,
        name,
        inputs=items,
        configs=configs,
        raw=raw,
        actual=got,
        source=source,
        migrated_source=migrated,
    )
    assert migrated == source  # Includes every native FD field, even for num_splits=1.
    flags = [check_schedule(item, c, a, s) for item, c, a, s in zip(items, configs, got, source)]
    return got, flags


@pytest.mark.parametrize("cores,ratio", itertools.product(range(1, 27), (1, 2)))
def test_complete_source_partitions(drivers, window_artifacts, cores, ratio):
    items, configs = [], []
    for seq, (m, n), (left, right, causal) in itertools.product(
        [
            ([], []),
            ([0], [0]),
            ([7], [0]),
            ([1], [16384]),
            ([9], [2]),
            ([2], [9]),
            ([65], [65]),
            ([0, 7, 3, 1, 0], [9, 1, 0, 3, 0]),
            ([1, 3, 7, 0, 9], [0, 513, 2, 7, 0]),
        ],
        [(128, 112), (256, 512)],
        [(0, 0, 0), (1, -1, 0), (-1, 2, 0), (4, 3, 1), (-1, -1, 0), (-1, -1, 1)],
    ):
        q, k = seq
        items.append(schedule_case(q, k, cores=cores, ratio=ratio, m=m, n=n))
        configs.append([max(q, default=0), max(k, default=0), left, right, causal])
    compare_schedules(drivers, items, configs, window_artifacts, f"partition-{cores}-{ratio}")


def test_random_candidates_and_repeated_calls(drivers, window_artifacts):
    rng = random.Random(90901)
    items, configs = [], []
    for _ in range(150):
        q = [rng.randrange(19) for _ in range(rng.randrange(1, 9))]
        k = [rng.choice([0, 1, 3, 17, 513, 16384]) for _ in q]
        items.append(
            schedule_case(
                q,
                k,
                heads=rng.choice([1, 3, 127, 257]),
                m=rng.choice([7, 128, 256]),
                n=rng.choice([16, 112, 448, 512]),
                cores=rng.randrange(1, 27),
                ratio=rng.choice([1, 2]),
            )
        )
        configs.append(
            [
                max(q) + 2,
                max(k),
                rng.choice([-1, 0, 1, 7, 112]),
                rng.choice([-1, 0, 1, 7, 448]),
                rng.randrange(2),
            ]
        )
    got, flags = compare_schedules(
        drivers, items + items[::-1], configs + configs[::-1], window_artifacts, "random-reuse"
    )
    assert got[: len(items)] == got[len(items) :][::-1]
    assert any(flags) and not all(flags)


@pytest.mark.parametrize("layout,page", [(0, 0), (1, 0), (2, 0), (0, 112), (2, 528)])
def test_raw_allocation_and_used_lengths(drivers, window_artifacts, layout, page):
    raw, items, configs, expected_spans = [], [], [], []
    for used, (left, right, causal) in itertools.product(
        (False, True), [(0, 0, 0), (-1, 1, 0), (2, -1, 0), (3, 2, 1)]
    ):
        q = [9] * 4 if layout == 0 else [7, 0, 13, 3]
        k = [page * 3 if page else 17] * 4 if layout != 1 else [17, 3, 0, 9]
        uq = [2, 0, 9, 0] if used else q
        uk = [min(k[0], 1), 0, min(k[2], 3), 0] if used else k
        raw.append(
            [
                2,
                129,
                128,
                112,
                20,
                2,
                4,
                layout,
                page,
                *q,
                *k,
                int(used),
                *(uq if used else []),
                int(used or bool(page)),
                *(uk if used or page else []),
            ]
        )
        items.append(schedule_case(uq, uk, m=128, n=112))
        configs.append([max(q), max(k), left, right, causal])
        qs, ks = list(itertools.accumulate(q, initial=0)), list(itertools.accumulate(k, initial=0))
        expected_spans.append(
            list(
                itertools.chain.from_iterable(
                    (qs[b], q[b], uq[b], uk[b])
                    if page
                    else (qs[b], ks[b], q[b], k[b], uq[b], uk[b])
                    for b in range(4)
                )
            )
        )
    got, _ = compare_schedules(
        drivers, items, configs, window_artifacts, f"raw-{layout}-{page}", raw=raw
    )
    for actual, spans in zip(got, expected_spans):
        assert actual[110:] == spans
    invalid = raw[-1].copy()
    invalid[10] = -1
    assert evaluate(drivers[0], [[2, *configs[-1], *invalid[1:]]])[0] == [101]


def test_rejections_preserve_output(drivers, window_artifacts):
    config = [9, 17, 1, 2, 0]
    configs = [
        [-1, 17, 1, 2, 0],
        [9, -1, 1, 2, 0],
        [2**63 - 1, 17, 1, 2, 0],
        [9, 2**63 - 1, 1, 2, 0],
        [9, 16385, 1, 2, 0],
    ]
    assert evaluate(drivers[0], [[3, *c] for c in configs]) == [[1], [1], [2], [2], [1]]
    good = [7, 3, 129, 256, 512, 0]
    cases, expected = [], []
    for field, value, status in [
        (0, -1, 1),
        (1, -1, 1),
        (2, 0, 1),
        (1, 16385, 1),
        (0, 2**63 - 1, 2),
        (2, 2**63 - 1, 2),
        (0, 2**30 + 1, 2),
        (2, 2**32 // 512, 2),
        (3, 0, 3),
        (3, 257, 3),
        (4, 513, 3),
        (4, -1, 3),
        (5, -1, 4),
        (5, 4, 4),
        (5, 2**63 - 1, 4),
        (0, 10, 1),
        (1, 18, 1),
    ]:
        item = good.copy()
        item[field] = value
        cases.append([0, *config, *item])
        expected.append([status])
    got = evaluate(drivers[0], cases)
    assert got == expected
    for left, right, causal in [(-(2**63), 0, 1), (-1, -(2**63), 0)]:
        bad_config = [9, 17, left, right, causal]
        assert evaluate(drivers[0], [[0, *bad_config, *good]]) == [[2]]
        assert evaluate(drivers[0], [[1, *bad_config, *schedule_case()[1:]]]) == [[2]]
    items = []
    for field, value in [(1, 0), (2, 0), (3, 513), (4, 0), (4, 27), (5, 0), (5, 3)]:
        item = schedule_case()
        item[field] = value
        items.append([1, *config, *item[1:]])
    items += [
        [1, 2**30, 16384, 0, 0, 0, *schedule_case([2**30], [16384], heads=1, m=1, n=1)[1:]],
        [
            1,
            131072,
            16384,
            1,
            0,
            0,
            *schedule_case([131072] * 2, [16384] * 2, heads=1, m=1, n=1)[1:],
        ],
        [1, *config, *schedule_case([7, -1], [3, 0])[1:]],
        [1, *config, *schedule_case([7, 0], [3, 2**63 - 1])[1:]],
    ]
    result = evaluate(drivers[0], items)
    assert result == [[1], [3], [3], [5], [5], [5], [5], [2], [2], [1], [2]]
    save(
        window_artifacts,
        "rejections",
        blocks=cases,
        results=got,
        schedules=items,
        schedule_results=result,
    )
