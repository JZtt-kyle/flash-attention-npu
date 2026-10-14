"""Native FD ownership, FA3 explicit slices and independent workspace/address checks.

This module owns CPU split-KV acceptance; there is no public split behavior yet.
Every oracle executes pinned source, while coverage/addresses are derived from
raw cases. No numerical reference is needed or recomputed by these CPU tests.
"""

import contextlib
import itertools
import os
from pathlib import Path
import random

import pytest
import torch

from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_split_kv_build import compile_drivers


@pytest.fixture(scope="session")
def drivers(tmp_path_factory):
    target = os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("split-kv")
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CpuDriver(p)) for p in compile_drivers(directory)]


def norm(config):
    q, k, left, right, causal = config
    left = -1 if left >= k - 1 else left
    right = -1 if right >= q - 1 else right
    if causal and not (q == 1 and left == right == -1):
        right = 0
    mask = 3 if left < 0 and right == 0 else 4 if left >= 0 or right >= 0 else 0
    return (
        mask,
        (2**31 - 1 if mask == 3 else k - 1 if left < 0 else left),
        (0 if mask == 3 else q - 1 if right < 0 else right),
    )


def geometry(q, k, h, m, n, block, config):
    mask, left, right = norm(config)
    rows = range(block * m, min((block + 1) * m, q * h))
    spans = []
    for row in rows:
        i = row // h
        lo = max(0, i + k - q - left) if mask == 4 else 0
        hi = min(k, i + k - q + right + 1) if mask else k
        if lo < hi:
            spans.append((lo, hi))
    lo = min((a for a, _ in spans), default=0)
    hi = max((b for _, b in spans), default=0)
    return lo // n, (hi + n - 1) // n, lo, hi, len(rows)


def cost(rows, keys):
    return 6 * ((rows + 15) // 16) + 10 * ((keys + 63) // 64)


def native_arrays(got):
    return [got[8 + 26 * i : 8 + 26 * (i + 1)] for i in range(9)] + [got[242:294], got[294:346]]


def check_native(got, seqs, h, m, n, config):
    assert got[0] == 0
    used, maximum, fd_count, max_splits, vecs, total, tiles = got[1:8]
    arrays = native_arrays(got)
    b_end, m_end, n_end, fd_b, fd_m, fd_s, starts, fd_rows, fd_tail, vec_head, vec_row = arrays
    tasks = []
    for b, (q, k) in enumerate(seqs):
        for block in range((q * h + m - 1) // m):
            start, end, _, _, rows = geometry(q, k, h, m, n, block, config)
            tasks.extend(((b, block, j), cost(rows, min(n, k - j * n))) for j in range(start, end))
    assert (len(tasks), sum(c for _, c in tasks)) == (tiles, total)
    owners, groups, cursor = {}, [], 0
    for core in range(used):
        end = (b_end[core], m_end[core], n_end[core])
        assigned = []
        while cursor < len(tasks) and tasks[cursor][0] < end:
            key, c = tasks[cursor]
            owners.setdefault(key[:2], set()).add(core)
            assigned.append((key, c))
            cursor += 1
        groups.append(assigned)
    assert cursor == len(tasks), (got, tasks[cursor:])
    assert max((sum(c for _, c in group) for group in groups), default=0) == maximum
    reductions = sorted(key for key, cores in owners.items() if len(cores) > 1)
    assert list(zip(fd_b[:fd_count], fd_m[:fd_count])) == reductions
    assert max_splits == max((len(owners[key]) for key in reductions), default=0)
    for i, key in enumerate(reductions):
        b, block = key
        rows = min(m, seqs[b][0] * h - block * m)
        assert (fd_s[i], fd_rows[i], fd_tail[i]) == (
            len(owners[key]),
            (rows + 7) // 8,
            (rows - 1) % 8 + 1,
        )
    for core, group in enumerate(groups):
        if group:
            key = group[0][0][:2]
            assert starts[core] == sum(owner < core for owner in owners[key])
    merge_tasks = [(i, row) for i in range(fd_count) for row in range(fd_rows[i])]
    ends = [merge_tasks.index((vec_head[i], vec_row[i])) + 1 for i in range(vecs)]
    assert (not merge_tasks and not ends) or (
        ends[-1] == len(merge_tasks) and ends == sorted(set(ends))
    )
    assert got[-2:] == [sum((q * h + m - 1) // m for q, _ in seqs), 8]


@pytest.mark.parametrize("cores,ratio", [(1, 1), (2, 2), (7, 1), (20, 2), (26, 1), (26, 2)])
def test_native_fd_source_and_independent_ownership(drivers, cores, ratio):
    cases = [
        ([(1, 16384)], 128, 128, 512),  # many FD owners of one M block
        ([(3, 513), (0, 17), (2, 448), (4, 0)], 129, 256, 112),
        ([(5, 1), (1, 0)], 5, 4, 3),  # leading masked rows and a zero-cost suffix
        ([(0, 513), (0, 0)], 8, 128, 512),
        ([(3, 0), (1, 0)], 8, 128, 512),
        ([], 8, 128, 512),
        ([(17, 8193), (7, 4097), (2, 1)], 8, 128, 448),
    ]
    observed_fd = False
    for seqs, h, m, n in cases:
        qmax = max((q for q, _ in seqs), default=0)
        kmax = max((k for _, k in seqs), default=0)
        for left, right, causal in [
            (-1, -1, 0),
            (-1, -1, 1),
            (2, 1, 0),
            (448, -1, 1),
            (-1, 2, 0),
            (1, -1, 0),
        ]:
            config = [qmax, kmax, left, right, causal]
            flat = list(itertools.chain.from_iterable(seqs))
            got = drivers[0].evaluate([[0, *config, h, m, n, cores, ratio, len(seqs), *flat]])[0]
            source = drivers[1].evaluate(
                [[1, *norm(config), h, m, n, cores, ratio, len(seqs), *flat]]
            )[0]
            padded = []
            offset = 7
            for width in [cores] * 9 + [cores * ratio] * 2:
                capacity = 26 if len(padded) < 9 * 26 else 52
                padded += source[offset : offset + width] + [0] * (capacity - width)
                offset += width
            assert got[:-2] == [0, *source[:7], *padded]
            check_native(got, seqs, h, m, n, config)
            observed_fd |= got[3] > 0
    assert observed_fd == (cores > 1)


@pytest.mark.parametrize("splits", [1, 2, 8, 256])
def test_explicit_split_coverage_masks_and_empty_slices(drivers, splits):
    cases = [
        (1, 1, 1, 128, 512),
        (3, 17, 5, 4, 3),
        (9, 3, 8, 128, 112),
        (1, 16384, 128, 128, 512),
        (17, 513, 129, 256, 448),
        (0, 17, 8, 128, 512),
        (3, 0, 8, 128, 512),
        (0, 0, 8, 128, 512),
    ]
    for q, k, h, m, n in cases:
        for left, right, causal in [(-1, -1, 0), (-1, -1, 1), (1, 2, 0), (2, -1, 1), (-1, 1, 0)]:
            config = [q, k, left, right, causal]
            for block in range(max(1, (q * h + m - 1) // m)):
                begin, end, lo, hi, rows = geometry(q, k, h, m, n, block, config)
                actual = drivers[0].evaluate(
                    [[1, *config, q, k, h, m, n, block, splits, s] for s in range(splits)]
                )
                native = drivers[2].evaluate([[1, begin, end, splits, s] for s in range(splits)])
                visited, sum_cost = [], 0
                for result, (first, last) in zip(actual, native):
                    assert result[0] == 0
                    first = min(first, last)  # FA3 empty tails may have min>max; canonicalize.
                    keys = (max(lo, first * n), min(hi, last * n)) if first < last else (hi, hi)
                    expected_cost = sum(cost(rows, min(n, k - j * n)) for j in range(first, last))
                    assert result[1:6] == [first, last, *keys, expected_cost]
                    assert result[6:10] == [begin, end, lo, hi]
                    visited.extend(range(first, last))
                    sum_cost += result[5]
                assert visited == list(range(begin, end))
                assert sum_cost == actual[0][-1]


def test_fa3_request_semantics_and_limits(drivers):
    cases = [[r, a] for r, a in itertools.product([-(2**63), -1, 0, 1, 2, 8, 256, 257], [1, 7, 26])]
    for (r, a), got, source in zip(
        cases,
        drivers[0].evaluate([[2, *c] for c in cases]),
        drivers[2].evaluate([[2, *c] for c in cases]),
    ):
        assert got == ([6, 987] if r > 256 else [0, a if r <= 0 else r])
        assert source == ([1] if r > 256 else got)
    bad = [[2, 2**32 + 1, 1], [2, 2**63 - 1, 1], [2, 0, 0], [2, 0, 257]]
    assert drivers[0].evaluate(bad) == [[6, 987]] * len(bad)
    assert drivers[0].evaluate([[2, 1, -(2**63)]]) == [[0, 1]]


@pytest.mark.parametrize(
    "m,n,rows",
    [(128, 112, 0), (128, 448, 1), (128, 512, 129), (256, 1, 257), (256, 512, 2**32 - 256)],
)
def test_source_workspace_and_nonoverlapping_segments(drivers, m, n, rows):
    for cores, block, fd in [(1, 1, 0), (1, 1, 1), (20, 7, 1), (26, 26, 1), (26, 1, 0)]:
        case = [m, n, rows, cores, block, fd, 16777216]
        got = drivers[0].evaluate([[3, *case]])[0]
        source = drivers[2].evaluate([[3, *case]])[0]
        assert got[0] == 0
        assert [*got[1:5], got[-1]] == source
        mm1, mm2, accum, sums = got[1:5]
        sizes = [2 * block * mm1 * 4, 2 * block * mm1 * 2, 2 * block * mm2 * 4, 2 * block * mm2 * 4]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        assert got[5:9] == offsets[:4]
        if fd:
            assert got[9:13] == [
                offsets[-1],
                offsets[-1] + accum * 4,
                offsets[-1] + accum * 4 + sums * 2,
                offsets[-1] + (accum + sums) * 4,
            ]
        else:
            assert got[9:13] == [0, 0, 0, offsets[-1]]
        assert all(offset % 32 == 0 for offset in got[5:13])
        assert got[12] <= got[13] and got[13] + got[14] == got[15]
        edge = [*case[:-1], 2**63 - 1 - got[13]]
        assert drivers[0].evaluate([[3, *edge]])[0][-1] == 2**63 - 1
        edge[-1] += 1
        assert drivers[0].evaluate([[3, *edge]]) == [[5]]


@pytest.mark.parametrize(
    "layout,b,q,t,h,s",
    [
        (0, 2, 3, 6, 5, 2),
        (0, 0, 7, 0, 8, 256),
        (0, 2, 0, 0, 8, 8),
        (0, 3, 9, 27, 128, 1),
        (1, 3, 7, 17, 5, 8),
        (1, 0, 0, 0, 8, 2),
        (1, 3, 7, 0, 8, 256),
        (1, 3, 7, 17, 8, 1),
        (1, 2, 2**30, 2**31 - 1, 1, 2),
    ],
)
def test_accumulator_capacity_shapes_strides_and_offsets(drivers, layout, b, q, t, h, s):
    case = [layout, b, q, t, h, s]
    got = drivers[0].evaluate([[4, *case]])[0]
    source = drivers[2].evaluate([[4, *case]])[0]
    assert got[0] == 0
    shapes = [] if s == 1 else ([s, h, t, 512] if layout else [s, b, h, q, 512])
    out_shape = shapes
    lse_shape = shapes[:-1]
    assert source == [len(out_shape), *out_shape, len(lse_shape), *lse_shape]
    for start, shape in [(1, out_shape), (14, lse_shape)]:
        if not shape:
            assert got[start : start + 13] == [0] * 13
            continue
        tensor = torch.empty(shape, dtype=torch.float32, device="meta")
        assert got[start : start + 13] == [
            len(shape),
            *shape,
            *([0] * (5 - len(shape))),
            *tensor.stride(),
            *([0] * (5 - len(shape))),
            tensor.numel(),
            tensor.numel() * 4,
        ]
    if s > 1 and t:
        queries = t if layout else q
        positions = [[0, 0, 0, 0], [s - 1, 0 if layout else b - 1, h - 1, queries - 1]]
        if queries > 4:
            positions.append([1, 0, 1 if h > 1 else 0, 4])  # allocation gaps remain addressable
        for split, batch, head, query in positions:
            index = ((split * (1 if layout else b) + batch) * h + head) * queries + query
            assert drivers[0].evaluate([[5, *case, split, batch, head, query]]) == [
                [0, index * 512 * 4, index * 4]
            ]


def test_validation_overflow_and_failure_atomicity(drivers):
    invalid = [
        [0, 2, 3, 6, 5, 0],
        [0, 2, 3, 6, 5, 257],
        [2, 2, 3, 6, 5, 2],
        [0, -1, 3, 6, 5, 2],
        [0, 2, -1, 6, 5, 2],
        [0, 2, 3, -1, 5, 2],
        [0, 2, 3, 6, 0, 2],
        [0, 2, 3, 5, 5, 2],
        [0, 2**32, 0, 0, 1, 2],
        [1, 2, 3, 2**31, 1, 2],
        [1, 2, 2**30 + 1, 3, 1, 2],
        [1, 2, 3, 6, (2**32 - 1) // 512 + 1, 2],
        [0, 2**32 - 1, 2**30, (2**32 - 1) * 2**30, 1, 256],
        [1, 2, 2**30, 2**31 - 1, (2**32 - 1) // 512, 256],
    ]
    assert all(g[0] != 0 for g in drivers[0].evaluate([[4, *c] for c in invalid]))
    base = [0, 2, 3, 6, 5, 2]
    coords = [
        (-1, 0, 0, 0),
        (2, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 2, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 5, 0),
        (0, 0, 0, -1),
        (0, 0, 0, 3),
    ]
    assert all(g == [1, 987, 987] for g in drivers[0].evaluate([[5, *base, *c] for c in coords]))
    for base in ([0, 2, 3, 6, 5, 1], [1, 2, 3, 6, 5, 2]):
        assert drivers[0].evaluate([[5, *base, 0, 1, 0, 0]]) == [[1, 987, 987]]
    workspace = [128, 512, 129, 20, 7, 1, 0]
    for field, value in [
        (0, 0),
        (0, 257),
        (1, 0),
        (1, 513),
        (2, -1),
        (2, 2**32),
        (3, 0),
        (3, 27),
        (4, 0),
        (4, 21),
        (6, 2**64 - 1),
    ]:
        case = workspace.copy()
        case[field] = value
        assert drivers[0].evaluate([[3, *case]])[0][0] != 0
    block = [1, 3, 513, -1, -1, 0, 3, 513, 128, 128, 512, 0, 2, 0]
    for field, value in [
        (1, -1),
        (2, 16385),
        (3, -(2**63)),
        (6, -1),
        (7, 514),
        (8, 0),
        (9, 0),
        (10, 513),
        (11, 3),
        (12, 0),
        (12, 257),
        (13, -1),
        (13, 2),
    ]:
        case = block.copy()
        case[field] = value
        assert drivers[0].evaluate([case])[0][0] != 0
    native = [0, 3, 513, -1, -1, 0, 128, 128, 512, 20, 2, 1, 3, 513]
    for field, value in [(6, 0), (7, 0), (8, 513), (9, 0), (9, 27), (10, 3), (12, -1), (13, 514)]:
        case = native.copy()
        case[field] = value
        assert drivers[0].evaluate([case])[0][0] != 0
    # Reject source uint32 tile accumulation before executing SplitCore.
    assert drivers[0].evaluate([[0, 2**30, 16384, -1, -1, 0, 1, 1, 1, 20, 2, 1, 2**30, 16384]]) == [
        [2]
    ]


def test_random_partition_coverage(drivers):
    rng = random.Random(941)
    for _ in range(40):
        q, k, h = rng.randint(1, 33), rng.randint(0, 1025), rng.choice([1, 5, 129])
        m, n = rng.choice([128, 256]), rng.choice([112, 448, 512])
        config = [q, k, rng.choice([-1, 0, 7]), rng.choice([-1, 0, 9]), rng.randrange(2)]
        seqs = [(q, k), (0, k), (q // 2, k // 3), (1, 0)]
        case = [0, *config, h, m, n, 20, 2, len(seqs), *itertools.chain.from_iterable(seqs)]
        check_native(drivers[0].evaluate([case])[0], seqs, h, m, n, config)
