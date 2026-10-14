"""Causal CPU geometry, fixed-source candidates, whole-M ownership and raw lengths.

The source executable links verbatim pinned CANN blobs only. Python independently
enumerates visible keys/row blocks and maps the source candidates to whole rows;
no migrated geometry, length parser or adapter is used to construct the oracle.
Both normal and ASan/UBSan suites execute this complete collection.
"""

import contextlib
import itertools
import json
import os
from pathlib import Path
import random

import pytest

from tests.mla.support.mla_causal_build import CpuDriver, compile_drivers


@pytest.fixture(scope="session")
def causal_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR")
    return Path(target) if target else tmp_path_factory.mktemp("mla-causal")


@pytest.fixture(scope="session")
def drivers(causal_artifacts):
    with contextlib.ExitStack() as stack:
        yield [
            stack.enter_context(CpuDriver(binary)) for binary in compile_drivers(causal_artifacts)
        ]


def evaluate(binary, cases):
    values = binary.evaluate(cases)
    assert len(values) == len(cases)
    return values


def save(directory, name, **data):
    (directory / (name + ".json")).write_text(json.dumps(data) + "\n")


def cost(rows, keys):
    return 6 * ((rows + 15) // 16) + 10 * ((keys + 63) // 64)


def block_oracle(q, k, heads, m, n, index):
    rows = list(range(index * m, min((index + 1) * m, q * heads)))
    counts = []
    for row in rows:
        query = row // heads
        # Explicit key enumeration for small cases; scalar row bounds for long KV.
        if k <= 32:
            counts.append(sum(key <= query + k - q for key in range(k)))
        else:
            counts.append(max(0, min(k, query + k - q + 1)))
    end = max(counts, default=0)
    blocks = (end + n - 1) // n
    normal = cost(len(rows), n) if blocks else 0
    last = cost(len(rows), min(n, k - (blocks - 1) * n)) if blocks else 0
    return [
        0,
        k - q,
        index * m,
        index * m + len(rows),
        rows[0] // heads if rows else 0,
        rows[-1] // heads + 1 if rows else 0,
        0,
        end,
        0,
        blocks,
        counts.count(0),
        (blocks - 1) * normal + last if blocks else 0,
        normal,
        last,
    ]


def compare_blocks(drivers, cases, directory, name):
    actual = evaluate(drivers[0], [[0, *item] for item in cases])
    source = evaluate(drivers[1], [[0, h, m, n, 1, 2, 1, q, k, b] for q, k, h, m, n, b in cases])
    save(directory, name, inputs=cases, actual=actual, source=source)
    for item, got, pinned in zip(cases, actual, source):
        assert got == block_oracle(*item), (item, got)
        assert [got[i] for i in (1, 8, 9, 11, 12, 13)] == pinned, (item, got, pinned)


@pytest.mark.parametrize(
    "heads,m,n",
    [
        (1, 1, 1),
        (3, 7, 16),
        (5, 4, 3),
        (127, 128, 64),
        (129, 256, 112),
        (257, 128, 448),
        (8, 256, 512),
    ],
)
def test_exhaustive_small_geometry(drivers, causal_artifacts, heads, m, n):
    cases = [
        (q, k, heads, m, n, b)
        for q, k in itertools.product(range(7), repeat=2)
        for b in range(max(1, (q * heads + m - 1) // m))
    ]
    compare_blocks(drivers, cases, causal_artifacts, f"small-{heads}-{m}-{n}")


def test_tile_and_kv_boundaries(drivers, causal_artifacts):
    cases = []
    for q, k, h, m, n in itertools.product(
        (1, 3, 513),
        (0, 1, 447, 448, 449, 511, 512, 513, 16384),
        (1, 129),
        (128, 256),
        (112, 448, 512),
    ):
        blocks = (q * h + m - 1) // m
        edge = max(0, q - k) * h // m
        for b in sorted({0, blocks - 1, min(edge, blocks - 1), max(0, min(edge - 1, blocks - 1))}):
            cases.append((q, k, h, m, n, b))
    compare_blocks(drivers, cases, causal_artifacts, "boundaries")


def schedule_case(q=(7,), k=(3,), heads=129, m=256, n=512, cores=20, ratio=2):
    return [1, heads, m, n, cores, ratio, len(q), *itertools.chain.from_iterable(zip(q, k))]


def check_schedule(item, got, native):
    _, heads, m, n, cores, ratio, batch, *lengths = item
    seqs = list(zip(lengths[::2], lengths[1::2]))
    tasks, prefix = [], [0]
    for b, (q, k) in enumerate(seqs):
        for index in range((q * heads + m - 1) // m):
            tasks.append((b, index, block_oracle(q, k, heads, m, n, index)))
        prefix.append(len(tasks))
    # Map the fixed source's candidate endpoints using independent task indices.
    endpoints = []
    for c in range(native[0]):
        b, row, s2 = native[7 + c], native[7 + cores + c], native[7 + 2 * cores + c]
        end = prefix[b] + row + bool(s2)
        if c == native[0] - 1:
            end = len(tasks)
        if not endpoints or end > endpoints[-1]:
            if end or not tasks:
                endpoints.append(end)
    expected = [
        0,
        len(endpoints),
        0,
        sum(t[2][11] for t in tasks),
        len(tasks),
        sum(t[2][9] for t in tasks),
    ]
    fields = [[0] * 26 for _ in range(4)]
    previous, visited = 0, []
    for c, end in enumerate(endpoints):
        b = next((b for b in range(batch) if prefix[b + 1] > end), batch)
        fields[0][c], fields[1][c] = b, end - prefix[b]
        assigned = tasks[previous:end]
        fields[3][c] = sum(t[2][11] for t in assigned)
        for sample, index, block in assigned:
            # Every effective Q/head row, including empty rows, belongs to one core.
            visited.extend((sample, row) for row in range(block[2], block[3]))
        previous = end
    expected[2] = max(fields[3])
    expected += list(itertools.chain.from_iterable(fields))
    assert got[:110] == expected, (item, got, expected, native)
    assert native[5:7] == [expected[3], expected[5]]
    assert visited == [(b, row) for b, (q, _) in enumerate(seqs) for row in range(q * heads)]
    assert len(visited) == len(set(visited))
    assert got[6 + got[1] - 1] == batch
    return bool(native[2]), bool(seqs and seqs[-1][0] and not seqs[-1][1])


def compare_schedules(drivers, items, directory, name, raw=None):
    got = evaluate(drivers[0], items if raw is None else raw)
    source = evaluate(drivers[1], items)
    migrated = evaluate(drivers[2], items)
    save(
        directory, name, inputs=items, raw=raw, actual=got, source=source, migrated_source=migrated
    )
    assert source == migrated
    flags = [
        check_schedule(item, actual, native) for item, actual, native in zip(items, got, source)
    ]
    return got, flags


@pytest.mark.parametrize("cores,ratio", itertools.product(range(1, 27), (1, 2)))
def test_complete_source_partitions(drivers, causal_artifacts, cores, ratio):
    lengths = [
        ([], []),
        ([0], [0]),
        ([7], [0]),
        ([1], [16384]),
        ([9], [2]),
        ([2], [9]),
        ([65], [65]),
        ([0, 7, 3, 1, 0], [9, 1, 0, 3, 0]),
        ([1, 3, 7, 0, 9], [0, 513, 2, 7, 0]),
        ([3] * 5, [2, 1, 3, 0, 0]),
    ]
    items = [
        schedule_case(q, k, cores=cores, ratio=ratio, m=m, n=n)
        for q, k in lengths
        for m, n in ((128, 112), (256, 512))
    ]
    compare_schedules(drivers, items, causal_artifacts, f"partition-{cores}-{ratio}")


def test_random_candidates_and_repeated_calls(drivers, causal_artifacts):
    rng = random.Random(80801)
    items = []
    for _ in range(200):
        batch = rng.randrange(1, 9)
        q = [rng.randrange(0, 19) for _ in range(batch)]
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
    got, flags = compare_schedules(drivers, items + items[::-1], causal_artifacts, "random-reuse")
    assert got[: len(items)] == got[len(items) :][::-1]
    assert any(fd for fd, _ in flags) and any(not fd for fd, _ in flags)
    assert any(suffix for _, suffix in flags)


@pytest.mark.parametrize("layout,page", [(0, 0), (1, 0), (2, 0), (0, 112), (2, 528)])
def test_raw_allocation_and_used_lengths(drivers, causal_artifacts, layout, page):
    raw, items, expected_spans = [], [], []
    for used in (False, True):
        q = [9, 9, 9, 9] if layout == 0 else [7, 0, 13, 3]
        k = [page * 3 if page else 17] * 4 if layout != 1 else [17, 3, 0, 9]
        uq = [2, 0, 9, 0] if used else q
        uk = [min(k[0], 1), 0, min(k[2], 3), 0] if used else k
        # Explicit KV lengths are mandatory for PA; absent used means allocation for non-PA.
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
        qstarts = list(itertools.accumulate(q, initial=0))
        kstarts = list(itertools.accumulate(k, initial=0))
        expected_spans.append(
            list(
                itertools.chain.from_iterable(
                    (qstarts[b], q[b], uq[b], uk[b])
                    if page
                    else (qstarts[b], kstarts[b], q[b], k[b], uq[b], uk[b])
                    for b in range(4)
                )
            )
        )
    got, _ = compare_schedules(drivers, items, causal_artifacts, f"raw-{layout}-{page}", raw=raw)
    for actual, spans in zip(got, expected_spans):
        assert actual[110:] == spans
    invalid = raw[1].copy()
    invalid[10] = -1  # Invalid physical allocation still rejects when some used lengths are zero.
    assert evaluate(drivers[0], [invalid])[0] == [101]


def test_wide_block_geometry(drivers, causal_artifacts):
    cases = [
        (2**30, 16384, 1, 256, 512, (2**30 // 256) - 1),
        (512, 0, (2**32 - 1) // 512, 256, 512, 16777213),
        (1, 1, (2**32 - 1) // 512, 256, 512, 32767),
    ]
    # Geometry has at most 256 rows even at the largest accepted field/address bounds.
    compare_blocks(drivers, cases, causal_artifacts, "wide-geometry")


def test_rejections_preserve_output(drivers, causal_artifacts):
    good = [7, 3, 129, 256, 512, 0]
    invalid = [
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
    ]
    cases, wanted = [], []
    for field, value, status in invalid:
        item = good.copy()
        item[field] = value
        cases.append([0, *item])
        wanted.append([status])
    cases += [[0, 0, 0, 1, 1, 1, 1], [0, 2**30, 1, 4, 256, 512, 0]]
    wanted += [[4], [2]]
    got = evaluate(drivers[0], cases)
    assert got == wanted
    schedules = []
    for field, value in ((1, 0), (2, 0), (3, 513), (4, 0), (4, 27), (5, 0), (5, 3)):
        item = schedule_case()
        item[field] = value
        schedules.append(item)
    schedules += [
        schedule_case([2**30], [16384], heads=1, m=1, n=1),
        schedule_case([7, -1], [3, 0]),
        schedule_case([7, 0], [3, 2**63 - 1]),
    ]
    result = evaluate(drivers[0], schedules)
    assert result == [[1], [3], [3], [5], [5], [5], [5], [2], [1], [2]]
    save(
        causal_artifacts,
        "rejections",
        blocks=cases,
        results=got,
        schedules=schedules,
        schedule_results=result,
    )
