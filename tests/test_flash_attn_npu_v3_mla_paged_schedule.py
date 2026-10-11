"""Independent PA CPU scheduling; no metadata or public PA entry is enabled.

The oracle uses pinned, verbatim CANN parser/tiling/SplitCore bodies. Its input
adapter is independent of the migrated normalization; KV is always per sample.
Native FD candidates (including all vectors/workspace) are checked separately
against the migrated SplitCore, and retained alongside FA3 whole-M results.
"""

import itertools
import json
import os
from pathlib import Path
import random

import pytest

from tests.test_flash_attn_npu_v3_mla_tnd_schedule import (
    build_oracle,
    check_tasks,
    compile_driver,
    evaluate,
    expected_spans,
    tnd_case,
)
from tests.test_flash_attn_npu_v3_mla_used_schedule import encode as encode_used, used_case


@pytest.fixture(scope="session")
def paged_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_PAGED_SCHEDULE_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-paged-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def paged_scheduler(paged_artifacts):
    return compile_driver(paged_artifacts, False, used=True, paged=True)


@pytest.fixture(scope="session")
def paged_oracles(paged_artifacts):
    source = build_oracle(paged_artifacts, used=True, paged=True)
    migrated = compile_driver(paged_artifacts, True, used=True, paged=True, migrated_split=True)
    return source, migrated


def paged_case(layout=0, q=(7, 7), lengths=(17, 0), uq=None, page=112, capacity=None, **kwargs):
    if capacity is None:
        capacity = ((max(lengths, default=0) + page - 1) // page + 1) * page
    return used_case(
        layout, q, [capacity] * len(q), uq, list(lengths), page=page, sk=capacity, **kwargs
    )


def encode(item):
    return [*encode_used(item), item["page"]]


def execute(binary, items, **kwargs):
    return evaluate(binary, [encode(item) for item in items], **kwargs)


def spans(item):
    q = item["q"]
    uq = q if item["uq"] is None else item["uq"]
    starts = list(itertools.accumulate(q, initial=0))
    return [
        max(uq, default=0),
        max(item["uk"], default=0),
        max(q, default=0),
        *itertools.chain.from_iterable(
            (starts[b], q[b], uq[b], item["uk"][b]) for b in range(len(q))
        ),
    ]


def compare(scheduler, oracles, items, directory, name):
    (directory / (name + "-inputs.json")).write_text(json.dumps(items) + "\n")
    actual = execute(scheduler, items)
    source = execute(oracles[0], items)
    native = execute(oracles[0], items, native=True)
    candidate = execute(oracles[1], items, native=True)
    (directory / (name + "-outputs.json")).write_text(
        json.dumps(dict(actual=actual, source=source, native=native, candidate=candidate)) + "\n"
    )
    counts = dict(native_non_fd=0, fa3_no_split_mapping=0, omitted_suffix=0)
    for item, got, want, original, split in zip(items, actual, source, native, candidate):
        assert got == want, (item, got, want)
        assert original == split, (item, original, split)
        assert got[0] == 0 and len(got) == 363 + 4 * len(item["q"])
        assert got[356:-4] == spans(item)
        page, capacity, qcap, source_s2 = got[-4:]
        assert (page, capacity, qcap) == (item["page"], item["sk"], max(item["q"]))
        assert source_s2 == (max(item["uk"]) + page - 1) // page * page <= capacity
        counts["native_non_fd" if original[5] == 0 else "fa3_no_split_mapping"] += 1
        if original[5] == 0:
            assert got == original
        compact = tnd_case(
            item["q"] if item["uq"] is None else item["uq"],
            item["uk"],
            heads=item.get("heads", 128),
        )
        check_tasks(compact, [*got[:356], *expected_spans(compact)])
        counts["omitted_suffix"] += got[18 + got[3] - 1] < len(item["q"])
    (directory / (name + ".json")).write_text(
        json.dumps(
            dict(
                cases=len(items),
                **counts,
                all_fields_spans_tasks_equal=True,
                native_fd_fields_workspace_equal=True,
            )
        )
        + "\n"
    )
    return actual


@pytest.mark.parametrize(
    "page,layout,ratio", itertools.product(range(16, 1025, 16), (0, 2), (1, 2))
)
def test_all_page_sizes(paged_scheduler, paged_oracles, paged_artifacts, page, layout, ratio):
    capacity = 16384 // page * page
    items = []
    for length in sorted(
        {1, page - 1, page, page + 1, 447, 448, 449, 511, 512, 513, 1025, capacity}
    ):
        for used in (False, True):
            q = [7] * 3 if layout == 0 else [7, 0, 3]
            items.append(
                paged_case(
                    layout,
                    q,
                    [length, 0, min(17, length)],
                    [3, 0, 1] if used else None,
                    page,
                    capacity,
                    ratio=ratio,
                )
            )
    compare(
        paged_scheduler, paged_oracles, items, paged_artifacts, f"pages-{page}-{layout}-{ratio}"
    )


@pytest.mark.parametrize("aic,ratio", itertools.product(range(1, 27), (1, 2)))
def test_core_boundaries(paged_scheduler, paged_oracles, paged_artifacts, aic, ratio):
    items = []
    for page, layout, batch in itertools.product(
        (112, 528), (0, 2), sorted({1, max(1, aic - 1), aic, aic + 1})
    ):
        q = [2] * batch if layout == 0 else [2 if b % 2 == 0 else 0 for b in range(batch)]
        for equal in (True, False):
            lengths = [1025] * batch if equal else [1025 if b % 2 == 0 else 0 for b in range(batch)]
            items.append(paged_case(layout, q, lengths, page=page, aic=aic, ratio=ratio))
    compare(paged_scheduler, paged_oracles, items, paged_artifacts, f"cores-{aic}-{ratio}")


def test_alignment_capacity_and_kv_by_batch(paged_scheduler, paged_oracles, paged_artifacts):
    items = [
        paged_case(layout, [2] * 10, [length] * 10, page=page, capacity=page * 20)
        for layout, page, length in itertools.product((0, 2), (112, 528), (1, 449, 1025))
    ]
    got = compare(paged_scheduler, paged_oracles, items, paged_artifacts, "alignment")
    for item, row in zip(items, got):
        assert row[1] == 128  # Equal *noncumulative* KV lengths, including TND Q.
        assert row[2] == (min(448, row[-1]) if item["page"] == 112 else 512)
    # Logical capacity changes validation/record geometry, not the source tiling.
    smaller = [dict(item, sk=item["page"] * 10, kv=[item["page"] * 10] * 10) for item in items]
    other = compare(paged_scheduler, paged_oracles, smaller, paged_artifacts, "capacity")
    assert [r[:-3] + r[-2:] for r in got] == [r[:-3] + r[-2:] for r in other]
    for page, row in ((112, got[2]), (528, got[5])):
        ranges = [(start, min(start + row[2], 1025)) for start in range(0, 1025, row[2])]
        assert any(a // page != (b - 1) // page for a, b in ranges)
    # P=112 divides each non-tail base; P=528 has a 512..1024 cross-page tile.
    assert got[2][2] % 112 == 0 and 512 // 528 != 1023 // 528


def test_random_and_zero_query_used(paged_scheduler, paged_oracles, paged_artifacts):
    rng = random.Random(70702)
    items = []
    for _ in range(160):
        page, layout, batch = rng.randrange(1, 65) * 16, rng.choice((0, 2)), rng.randrange(1, 28)
        capacity = rng.randrange(1, 16384 // page + 1) * page
        q = [rng.choice((1, 7, 17, 257))] * batch
        if layout == 2:
            q[1:] = [rng.randrange(q[0] + 1) for _ in q[1:]]
        lengths = [rng.randrange(1, capacity + 1)] + [rng.randrange(capacity + 1) for _ in q[1:]]
        uq = None if rng.randrange(2) else [rng.randrange(v + 1) for v in q]
        items.append(
            paged_case(
                layout,
                q,
                lengths,
                uq,
                page,
                capacity,
                heads=rng.choice((1, 8, 73, 128, 257)),
                aic=rng.randrange(1, 27),
                ratio=rng.choice((1, 2)),
            )
        )
    items += [
        paged_case(layout, [7, 7], [17, 0], [0, 0], page)
        for layout, page in itertools.product((0, 2), (112, 528))
    ]
    got = compare(paged_scheduler, paged_oracles, items, paged_artifacts, "random")
    assert all(row[4] == 0 and row[13] > 0 for row in got[-4:])


@pytest.mark.parametrize("layout", (0, 2))
@pytest.mark.parametrize(
    "q,lengths,capacity",
    [([], [], 112), ([0, 0], [17, 0], 112), ([7, 7], [0, 0], 112), ([7, 7], [0, 0], 0)],
)
def test_empty_still_validates(paged_scheduler, layout, q, lengths, capacity):
    item = paged_case(layout, q, lengths, capacity=capacity)
    got = execute(paged_scheduler, [item])[0]
    assert got[:18] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert got[18:356] == [0] * 338 and got[356:-4] == spans(item)
    assert execute(paged_scheduler, [item], parse=True) == [[0, *spans(item)]]
    assert execute(paged_scheduler, [dict(item, ukcount=2**64 - 1)]) == [[1]]


@pytest.mark.parametrize("layout", (0, 2))
@pytest.mark.parametrize(
    "updates,status",
    [({"page": p}, 1) for p in (-(2**63), -1, 0, 1, 15, 17, 1025, 1040, 2**63 - 1)]
    + [
        ({"sk": -1}, 1),
        ({"sk": 111}, 1),
        ({"sk": 16464}, 1),
        ({"sk": 2**63 - 1}, 1),
        ({"batch": -1}, 1),
        ({"batch": 2**32}, 2),
        ({"sq": -1}, 1),
        ({"sq": 2**32}, 2),
        ({"heads": 0}, 1),
        ({"heads": 2**23}, 2),
        ({"sq": 2**30 + 1}, 2),
        ({"sq": 2**30, "heads": 4}, 2),
        ({"total_q": -1}, 1),
        ({"total_q": 13}, 1),
        ({"total_q": 2**63 - 1}, 2),
        ({"layout": 1}, 1),
        ({"uq": [-1, 0]}, 1),
        ({"uq": [8, 0]}, 1),
        ({"uq": [2**31 - 1, 0]}, 1),
        ({"uk": [0, -(2**31)]}, 1),
        ({"uk": [0, 113]}, 1),
        ({"uk": [0, 2**31 - 1]}, 1),
        ({"uq": [1], "uqcount": 1}, 1),
        ({"uk": [1], "ukcount": 1}, 1),
        ({"uq": [1, 1], "uqcount": 0}, 1),
        ({"uk": [1, 1], "ukcount": 0}, 1),
        ({"uqcount": 1}, 1),
        ({"ukcount": 2**64 - 1}, 1),
        ({"has_uk": False}, 1),
        ({"has_uk": False, "ukcount": 0}, 1),
        ({"qcount": 2**64 - 1}, 1),
        ({"aic": 0, "aiv": 0}, 3),
        ({"aic": 27}, 3),
        ({"ratio": 3}, 3),
        ({"aiv": 21}, 3),
        ({"block_dim": 21}, 3),
        ({"libapi": 2**64 - 1}, 5),
    ],
)
def test_rejections_preserve_output(paged_scheduler, layout, updates, status):
    item = dict(paged_case(layout, capacity=112), **updates)
    assert execute(paged_scheduler, [item]) == [[status]]
    if status in (1, 2):
        assert execute(paged_scheduler, [item], parse=True) == [[status]]


@pytest.mark.parametrize(
    "prefix", ([1, 7, 14], [-1, 7, 14], [0, 8, 7], [0, -(2**31), 14], [0, 7, 13])
)
def test_query_allocation_with_zero_used(paged_scheduler, prefix):
    item = paged_case(2, uq=[0, 0], cq=prefix)
    assert execute(paged_scheduler, [item]) == [[1]]


def test_query_prefixes_bounds_and_updates(paged_scheduler, paged_oracles, paged_artifacts):
    bad = [
        paged_case(2, [3, 0, 1], [17, 0, 33], [0, 1, 0]),
        paged_case(2, has_q=False),
        paged_case(0, has_q=True),
        paged_case(2, [], [], has_q=False),
        paged_case(2, sq=6),
        paged_case(2, total_q=2**31),
    ]
    assert execute(paged_scheduler, bad) == [[1]] * 5 + [[2]]
    items = [
        paged_case(2, [3, 0, 1], [17, 0, 33], uq=uq, page=112, capacity=224)
        for uq in (None, [3, 0, 1], [1, 0, 0])
    ]
    got = compare(paged_scheduler, paged_oracles, items, paged_artifacts, "query-updates")
    assert [r[359::4][:3] for r in got] == [[0, 3, 3]] * 3
    assert execute(paged_scheduler, [dict(item, sq=257) for item in items]) == got


def test_empty_values_and_absent_arrays(paged_scheduler):
    invalid = []
    valid = []
    for layout in (0, 2):
        invalid += [
            paged_case(layout, [0, 0], [-1, 0], capacity=112),
            paged_case(layout, [7, 7], [0, 113], [0, 0], capacity=112),
            paged_case(layout, [7, 7], [0, 0], [8, 0], capacity=0),
            paged_case(layout, [0, 0], [0, 0], [0, 1], capacity=112),
        ]
        valid.append(paged_case(layout, [], [], capacity=0, has_uk=False))
    assert execute(paged_scheduler, invalid) == [[1]] * len(invalid)
    assert execute(paged_scheduler, invalid, parse=True) == [[1]] * len(invalid)
    assert all(row[:4] == [0, 0, 0, 1] for row in execute(paged_scheduler, valid))


def test_wide_query_addresses_and_tile_limit(paged_scheduler, paged_artifacts):
    batch = 65538
    item = paged_case(
        0, [65536] * batch, [0] * batch, [0] * batch, page=1024, capacity=16384, heads=1
    )
    got = execute(paged_scheduler, [item], parse=True)[0]
    assert got == [0, *spans(item)] and got[-4] > 2**32
    full = paged_case(
        2, [2**27] * 15 + [2**27 - 1], [16384] * 16, page=1024, capacity=16384, heads=16
    )
    assert execute(paged_scheduler, [full], parse=True) == [[0, *spans(full)]]
    assert execute(paged_scheduler, [full]) == [[2]]
    assert execute(paged_scheduler, [dict(full, uq=[0] * 16)])[0][4] == 0
    (paged_artifacts / "wide-addresses.json").write_text(
        json.dumps(dict(last_query_start=got[-4], active_tile_overflow_rejected=True)) + "\n"
    )


@pytest.mark.parametrize("ratio", (1, 2))
def test_resource_limits(paged_scheduler, paged_oracles, paged_artifacts, ratio):
    item = paged_case(ratio=ratio, block_dim=20)
    got = compare(paged_scheduler, paged_oracles, [item], paged_artifacts, f"workspace-{ratio}")[0]
    limit = dict(item, libapi=2**64 - 1 - got[13])
    assert execute(paged_scheduler, [limit])[0][15] == 2**64 - 1
    assert execute(paged_scheduler, [dict(limit, libapi=limit["libapi"] + 1)]) == [[5]]
    for key, value in (
        ("l1", 516096),
        ("ub", 178176 if ratio == 1 else 174080),
        ("l0a", 65536),
        ("l0b", 65536),
        ("l0c", 131072),
    ):
        assert execute(paged_scheduler, [dict(item, **{key: value - 1})]) == [[4]]
    assert execute(paged_scheduler, [paged_case(q=[7] * 20, lengths=[513] * 20, block_dim=1)]) == [
        [3]
    ]
