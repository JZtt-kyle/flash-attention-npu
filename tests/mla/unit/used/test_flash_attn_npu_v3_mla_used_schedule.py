"""CPU used-length normalization and pinned-source scheduling differential.

Allocation geometry and effective source lengths are deliberately separate.
The oracle compiles unmodified CANN parser/tiling/SplitCore functions with an
independent FA3 input adapter; it never links the migrated length parser.
No device metadata or public used-length entry is enabled by these tests.
"""

import itertools
import json
import os
from pathlib import Path
import random

import pytest

from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    compile_driver as compile_bsnd,
    evaluate as evaluate_bsnd,
    length_case,
)
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_schedule import case
from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_schedule import mixed_case
from tests.mla.unit.tnd.test_flash_attn_npu_v3_mla_tnd_schedule import (
    build_oracle,
    check_tasks,
    compile_driver,
    evaluate,
    expected_spans as compact_spans,
    tnd_case,
)


@pytest.fixture(scope="session")
def used_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_USED_SCHEDULE_ARTIFACT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-used-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def used_scheduler(used_artifacts):
    return compile_driver(used_artifacts, False, used=True)


@pytest.fixture(scope="session")
def used_oracle(used_artifacts):
    return build_oracle(used_artifacts, used=True)


def used_case(layout=0, q=(7, 7), kv=(33, 33), uq=None, uk=None, **kwargs):
    """Build raw driver input, independently of any migrated normalization."""
    item = dict(layout=layout, q=list(q), kv=list(kv), uq=uq, uk=uk)
    item.update(kwargs)
    return item


def encode(item):
    layout, q, kv = item["layout"], item["q"], item["kv"]
    cq = item.get("cq", list(itertools.accumulate(q, initial=0)) if layout != 0 else [])
    ck = item.get("ck", list(itertools.accumulate(kv, initial=0)) if layout == 1 else [])
    defaults = dict(batch=len(q), sq=max(q, default=0), sk=max(kv, default=0))
    defaults.update(
        {
            k: v
            for k, v in item.items()
            if k
            in (
                "batch",
                "sq",
                "sk",
                "heads",
                "aic",
                "aiv",
                "ratio",
                "block_dim",
                "libapi",
                "l1",
                "ub",
                "l0a",
                "l0b",
                "l0c",
            )
        }
    )
    encoded = case(**defaults) + [
        int(item.get("has_k", layout == 1)),
        item.get("kcount", len(ck)),
        len(ck),
        *ck,
        item.get("total_q", sum(q)),
        item.get("total_k", sum(kv)),
        int(item.get("has_q", layout != 0)),
        item.get("qcount", len(cq)),
        len(cq),
        *cq,
        layout,
    ]
    for name in ("uq", "uk"):
        values = item[name]
        encoded += [
            int(item.get("has_" + name, values is not None)),
            item.get(name + "count", 0 if values is None else len(values)),
            0 if values is None else len(values),
            *(values or []),
        ]
    return encoded


def execute(binary, items, **kwargs):
    return evaluate(binary, [encode(item) for item in items], **kwargs)


def spans(item):
    q, kv = item["q"], item["kv"]
    uq = q if item["uq"] is None else item["uq"]
    uk = kv if item["uk"] is None else item["uk"]
    starts_q = list(itertools.accumulate(q, initial=0))
    starts_k = list(itertools.accumulate(kv, initial=0))
    return [
        max(uq, default=0),
        max(uk, default=0),
        max(q, default=0),
        max(kv, default=0),
        *itertools.chain.from_iterable(
            (starts_q[b], starts_k[b], q[b], kv[b], uq[b], uk[b]) for b in range(len(q))
        ),
    ]


def compare(scheduler, oracle, items, directory, name):
    (directory / (name + "-inputs.json")).write_text(json.dumps(items) + "\n")
    actual, source, native = (
        execute(scheduler, items),
        execute(oracle, items),
        execute(oracle, items, native=True),
    )
    (directory / (name + "-outputs.json")).write_text(
        json.dumps(dict(actual=actual, source=source, native=native)) + "\n"
    )
    counts = dict(native_non_fd=0, fa3_no_split_mapping=0, omitted_suffix=0)
    for item, got, want, original in zip(items, actual, source, native):
        assert got == want, (item, got, want)
        assert got[0] == 0 and len(got) == 360 + 6 * len(item["q"])
        assert got[356:] == spans(item)
        counts["native_non_fd" if original[5] == 0 else "fa3_no_split_mapping"] += 1
        if original[5] == 0:
            assert got == original
        # Scalar enumeration only uses the original test inputs' used values,
        # never the normalized output or either scheduler's length parser.
        compact = tnd_case(
            item["q"] if item["uq"] is None else item["uq"],
            item["kv"] if item["uk"] is None else item["uk"],
            heads=item.get("heads", 128),
        )
        check_tasks(compact, [*got[:356], *compact_spans(compact)])
        counts["omitted_suffix"] += got[18 + got[3] - 1] < len(item["q"])
    (directory / (name + ".json")).write_text(
        json.dumps(dict(cases=len(items), **counts, all_fields_spans_tasks_equal=True)) + "\n"
    )
    return actual


def test_u01_t01_physical_spans(used_scheduler, used_oracle, used_artifacts):
    cases = [
        used_case(uq=[3, 1], uk=[17, 0]),
        used_case(1, [3, 0, 1], [17, 0, 33], [2, 0, 1], [7, 0, 17]),
        used_case(2, [3, 0, 1], [64] * 3, [2, 0, 1], [17, 0, 33]),
        used_case(uq=[0, 0], uk=[0, 0]),
        used_case(1, [3, 0, 1], [17, 0, 33], [0, 0, 0], [0, 0, 0]),
    ]
    got = compare(used_scheduler, used_oracle, cases, used_artifacts, "u01-t01")
    assert got[0][360:] == [0, 0, 7, 33, 3, 17, 7, 33, 7, 33, 1, 0]
    assert got[1][360::6] == [0, 3, 3] and got[1][361::6] == [0, 17, 17]
    assert got[2][361::6] == [0, 64, 128]
    assert all(row[13] > 0 for row in got)  # Zero used is not zero physical capacity.


@pytest.mark.parametrize("layout,aic,ratio", itertools.product(range(3), range(1, 27), (1, 2)))
def test_source_boundaries(used_scheduler, used_oracle, used_artifacts, layout, aic, ratio):
    cases = []
    for batch, heads, (qcap, kcap) in itertools.product(
        sorted({1, max(1, aic - 1), aic, aic + 1}), (1, 128, 171), ((7, 33), (17, 513), (257, 8193))
    ):
        q = [qcap] * batch if layout == 0 else [[qcap, 0, 1][b % 3] for b in range(batch)]
        kv = [[kcap, 0, 17][b % 3] for b in range(batch)] if layout == 1 else [kcap] * batch
        uq = [min(n, [3, 0, 1, 7][b % 4]) for b, n in enumerate(q)]
        uk = [min(n, [17, 0, 513, kcap][b % 4]) for b, n in enumerate(kv)]
        cases.append(used_case(layout, q, kv, uq, uk, heads=heads, aic=aic, ratio=ratio))
    compare(used_scheduler, used_oracle, cases, used_artifacts, f"boundary-{layout}-{aic}-{ratio}")


def test_random_optional_updates_and_reuse(used_scheduler, used_oracle, used_artifacts):
    rng = random.Random(60641)
    cases = []
    for _ in range(240):
        layout, batch = rng.randrange(3), rng.randrange(1, 28)
        qc, kc = rng.choice((1, 7, 17, 257)), rng.choice((1, 33, 511, 512, 513, 8193, 16384))
        q = (
            [qc] * batch
            if layout == 0
            else [qc] + [rng.randrange(qc + 1) for _ in range(batch - 1)]
        )
        kv = (
            [kc] + [rng.randrange(kc + 1) for _ in range(batch - 1)]
            if layout == 1
            else [kc] * batch
        )
        uq = None if rng.randrange(2) else [rng.randrange(n + 1) for n in q]
        uk = None if rng.randrange(2) else [rng.randrange(n + 1) for n in kv]
        cases.append(
            used_case(
                layout,
                q,
                kv,
                uq,
                uk,
                heads=rng.choice((1, 8, 73, 128, 257)),
                aic=rng.randrange(1, 27),
                ratio=rng.choice((1, 2)),
            )
        )
    got = compare(used_scheduler, used_oracle, cases + cases[::-1], used_artifacts, "random-reuse")
    assert got[: len(cases)] == got[len(cases) :][::-1]
    # Each call snapshots new used lengths; physical offsets/capacities stay fixed.
    changed = [dict(c, uq=[0] * len(c["q"]), uk=[0] * len(c["kv"])) for c in cases]
    changed_got = compare(used_scheduler, used_oracle, changed, used_artifacts, "updates-zero")
    for before, after in zip(got, changed_got):
        for field in range(4):
            assert before[360 + field :: 6] == after[360 + field :: 6]
        assert after[4] == 0 and after[13] > 0


def test_source_array_semantics_and_capacity_geometry(used_scheduler, used_oracle, used_artifacts):
    items = [
        used_case(1, [2] * 10, [511] * 10, uq, uk)
        for uq, uk in itertools.product((None, [1] * 10), (None, [511] * 10, [511] + [0] * 9))
    ]
    got = compare(used_scheduler, used_oracle, items, used_artifacts, "source-length-mode")
    # Absent used preserves cumulative-tail equality; present used is noncumulative.
    # Q-used does not change the allocated geometry or its M-base candidate.
    assert [g[1] for g in got] == [256, 128, 256, 256, 128, 256]
    assert all(g[2] == 511 for g in got)
    upper = [dict(c, sq=7, sk=513) for c in items]
    assert execute(used_scheduler, upper) == got  # TND max arguments are upper bounds.


def test_absent_used_matches_existing_schedules(used_scheduler, used_artifacts):
    for layout in range(3):
        directory = used_artifacts / f"legacy-{layout}"
        directory.mkdir(exist_ok=True)
        legacy = (
            compile_bsnd(directory, False)
            if layout == 0
            else compile_driver(directory, False, mixed=layout == 2)
        )
        items, old = [], []
        for batch, sq, sk, heads, ratio in itertools.product(
            (1, 3, 10, 20), (1, 2, 17), (1, 513), (1, 128), (1, 2)
        ):
            q = [sq] * batch if layout == 0 else [sq if b % 2 == 0 else 0 for b in range(batch)]
            kv = [sk if b % 2 == 0 else 0 for b in range(batch)]
            shape = dict(heads=heads, ratio=ratio)
            items.append(
                used_case(
                    layout,
                    q,
                    kv if layout == 1 else [sk] * batch,
                    uk=kv if layout != 1 else None,
                    **shape,
                )
            )
            if layout == 0:
                old.append(length_case(kv, batch=batch, sq=sq, sk=sk, **shape))
            elif layout == 1:
                old.append(tnd_case(q, kv, **shape))
            else:
                old.append(mixed_case(q, kv, capacity=sk, **shape))
        actual = execute(used_scheduler, items)
        expected = evaluate_bsnd(legacy, old) if layout == 0 else evaluate(legacy, old)
        assert [g[:356] for g in actual] == [g[:356] for g in expected]
        assert all(g[356:] == spans(item) for g, item in zip(actual, items))
        (directory / "equivalence.json").write_text(
            json.dumps(dict(cases=len(items), all_schedule_fields_equal=True)) + "\n"
        )


@pytest.mark.parametrize(
    "layout,q,kv",
    [
        (layout, q, k)
        for layout in range(3)
        for q, k in (([], []), ([0, 0], [33, 33]), ([7, 7], [0, 0]))
    ],
)
@pytest.mark.parametrize("explicit", (False, True))
def test_global_empty_still_validates(used_scheduler, layout, q, kv, explicit):
    item = used_case(
        layout, q, kv, [0] * len(q) if explicit else None, [0] * len(kv) if explicit else None
    )
    got = execute(used_scheduler, [item])[0]
    assert got[:18] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert got[18:356] == [0] * 338 and got[356:] == spans(item)
    assert execute(used_scheduler, [item], parse=True) == [[0, *spans(item)]]
    bad = dict(item, uqcount=2**64 - 1)
    assert execute(used_scheduler, [bad]) == [[1]]


@pytest.mark.parametrize("layout", range(3))
@pytest.mark.parametrize(
    "updates,status",
    [
        ({"uq": [-1, 0]}, 1),
        ({"uk": [0, -(2**31)]}, 1),
        ({"uq": [8, 0]}, 1),
        ({"uk": [0, 34]}, 1),
        ({"uq": [2**31 - 1, 0]}, 1),
        ({"uk": [0, 2**31 - 1]}, 1),
        ({"uq": [1], "uqcount": 1}, 1),
        ({"uk": [1], "ukcount": 1}, 1),
        ({"uq": [1, 1], "uqcount": 0}, 1),
        ({"uk": [1, 1], "ukcount": 0}, 1),
        ({"uqcount": 1}, 1),
        ({"ukcount": 2**64 - 1}, 1),
        ({"uq": [1, 1], "has_uq": False}, 1),
        ({"uk": [1, 1], "has_uk": False}, 1),
        ({"qcount": 2**64 - 1}, 1),
        ({"kcount": 2**64 - 1}, 1),
        ({"batch": -1}, 1),
        ({"sq": -1}, 1),
        ({"sk": -1}, 1),
        ({"heads": 0}, 1),
        ({"total_q": -1}, 1),
        ({"total_k": -1}, 1),
        ({"total_q": 13}, 1),
        ({"total_k": 65}, 1),
        ({"sk": 16385}, 1),
        ({"batch": 2**32}, 2),
        ({"sq": 2**32}, 2),
        ({"heads": 2**23}, 2),
        ({"sq": 2**30 + 1}, 2),
        ({"sq": 2**30, "heads": 4}, 2),
        ({"total_q": 2**63 - 1}, 2),
        ({"layout": 99}, 1),
        ({"aic": 0, "aiv": 0}, 3),
        ({"aic": 27}, 3),
        ({"ratio": 3}, 3),
        ({"aiv": 21}, 3),
        ({"block_dim": 21}, 3),
        ({"libapi": 2**64 - 1}, 5),
    ],
)
def test_rejections_preserve_output(used_scheduler, layout, updates, status):
    item = used_case(layout)
    item.update(updates)
    assert execute(used_scheduler, [item]) == [[status]]
    if status in (1, 2):
        assert execute(used_scheduler, [item], parse=True) == [[status]]


@pytest.mark.parametrize("field", ("cq", "ck"))
@pytest.mark.parametrize(
    "prefix", ([1, 7, 14], [-1, 7, 14], [0, 8, 7], [0, -(2**31), 14], [0, 7, 13])
)
def test_allocation_prefix_validated_even_with_zero_used(used_scheduler, field, prefix):
    item = used_case(1, [7, 7], [7, 7], [0, 0], [0, 0], **{field: prefix})
    assert execute(used_scheduler, [item]) == [[1]]


def test_per_sample_bounds_and_required_prefixes(used_scheduler):
    cases = [
        used_case(1, [3, 0, 1], [17, 0, 33], uq=[0, 1, 0]),
        used_case(1, [3, 0, 1], [17, 0, 33], uk=[0, 1, 0]),
        used_case(2, [3, 0, 1], [33] * 3, uq=[0, 0, 2]),
        used_case(1, has_q=False),
        used_case(1, has_k=False),
        used_case(2, has_q=False),
        used_case(0, has_q=True),
        used_case(0, has_k=True),
        used_case(2, has_k=True),
        used_case(1, [], [], has_q=False),
        used_case(1, [], [], has_k=False),
        used_case(1, sq=6),
        used_case(1, sk=32),
        used_case(1, total_q=2**31),
        used_case(1, total_k=2**31),
    ]
    assert execute(used_scheduler, cases) == [[1]] * 13 + [[2], [2]]


def test_wide_addresses_and_active_tile_bound(used_scheduler, used_artifacts):
    batch = 65538
    item = used_case(0, [65536] * batch, [16384] * batch, [0] * batch, [0] * batch, heads=1)
    parsed = execute(used_scheduler, [item], parse=True)[0]
    assert parsed == [0, *spans(item)] and parsed[-6] > 2**32
    batch = 2**18 + 2
    mixed = used_case(2, [0] * batch, [16384] * batch, uk=[0] * batch)
    assert execute(used_scheduler, [mixed], parse=True)[0][-5] > 2**32
    full = used_case(1, [2**27] * 15 + [2**27 - 1], [16384] * 16, heads=16)
    assert execute(used_scheduler, [full], parse=True) == [[0, *spans(full)]]
    assert execute(used_scheduler, [full]) == [[2]]  # >uint32 active physical tiles.
    zero = dict(full, uq=[0] * 16)
    assert execute(used_scheduler, [zero])[0][4] == 0
    # Physical address validation still applies even when no query is used.
    bad = used_case(1, [2**29] * 3, [0] * 3, [0] * 3, heads=8388607, sq=1)
    assert execute(used_scheduler, [bad]) == [[2]]
    (used_artifacts / "wide-addresses.json").write_text(
        json.dumps(dict(last_bsnd_q_start=parsed[-6], last_mixed_kv_start=(batch - 1) * 16384))
        + "\n"
    )


@pytest.mark.parametrize("ratio", (1, 2))
def test_platform_and_workspace_edges(used_scheduler, used_oracle, used_artifacts, ratio):
    item = used_case(uq=[3, 1], uk=[17, 0], ratio=ratio, block_dim=20)
    got = compare(used_scheduler, used_oracle, [item], used_artifacts, f"workspace-{ratio}")[0]
    limit = dict(item, libapi=2**64 - 1 - got[13])
    assert execute(used_scheduler, [limit])[0][15] == 2**64 - 1
    assert execute(used_scheduler, [dict(limit, libapi=limit["libapi"] + 1)]) == [[5]]
    for key, value in (
        ("l1", 516096),
        ("ub", 178176 if ratio == 1 else 174080),
        ("l0a", 65536),
        ("l0b", 65536),
        ("l0c", 131072),
    ):
        assert execute(used_scheduler, [dict(item, **{key: value - 1})]) == [[4]]
    assert execute(used_scheduler, [used_case(q=[7] * 20, kv=[513] * 20, block_dim=1)]) == [[3]]
