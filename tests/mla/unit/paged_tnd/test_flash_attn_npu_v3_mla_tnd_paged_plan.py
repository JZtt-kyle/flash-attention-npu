"""TND PA static plans against pinned source schedules and independent v6 contexts.

Only CPU descriptors enter the plan. Allocation prefixes/used are supplied
separately to the source oracle, serializer and consumer; no attention launch
or public TND PA support is implied. MLA_PAGED_PLAN_SANITIZE=1 adds ASan/UBSan.
"""

from copy import deepcopy
import itertools

import pytest

from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_layout import view
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_plan import (
    paged_artifacts as paged_artifacts,
    paged_builder as paged_builder,
    paged_consumer as paged_consumer,
    paged_oracles as paged_oracles,
    paged_plan as paged_plan,
    I64_MAX,
    LOCAL_MIN,
    MAX_BATCH,
    U32_MAX,
    check_consumed,
    check_source,
    check_static,
    context_for,
    decode,
    evaluate,
    expected_record,
    paged_case,
    run_builder,
    static_case,
)


def tnd_static_case(total=10, batch=3, max_q=19, **kwargs):
    spec = static_case(batch=batch, **kwargs)
    heads = spec["q"]["sizes"][2]
    for name, dim, padding, offset in (("q", 64, 3, 7), ("qv", 512, 5, 13)):
        spec[name] = view(
            (total, heads, dim), (heads * (dim + padding) + 11, dim + padding, 1), offset
        )
    spec["max_q"] = max_q
    return spec


def schedule_item(spec, query=(7, 0, 3), lengths=(17, 0, 33), used=None):
    p = spec["platform"]
    capacity = spec["table"]["sizes"][1] * spec["k"]["sizes"][1]
    assert sum(query) == spec["q"]["sizes"][0]
    assert len(query) == spec["table"]["sizes"][0] == len(lengths)
    return paged_case(
        2,
        query,
        lengths,
        (used if used is not None else [n // 2 for n in query]) if spec["used"] else None,
        spec["k"]["sizes"][1],
        capacity,
        sq=spec["max_q"],
        heads=spec["q"]["sizes"][1],
        aic=p[0],
        aiv=p[1],
        block_dim=p[0] if sum(query) and capacity else 1,
        l1=p[2],
        ub=p[3],
        l0a=p[4],
        l0b=p[5],
        l0c=p[6],
        libapi=p[7],
    )


@pytest.mark.parametrize("page,dtype", itertools.product(range(16, 1025, 16), (1, 2)))
def test_source_budgets_and_context(
    paged_plan, paged_builder, paged_oracles, paged_consumer, page, dtype
):
    for used in (False, True):
        spec = tnd_static_case(page=page, width=16384 // page, dtype=dtype, used=used)
        plan = evaluate(paged_plan, [spec])[0]
        check_static(spec, plan)
        items = [
            schedule_item(spec, lengths=lengths)
            for length in sorted({1, page - 1, page, page + 1, 449, 513, 16384 // page * page})
            for lengths in ([length] * 3, [length, 0, min(length, 17)])
        ]
        rows = check_source(
            paged_builder, paged_oracles, items, dtype, f"tnd-plan-{page}-{dtype}-{used}"
        )
        for item, row in zip(items, rows):
            check_consumed(paged_consumer, row, plan, item)
            # The third sample starts after 7 allocated tokens even with Q-used.
            assert row[418:421] == [7, 0, 3]


@pytest.mark.parametrize("aic", range(1, 27))
def test_core_counts_and_m_base_change(
    paged_plan, paged_builder, paged_oracles, paged_consumer, aic
):
    batch = max(2, aic // 2)
    spec = tnd_static_case(total=2 * batch, batch=batch, max_q=9, page=528, width=4)
    spec["platform"][:2] = [aic, 2 * aic]
    plan = evaluate(paged_plan, [spec])[0]
    check_static(spec, plan)
    items = [
        schedule_item(spec, [2] * batch, lengths)
        for lengths in ([1025] * batch, [1025] + [17] * (batch - 1))
    ]
    rows = check_source(paged_builder, paged_oracles, items, 1, f"tnd-cores-{aic}")
    if aic >= 4 and aic % 2 == 0:
        assert [row[15] for row in rows] == [128, 256]
    for item, row in zip(items, rows):
        check_consumed(paged_consumer, row, plan, item)


@pytest.mark.parametrize(
    "page,width,query,heads",
    itertools.product((16, 112, 528, 1024), (1, 2), ((0, 1, 0), (3, 0, 1)), (1, 17)),
)
def test_short_allocations_bound_loose_maximum(
    paged_plan, paged_builder, paged_oracles, paged_consumer, page, width, query, heads
):
    spec = tnd_static_case(total=sum(query), max_q=100, heads=heads, page=page, width=width)
    plan = evaluate(paged_plan, [spec])[0]
    check_static(spec, plan)
    items = [schedule_item(spec, query, [n] * 3) for n in (1, page * width)]
    rows = check_source(
        paged_builder, paged_oracles, items, 1, f"tnd-short-{page}-{width}-{query}-{heads}"
    )
    for item, row in zip(items, rows):
        check_consumed(paged_consumer, row, plan, item)
    # Tightening the declared maximum to T cannot change the required reserve.
    tight = dict(spec, max_q=sum(query))
    assert evaluate(paged_plan, [tight])[0]["workspace"] == plan["workspace"]


@pytest.mark.parametrize(
    "dtype,used,empty",
    itertools.product(
        (1, 2), (False, True), ("batch", "q", "max_q", "table", "pool", "lengths", "used")
    ),
)
def test_empty_allocations_and_dynamic_lengths(
    paged_plan, paged_builder, paged_consumer, dtype, used, empty
):
    kwargs = {
        "batch": dict(batch=0, total=0),
        "q": dict(total=0),
        "max_q": dict(total=0, max_q=0),
        "table": dict(width=0),
        "pool": dict(pool=0),
    }.get(empty, {})
    spec = tnd_static_case(dtype=dtype, used=used, **kwargs)
    plan = evaluate(paged_plan, [spec])[0]
    check_static(spec, plan)
    batch = spec["table"]["sizes"][0]
    query = [7, 0, 3] if spec["q"]["sizes"][0] else [0] * batch
    lengths = [17] * batch if empty in ("q", "max_q", "used") else [0] * batch
    item = schedule_item(spec, query, lengths, used=[0] * batch)
    row = run_builder(paged_builder, [item], dtype)[0]
    if empty != "used":
        assert row == expected_record(item, dtype, None)
        assert row[383:391] == [0] * 8
    check_consumed(paged_consumer, row, plan, item)


@pytest.mark.parametrize("name", ("q", "qv", "k", "v", "table"))
@pytest.mark.parametrize("profile", ("broadcast", "overlap", "offset", "wide"))
def test_independent_views(paged_plan, name, profile):
    spec = tnd_static_case()
    tensor = spec[name]
    strides = list(tensor["strides"])
    if profile == "broadcast":
        strides[:-1] = [0] * (len(strides) - 1)
    elif profile == "overlap":
        strides[:-1] = [1] * (len(strides) - 1)
    elif profile == "wide":
        strides[0] = (1 << 32) + 17
    spec[name] = view(
        tensor["sizes"],
        strides,
        (1 << 32) + 3 if profile == "offset" else 7,
        4 if name == "table" else 2,
    )
    check_static(spec, evaluate(paged_plan, [spec])[0])


def test_ntd_views_and_dynamic_prefix_reuse(paged_plan, paged_builder, paged_consumer):
    spec = tnd_static_case(used=True)
    for name in ("q", "qv"):
        total, heads, dim = spec[name]["sizes"]
        spec[name] = view((total, heads, dim), (dim + 3, total * (dim + 3) + 7, 1), 17)
    plan = evaluate(paged_plan, [spec])[0]
    check_static(spec, plan)
    items = [schedule_item(spec, query, used=[1, 0, 0]) for query in ((7, 0, 3), (1, 9, 0))]
    rows = run_builder(paged_builder, items)
    for item, row in zip(items, rows):
        check_consumed(paged_consumer, row, plan, item)
    # Static context is reusable; the old snapshot must still reject new prefixes.
    assert decode(paged_consumer, rows[0], context_for(plan), items[1])[0] != 0
    assert [rows[0][418], rows[1][418]] == [7, 10]


def broadcast_geometry(**kwargs):
    spec = tnd_static_case(**kwargs)
    for name in ("q", "qv", "k", "v", "table"):
        sizes = spec[name]["sizes"]
        spec[name] = view(sizes, [0] * (len(sizes) - 1) + [1], 0, 4 if name == "table" else 2)
    return spec


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch=MAX_BATCH, total=0),
        dict(batch=MAX_BATCH, total=1, max_q=1 << 30, heads=1),
        dict(batch=2, total=(1 << 31) - 1, max_q=1 << 30, heads=1),
        dict(total=1, max_q=1, heads=U32_MAX // 512),
        dict(
            batch=MAX_BATCH,
            total=(I64_MAX // 1024) // (U32_MAX // 512),
            max_q=2,
            heads=U32_MAX // 512,
        ),
        dict(pool=1 << 31),
        dict(pool=100000, page=1024, width=1),
    ],
)
def test_compact_and_wide_geometry_without_batch_allocation(paged_plan, kwargs):
    spec = broadcast_geometry(**kwargs)
    check_static(spec, evaluate(paged_plan, [spec])[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch=MAX_BATCH + 1, total=0),
        dict(batch=0, total=1),
        dict(total=1, max_q=0),
        dict(batch=2, total=11, max_q=5),
        dict(batch=3, total=1 << 31, max_q=1 << 30, heads=1),
        dict(max_q=(1 << 30) + 1, heads=1),
        dict(max_q=1 << 30, heads=4),
        dict(total=1, max_q=1, heads=U32_MAX // 512 + 1),
        dict(
            batch=MAX_BATCH,
            total=(I64_MAX // 1024) // (U32_MAX // 512) + 1,
            max_q=2,
            heads=U32_MAX // 512,
        ),
        dict(heads=0),
        dict(page=1024, width=17),
    ],
)
def test_shape_bounds(paged_plan, kwargs):
    assert evaluate(paged_plan, [broadcast_geometry(**kwargs)])[0]["status"] == 1


@pytest.mark.parametrize("name", ("q", "qv", "k", "v", "table"))
@pytest.mark.parametrize("fault", ("stride", "last", "storage", "offset", "bytes"))
def test_complete_storage_rejected(paged_plan, name, fault):
    spec = tnd_static_case(used=True)
    t = spec[name]
    if fault == "stride":
        t["strides"][0] = -1
    elif fault == "last":
        t["strides"][-1] = 2
    elif fault == "storage":
        t["storage"] -= 1
    elif fault == "offset":
        t["offset"] = -1
    else:
        t["strides"][0] = I64_MAX
        t["storage"] = (1 << 64) - 1
    assert evaluate(paged_plan, [spec])[0]["status"] == 2


@pytest.mark.parametrize(
    "field,value,status",
    [
        ("dtype", 0, 3),
        ("dtype", 3, 3),
        ("max_q", -1, 1),
        ("q.sizes.0", -1, 1),
        ("q.sizes.1", -1, 1),
        ("q.sizes.2", 512, 1),
        ("qv.sizes.0", 9, 1),
        ("qv.sizes.1", 64, 1),
        ("qv.sizes.2", 64, 1),
        ("table.sizes.0", -1, 1),
        ("table.sizes.1", -1, 2),
        ("k.sizes.0", (1 << 31) + 1, 2),
        ("k.sizes.1", 0, 2),
        ("k.sizes.1", 15, 2),
        ("k.sizes.1", 1025, 2),
        ("k.sizes.2", 2, 2),
        ("k.sizes.3", 512, 2),
        ("v.sizes.0", 8, 2),
        ("v.sizes.1", 128, 2),
        ("v.sizes.2", 2, 2),
        ("v.sizes.3", 64, 2),
        ("platform.0", 0, 4),
        ("platform.0", 27, 4),
        ("platform.1", 20, 4),
        ("platform.1", U32_MAX, 4),
        ("blocks.0", 0, 4),
        ("blocks.10", 21, 4),
        ("blocks.19", 19, 4),
    ],
)
def test_invalid_fields(paged_plan, field, value, status):
    spec = tnd_static_case()
    fields = field.split(".")
    target = spec
    for key in fields[:-1]:
        target = target[int(key) if isinstance(target, list) else key]
    target[int(fields[-1]) if isinstance(target, list) else fields[-1]] = value
    assert evaluate(paged_plan, [spec])[0]["status"] == status


@pytest.mark.parametrize("empty", (False, True))
def test_local_resources_and_workspace_boundary(paged_plan, empty):
    spec = tnd_static_case(total=0 if empty else 10)
    spec["platform"][2:7] = LOCAL_MIN
    good = evaluate(paged_plan, [spec])[0]
    check_static(spec, good)
    for axis in range(2, 7):
        bad = deepcopy(spec)
        bad["platform"][axis] -= 1
        assert evaluate(paged_plan, [bad])[0]["status"] == 5
    if empty:
        spec["platform"][-1] = (1 << 64) - 1
        check_static(spec, evaluate(paged_plan, [spec])[0])
        spec["blocks"][0] = 2
        assert evaluate(paged_plan, [spec])[0]["status"] == 4
    else:
        spec["platform"][-1] = I64_MAX - good["context"][13]
        assert evaluate(paged_plan, [spec])[0]["workspace"] == I64_MAX
        spec["platform"][-1] += 1
        assert evaluate(paged_plan, [spec])[0]["status"] == 6


def test_empty_query_still_validates_full_views(paged_plan):
    cases = []
    for name in ("k", "v", "table"):
        spec = tnd_static_case(total=0)
        spec[name]["storage"] -= 1
        cases.append(spec)
    assert [r["status"] for r in evaluate(paged_plan, cases)] == [2, 2, 2]
    spec = broadcast_geometry(batch=0, total=0, width=I64_MAX)
    assert evaluate(paged_plan, [spec])[0]["status"] == 2
    spec = tnd_static_case(batch=0, total=0, pool=0, width=0)
    for name in ("q", "qv", "k", "v", "table"):
        spec[name]["strides"][0] = I64_MAX
        spec[name]["offset"] = I64_MAX
        spec[name]["storage"] = 0
    check_static(spec, evaluate(paged_plan, [spec])[0])


@pytest.mark.parametrize("dtype", (1, 2))
def test_reserved_workspace_does_not_relax_empty_payload(
    paged_plan, paged_builder, paged_consumer, dtype
):
    spec = tnd_static_case(dtype=dtype)
    plan = evaluate(paged_plan, [spec])[0]
    item = schedule_item(spec, lengths=[0, 0, 0])
    row = run_builder(paged_builder, [item], dtype)[0]
    assert plan["workspace"] > 0
    check_consumed(paged_consumer, row, plan, item)
    for index in (15, 16, 20, *range(23, 401)):
        bad = list(row)
        bad[index] += 1
        assert decode(paged_consumer, bad, context_for(plan), item)[0] != 0, index
