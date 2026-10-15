"""BSND PA host plan: full views and static budgets, without reading device data.

Pinned CANN tiling supplies independent schedules/workspace requirements. The
existing v6 consumer receives the new plan's context/cache descriptor, never
a context inferred from the record. This does not launch public PA attention.
MLA_PAGED_PLAN_SANITIZE=1 instruments the same static input matrix with ASan/UBSan.
"""

from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla.support.mla_metadata_build import MLA, ROOT, sha256
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_layout import view
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_metadata import (
    paged_artifacts as paged_artifacts,
    paged_builder as paged_builder,
    paged_oracles as paged_oracles,
    check_source,
    run_builder,
)
from tests.mla.support.components.paged_metadata import expected_record, paged_case
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_metadata_consumer import (
    paged_consumer as paged_consumer,
    decode,
)
from tests.mla.support.components.paged_metadata_consumer import (
    CacheLayout,
    Context,
    Memory,
    Query,
    TensorLayout,
    expected_schedule,
)

I64_MAX = (1 << 63) - 1
U32_MAX = (1 << 32) - 1
MAX_BATCH = (U32_MAX - 408) // 5
LOCAL_MIN = [516096, 174080, 65536, 65536, 131072]


@pytest.fixture(scope="module")
def paged_plan(tmp_path_factory):
    target = os.environ.get("MLA_PAGED_PLAN_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-paged-plan")
    directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "tests/mla/support/mla_paged_plan_driver.cpp"] + [
        MLA / name
        for name in (
            "mla_paged_plan.cpp",
            "mla_paged_layout.cpp",
            "mla_bsnd_layout.cpp",
            "mla_tnd_layout.cpp",
            "mla_schedule.cpp",
            "detail/cann_split_core.cpp",
        )
    ]
    binary = directory / "plan"
    command = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O2",
        "-g",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-D_GLIBCXX_ASSERTIONS",
        f"-I{MLA}",
        *map(str, sources),
        "-o",
        str(binary),
    ]
    sanitize = os.environ.get("MLA_PAGED_PLAN_SANITIZE") == "1"
    if sanitize:
        command += [
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer",
        ]
    (directory / "compile.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    tracked = [*sources, *MLA.glob("*.h"), *MLA.glob("detail/*.h"), Path(__file__)]
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                binary=str(binary.resolve()),
                binary_sha256=sha256(binary),
                sanitizer=sanitize,
                sources={str(p.relative_to(ROOT)): sha256(p) for p in tracked},
            ),
            indent=2,
        )
        + "\n"
    )
    return binary


def static_case(page=112, batch=3, sq=7, heads=128, width=5, pool=7, dtype=1, used=False):
    def query(dim, padding, offset):
        row = heads * (dim + padding) + 11
        return view((batch, sq, heads, dim), (sq * row + 17, row, dim + padding, 1), offset)

    return dict(
        q=query(64, 3, 7),
        qv=query(512, 5, 13),
        k=view((pool, page, 1, 64), (page * 67 + 19, 67, 79, 1), 17),
        v=view((pool, page, 1, 512), (page * 521 + 31, 521, 547, 1), 23),
        table=view((batch, width), (width + 3, 1), 3, 4),
        platform=[20, 40, 524288, 196608, 65536, 65536, 131072, 16 * 1024 * 1024],
        blocks=list(range(1, 27)),
        dtype=dtype,
        used=used,
    )


def evaluate(binary, cases):
    tnd = "max_q" in cases[0]
    assert all(("max_q" in item) == tnd for item in cases)
    rows = []
    for item in cases:
        row = [item["dtype"], int(item["used"])]
        if tnd:
            row += [item["max_q"]]
        for name in ("q", "qv", "k", "v", "table"):
            t = item[name]
            row += [*t["sizes"], *t["strides"], t["offset"], t["storage"]]
        rows.append([*row, *item["platform"], *item["blocks"]])
    stdin = "".join(" ".join(map(str, row)) + "\n" for row in rows)
    command = [str(binary)] + (["--tnd"] if tnd else [])
    result = subprocess.run(command, input=stdin, capture_output=True, text=True, timeout=30)
    calls = binary.parent / "calls"
    calls.mkdir(exist_ok=True)
    (calls / f"{len(list(calls.glob('*.json'))):04d}.json").write_text(
        json.dumps(
            dict(
                command=command,
                inputs=cases,
                stdin=stdin,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
        )
        + "\n"
    )
    assert result.returncode == 0 and not result.stderr, result.stderr
    results = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(results) == len(cases)
    assert all(r["unchanged"] for r in results if r["status"])
    return results


def expected_layout(t):
    sizes, strides = t["sizes"], t["strides"]
    # Unbounded Python arithmetic, independent of the checked C++ builders.
    span = 0 if 0 in sizes else 1 + sum((n - 1) * s for n, s in zip(sizes, strides))
    return [*strides[:-1], span]


def check_static(item, plan):
    assert plan["status"] == 0, (item, plan)
    tnd = "max_q" in item
    if tnd:
        total, heads, _ = item["q"]["sizes"]
        batch, sq = item["table"]["sizes"][0], item["max_q"]
    else:
        batch, sq, heads, _ = item["q"]["sizes"]
        total = batch * sq
    pool, page = item["k"]["sizes"][:2]
    width = item["table"]["sizes"][1]
    capacity = width * page
    blocks = item["platform"][0] if batch and total and capacity else 1
    assert plan["shape"] == [batch, heads, int(tnd), total, sq, capacity, page]
    assert plan["words"] == 408 + 5 * batch <= U32_MAX
    assert plan["blocks"] == [blocks] * 26
    assert plan["platform"] == item["platform"]
    context = plan["context"]
    assert context[:13] == [
        item["dtype"],
        batch,
        heads,
        int(item["used"]),
        int(tnd),
        total & U32_MAX,
        total >> 32,
        sq,
        page,
        capacity,
        *item["platform"][:2],
        blocks,
    ]
    assert context[15:] == item["platform"][2:7]
    assert 0 <= plan["workspace"] == sum(context[13:15]) <= I64_MAX
    if not (batch and total and capacity):
        assert plan["workspace"] == 0
    else:
        assert context[13] > 0 and context[14] == item["platform"][-1]
    layouts = [expected_layout(item[name]) for name in ("q", "qv", "k", "v")]
    if tnd:
        layouts[:2] = [[0, *layout] for layout in layouts[:2]]
    assert plan["layouts"] == layouts
    row_stride, span = expected_layout(item["table"])
    assert plan["cache"] == [batch, width, capacity, pool, page, row_stride, span, *layouts[2:]]


def context_for(plan):
    c = plan["context"]
    return Context(*c[:4], Query(*c[4:8]), *c[8:15], Memory(*c[15:]))


def schedule_item(spec, lengths, used=None):
    batch, sq, heads = spec["q"]["sizes"][:3]
    page = spec["k"]["sizes"][1]
    p = spec["platform"]
    return paged_case(
        0,
        [sq] * batch,
        lengths,
        (used if used is not None else [sq // 2] * batch) if spec["used"] else None,
        page,
        spec["table"]["sizes"][1] * page,
        sq=sq,
        heads=heads,
        aic=p[0],
        aiv=p[1],
        block_dim=p[0] if batch and sq and spec["table"]["sizes"][1] else 1,
        l1=p[2],
        ub=p[3],
        l0a=p[4],
        l0b=p[5],
        l0c=p[6],
        libapi=p[7],
    )


def check_consumed(consumer, row, plan, item, *, pool=None):
    p = plan["cache"]
    cache = CacheLayout(*p[:7], TensorLayout(*p[7]), TensorLayout(*p[8]))
    spec = dict(
        pool=p[3] if pool is None else pool, row_stride=p[5], k_strides=p[7][:3], v_strides=p[8][:3]
    )
    result, output, _ = decode(
        consumer, row, context_for(plan), item, spec=spec, cache_changes={"layout": cache}
    )
    assert result == 0, (item, plan, result)
    assert output == expected_schedule(row)
    for index in (383, 385):  # Independently serialized kernel/user byte requirements.
        assert row[index] | row[index + 1] << 32 <= plan["context"][13]
    assert row[389] | row[390] << 32 <= plan["workspace"]


@pytest.mark.parametrize("page,dtype", itertools.product(range(16, 1025, 16), (1, 2)))
def test_source_budgets_and_context(
    paged_plan, paged_builder, paged_oracles, paged_consumer, page, dtype
):
    for used in (False, True):
        spec = static_case(page, width=16384 // page, dtype=dtype, used=used)
        plan = evaluate(paged_plan, [spec])[0]
        check_static(spec, plan)
        items = [
            schedule_item(spec, lengths)
            for length in sorted({1, page - 1, page, page + 1, 449, 513, 16384 // page * page})
            for lengths in ([length] * 3, [length, 0, min(length, 17)])
        ]
        rows = check_source(
            paged_builder, paged_oracles, items, dtype, f"plan-{page}-{dtype}-{used}"
        )
        for item, row in zip(items, rows):
            check_consumed(paged_consumer, row, plan, item)


@pytest.mark.parametrize("aic", range(1, 27))
def test_core_counts_and_m_base_change(
    paged_plan, paged_builder, paged_oracles, paged_consumer, aic
):
    # For even core counts, equal KV can choose M=128 while unequal KV selects
    # M=256. Both records must fit the same static allocation and full launch.
    spec = static_case(page=528, batch=max(2, aic // 2), sq=2, heads=128, width=4)
    spec["platform"][:2] = [aic, 2 * aic]
    plan = evaluate(paged_plan, [spec])[0]
    check_static(spec, plan)
    batch = spec["q"]["sizes"][0]
    items = [
        schedule_item(spec, lengths) for lengths in ([1025] * batch, [1025] + [17] * (batch - 1))
    ]
    rows = check_source(paged_builder, paged_oracles, items, 1, f"cores-{aic}")
    if aic >= 4 and aic % 2 == 0:
        assert [row[15] for row in rows] == [128, 256]
        assert rows[1][385] > rows[0][385]
    for item, row in zip(items, rows):
        check_consumed(paged_consumer, row, plan, item)


@pytest.mark.parametrize(
    "page,width,query", itertools.product((16, 112, 528, 1024), (1, 2), ((1, 1), (1, 17), (3, 43)))
)
def test_short_capacity_budgets(
    paged_plan, paged_builder, paged_oracles, paged_consumer, page, width, query
):
    sq, heads = query
    spec = static_case(page=page, batch=2, sq=sq, heads=heads, width=width)
    plan = evaluate(paged_plan, [spec])[0]
    check_static(spec, plan)
    items = [schedule_item(spec, [n, n]) for n in (1, page * width)]
    rows = check_source(
        paged_builder, paged_oracles, items, 1, f"short-{page}-{width}-{sq}-{heads}"
    )
    for item, row in zip(items, rows):
        check_consumed(paged_consumer, row, plan, item)


@pytest.mark.parametrize(
    "dtype,used,empty",
    itertools.product((1, 2), (False, True), ("batch", "q", "table", "pool", "lengths", "used")),
)
def test_empty_allocations_and_dynamic_lengths(
    paged_plan, paged_builder, paged_consumer, dtype, used, empty
):
    kwargs = (
        {"batch": 0}
        if empty == "batch"
        else {"sq": 0}
        if empty == "q"
        else {"width": 0}
        if empty == "table"
        else {"pool": 0}
        if empty == "pool"
        else {}
    )
    spec = static_case(dtype=dtype, used=used, **kwargs)
    plan = evaluate(paged_plan, [spec])[0]
    check_static(spec, plan)
    batch = spec["q"]["sizes"][0]
    lengths = [17] * batch if empty in ("q", "used") else [0] * batch
    item = schedule_item(spec, lengths, used=[0] * batch)
    row = run_builder(paged_builder, [item], dtype)[0]
    if empty != "used":
        assert row == expected_record(item, dtype, None)
        assert row[21] == plan["context"][12]
        assert row[383:391] == [0] * 8
    check_consumed(paged_consumer, row, plan, item)


@pytest.mark.parametrize("name", ("q", "qv", "k", "v", "table"))
@pytest.mark.parametrize("profile", ("broadcast", "overlap", "offset", "wide"))
def test_independent_views(paged_plan, name, profile):
    spec = static_case()
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


def broadcast_geometry(**kwargs):
    spec = static_case(**kwargs)
    for name in ("q", "qv", "k", "v", "table"):
        sizes = spec[name]["sizes"]
        spec[name] = view(sizes, [0] * (len(sizes) - 1) + [1], 0, 4 if name == "table" else 2)
    return spec


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch=MAX_BATCH, sq=0),
        dict(batch=5, sq=1 << 30, heads=1),
        dict(batch=1, sq=1 << 30, heads=3),
        dict(batch=1, sq=1, heads=U32_MAX // 512),
        dict(pool=1 << 31),
        dict(pool=100000, page=1024, width=1),
    ],
)
def test_wide_shape_fields_without_allocating_per_batch(paged_plan, kwargs):
    spec = broadcast_geometry(**kwargs)
    check_static(spec, evaluate(paged_plan, [spec])[0])


@pytest.mark.parametrize("name", ("q", "qv", "k", "v", "table"))
@pytest.mark.parametrize("fault", ("stride", "last", "storage", "offset", "bytes"))
def test_complete_storage_rejected(paged_plan, name, fault):
    spec = static_case()
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
    "kwargs",
    [
        dict(batch=MAX_BATCH + 1, sq=0),
        dict(batch=5, sq=1 << 30, heads=4),
        dict(batch=MAX_BATCH, sq=1 << 30, heads=1),
        dict(sq=(1 << 30) + 1),
        dict(heads=U32_MAX // 512 + 1),
        dict(heads=0),
        dict(page=1024, width=17),
    ],
)
def test_schedule_shape_bounds(paged_plan, kwargs):
    spec = broadcast_geometry(**kwargs)
    assert evaluate(paged_plan, [spec])[0]["status"] == 1


def test_empty_table_capacity_multiplication_overflow(paged_plan):
    spec = broadcast_geometry(batch=0, width=I64_MAX)
    assert evaluate(paged_plan, [spec])[0]["status"] == 2


@pytest.mark.parametrize(
    "field,value,status",
    [
        ("dtype", 0, 3),
        ("dtype", 3, 3),
        ("q.sizes.0", -1, 1),
        ("q.sizes.1", -1, 1),
        ("q.sizes.3", 512, 1),
        ("qv.sizes.0", 4, 1),
        ("qv.sizes.1", 8, 1),
        ("qv.sizes.2", 64, 1),
        ("qv.sizes.3", 64, 1),
        ("table.sizes.0", 4, 1),
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
    spec = static_case()
    fields = field.split(".")
    target = spec
    for key in fields[:-1]:
        target = target[int(key) if isinstance(target, list) else key]
    target[int(fields[-1]) if isinstance(target, list) else fields[-1]] = value
    assert evaluate(paged_plan, [spec])[0]["status"] == status


@pytest.mark.parametrize("empty", (False, True))
def test_local_resources_and_workspace_boundary(paged_plan, empty):
    spec = static_case(sq=0 if empty else 7)
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


def test_empty_query_still_validates_cache_and_table(paged_plan):
    cases = []
    for name in ("k", "v", "table"):
        spec = static_case(sq=0)
        spec[name]["storage"] -= 1
        cases.append(spec)
    assert [r["status"] for r in evaluate(paged_plan, cases)] == [2, 2, 2]
    spec = static_case(batch=0, sq=0, pool=0, width=0)
    for name in ("q", "qv", "k", "v", "table"):
        spec[name]["strides"][0] = I64_MAX
        spec[name]["offset"] = I64_MAX
        spec[name]["storage"] = 0
    check_static(spec, evaluate(paged_plan, [spec])[0])


@pytest.mark.parametrize("dtype", (1, 2))
def test_reserved_workspace_does_not_relax_empty_payload(
    paged_plan, paged_builder, paged_consumer, dtype
):
    spec = static_case(dtype=dtype)
    plan = evaluate(paged_plan, [spec])[0]
    item = schedule_item(spec, [0, 0, 0])
    row = run_builder(paged_builder, [item], dtype)[0]
    assert plan["workspace"] > 0
    check_consumed(paged_consumer, row, plan, item)
    for index in (15, 16, 20, *range(23, 401)):
        bad = list(row)
        bad[index] += 1
        status, _, _ = decode(paged_consumer, bad, context_for(plan), item)
        assert status != 0, index
