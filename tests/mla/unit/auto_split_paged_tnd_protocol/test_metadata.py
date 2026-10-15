"""v26 current-length/page validation, independent queries and corrupt records."""

import math
import itertools

import pytest

from .test_plan import PAIRS, CORNERS, allocations, arrays, build, config


def request(
    c=None,
    *,
    uq=None,
    kv=None,
    cu=None,
    counts=None,
    absent=(),
    window=(-1, -1, 0),
    dtype=1,
    mode=0,
    mutation=-1,
    value=0,
    task=0,
    split=0,
    row=0,
    merge=(0, 0),
):
    c = c or config()
    return [
        0,
        *c,
        *window,
        dtype,
        mode,
        mutation,
        value,
        task,
        split,
        row,
        *merge,
        *arrays(c, uq, kv, cu, counts, absent),
    ]


def check_protocol(drivers, c, uq, kv, window, dtype, caps=None):
    caps = allocations(c) if caps is None else caps
    cu = [0, *itertools.accumulate(caps)]
    plan = build(drivers[0], c, uq=uq, kv=kv, cu=cu, window=window)
    assert plan["status"] == 0, plan
    p = plan["plan"]
    b, q, k, page, h, ratio, cores = c[:7]
    targets = sorted({0, max(0, p["groups"] - 1), p["groups"] // 2, p["groups"], 2**64 - 1})
    requests = []
    expected = []
    for g in targets:
        t = drivers[0].evaluate([[2, g, 0]])[0]
        splits = sorted({0, t["task"][5] - 1, t["task"][5]}) if not t["status"] else [0]
        for split in splits:
            t = drivers[0].evaluate([[2, g, split]])[0]
            row = max(0, t["whole"][1] - t["whole"][0] - 1) if not t["status"] else 0
            r = drivers[0].evaluate([[4, g, split, row]])[0]
            requests.append(
                request(
                    c, uq=uq, kv=kv, cu=cu, window=window, dtype=dtype, task=g, split=split, row=row
                )
            )
            expected.append((t, r))
    results = drivers[1].evaluate(requests)
    for got, (t, r) in zip(results, expected):
        assert got["built"] and got["generated"] == got["read"] == 0, got
        assert got["groups"] == p["groups"] and got["slots"] == p["slots"]
        assert (
            got["bytes"] == p["layout"]["buffers"][3]
            and got["budget"] == plan["budget"]["bounds"][4]
        )
        assert got["bases"] == plan["tiling"][3:]
        assert got["page_reads"] == sum(math.ceil(v / page) for v in kv)
        record = got["record"]
        base = 408 + 5 * b
        assert len(record) == 756 + 7 * b and record[1:3] == [26, 26]
        assert [
            record[base + 346 + 2 * i] + 2**32 * record[base + 347 + 2 * i] for i in range(b + 1)
        ] == p["prefix"]
        for bi in range(b):
            start = cu[bi]
            assert record[408 + 5 * bi : 413 + 5 * bi] == [
                start % 2**32,
                start // 2**32,
                caps[bi],
                caps[bi] if uq is None else uq[bi],
                kv[bi],
            ]
        if t["status"]:
            assert not got["task"] and got["task_unchanged"] and got["row_unchanged"]
        else:
            assert got["task"] == t["task"]
            assert got["range"] == t["whole"][:2] + t["fragment"][:4]
            assert got["offsets"] == t["offsets"]
            assert got["row_ok"] and got["row"] == r["row"]
    fd = p["native"][2]
    merges = {(fd, 0), (0, 2**32 - 1)}
    if fd:
        merges |= {(0, 0), (fd - 1, p["arrays"][7][fd - 1] - 1)}
    for pair in sorted(merges):
        got = drivers[1].evaluate(
            [request(c, uq=uq, kv=kv, cu=cu, window=window, dtype=dtype, merge=pair, row=2**32 - 1)]
        )[0]
        expect = drivers[0].evaluate([[3, *pair]])[0]
        assert got["read"] == 0 and not got["row_ok"] and got["row_unchanged"]
        assert got["merge_ok"] == int(expect["status"] == 0)
        if expect["status"]:
            assert got["merge_unchanged"]
        else:
            assert got["merge"] == expect["merge"] and got["merge_offsets"] == expect["offsets"]
    return results[0]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("page,window,cores", PAIRS)
def test_protocol_pairs(drivers, dtype, page, window, cores):
    check_protocol(
        drivers,
        config(p=page, k=math.ceil(1057 / page) * page, cores=cores, total=22),
        [13, 3],
        [1057, 1],
        window,
        dtype,
    )


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("name,options,uq,kv,window", CORNERS, ids=[c[0] for c in CORNERS])
def test_protocol_corners(drivers, dtype, name, options, uq, kv, window):
    check_protocol(drivers, config(**options), uq, kv, window, dtype)


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize(
    "options,kwargs",
    [
        ({"ratio": 1}, {}),
        ({"ratio": 3}, {}),
        ({"p": 17}, {}),
        ({"p": 0}, {}),
        ({"k": 111}, {}),
        ({"k": 16464}, {}),
        ({"q": -1}, {}),
        ({"q": 2**30 + 1}, {}),
        ({"total": 35}, {}),
        ({"total": 2**31}, {}),
        ({"layout": 0}, {}),
        ({"memory": 1}, {}),
        ({"lib": 2**64 - 1}, {}),
        ({}, {"uq": [-1, 0]}),
        ({}, {"uq": [18, 0]}),
        ({}, {"kv": [1121, 0]}),
        ({}, {"kv": [-1, 0]}),
        ({}, {"counts": [0, 0, 1]}),
        ({}, {"absent": [2]}),
        ({}, {"cu": [0, 18, 34]}),
        ({}, {"cu": [1, 17, 34]}),
        ({}, {"cu": [0, 17, 33]}),
        ({}, {"absent": [0]}),
        ({"q": 0}, {"kv": [0, -1]}),
        ({"k": 0}, {"uq": [18, 0]}),
        ({}, {"window": (-(2**63), -1, 0)}),
    ],
)
def test_generation_failures_clear_payload(drivers, dtype, options, kwargs):
    got = drivers[1].evaluate([request(config(**options), dtype=dtype, **kwargs)])[0]
    assert got["built"] and got["generated"] and got["cleared"], got
    assert got["read"] and got["unchanged"] and got["groups"] == 12345


@pytest.mark.parametrize("dtype", [0, 3, 2**31 - 1])
def test_bad_dtype(drivers, dtype):
    got = drivers[1].evaluate([request(dtype=dtype)])[0]
    assert got["generated"] and got["read"] and got["cleared"] and got["unchanged"]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("mode", [*range(1, 20), 22, 23, 24, 26, 27, 28, 29, 30, 31, 34, 35])
def test_descriptors_live_lengths_pages_and_budgets(drivers, dtype, mode):
    got = drivers[1].evaluate([request(uq=[13, 3], kv=[1057, 1], dtype=dtype, mode=mode)])[0]
    assert got["read"] and got["unchanged"] and got["groups"] == 12345, got
    if mode in (1, 17, 18, 19):
        assert not got["built"] and got["intact"]
    # Descriptor/header failures must precede page reads.
    if mode in (1, 5, 6, 7, 8, 11, 12, 13, 14, 15, 17, 18, 19, 22, 23, 24, 26, 30, 31):
        assert got["page_reads"] == 0


CORRUPT = [
    (0, 0),
    (1, 6),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 22),
    (2, 15),
    (3, 0),
    (4, 2),
    (5, 1),
    (6, 1),
    (7, 1),
    (8, 3),
    (9, 0),
    (10, 2),
    (11, 16),
    (12, 16),
    (13, 64),
    (14, 512),
    (15, 0),
    (16, 0),
    (17, 0),
    (18, 2**32 - 1),
    (44, 2**32 - 1),
    (70, 1),
    (401, 0),
    (402, 1),
    (403, 1),
    (404, 18),
    (405, 0),
    (406, 128),
    (407, 1008),
    (408, 1),
    (409, 1),
    (410, 18),
    (411, 14),
    (412, 1056),
    (1, 23),
    (1, 24),
    (1, 25),
]


@pytest.mark.parametrize("mutation,value", CORRUPT)
def test_corrupt_record(drivers, mutation, value):
    got = drivers[1].evaluate([request(uq=[13, 3], kv=[1057, 1], mutation=mutation, value=value)])[
        0
    ]
    assert got["read"] and got["unchanged"], (mutation, value, got)


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize(
    "options,uq,kv",
    [
        ({"q": 0}, [0, 0], [1120, 1]),
        ({}, [0, 0], [1120, 1]),
        ({"q": 257, "h": 1}, [257, 1], [1, 1]),
    ],
)
def test_pages_checked_when_queries_are_empty_or_masked(drivers, dtype, options, uq, kv):
    got = drivers[1].evaluate(
        [request(config(**options), uq=uq, kv=kv, window=(-1, -1, 1), dtype=dtype, mode=9)]
    )[0]
    assert got["read"] and got["page_reads"] > 0 and got["unchanged"]


@pytest.mark.parametrize("mode", [32, 33])
def test_page_mapping_and_strides_are_invocation_state(drivers, mode):
    args = dict(uq=[13, 3], kv=[1057, 1], window=(19, 1, 0))
    a, b = drivers[1].evaluate([request(**args), request(mode=mode, **args)])
    assert a["read"] == b["read"] == 0 and a["record"] == b["record"]
    assert a["task"] == b["task"] and a["page_reads"] == b["page_reads"]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("window", [(-1, -1, 0), (-1, -1, 1), (530, 1, 0)])
def test_representative(drivers, dtype, window):
    check_protocol(drivers, config(p=528, k=1584), [13, 3], [1057, 1], window, dtype)


def test_capacity_cost_boundary(drivers):
    c = config(b=32768, q=65535, k=16, p=16, h=1024)
    group = c[0] * math.ceil(c[1] * c[4] / 256) - 1
    got = drivers[1].evaluate([request(c, kv=[0] * c[0], mode=20, task=group, row=255)])[0]
    assert got["generated"] == got["read"] == 0, got
    assert got["groups"] == group + 1 > 2**32 and got["task"][2] == c[0] - 1
    assert got["row_ok"] and got["row"][0] > 2**32


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("capacity", [0, 16])
def test_consumer_capacity_cost_boundary_accumulator_overflow(drivers, dtype, capacity):
    b, q = 65536, 32767
    h = (2**63 - 1) // (b * q * 26 * 512 * 4) + 1
    c = config(b=b, q=q, k=capacity, p=16, h=h)
    got = drivers[1].evaluate([request(c, uq=[0] * b, kv=[0] * b, mode=25, dtype=dtype)])[0]
    assert got["generated"] == 0 and got["read"] and got["unchanged"], got


# Independent fields, inactive arrays, wide prefixes, FD and Vector endpoints.
TAIL_WORDS = [3, 82, 83, 84, 85, *range(346, 352)]
for start, size in (
    (4, 26),
    (30, 26),
    (56, 26),
    (86, 26),
    (112, 26),
    (138, 26),
    (164, 26),
    (190, 26),
    (216, 26),
    (242, 52),
    (294, 52),
):
    TAIL_WORDS.extend([start, start + size - 1])


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("word", TAIL_WORDS)
def test_corrupt_native_partition_tail(drivers, dtype, word):
    c = config(p=16, k=16384)
    got = drivers[1].evaluate(
        [request(c, uq=[13, 3], kv=[16383, 1], dtype=dtype, mutation=418 + word, value=2**32 - 1)]
    )[0]
    assert got["read"] and got["unchanged"], got


@pytest.mark.parametrize("word", range(367, 401))
def test_each_source_workspace_and_local_memory_word(drivers, word):
    baseline = drivers[1].evaluate([request()])[0]
    assert baseline["read"] == 0
    got = drivers[1].evaluate([request(mutation=word, value=baseline["record"][word] ^ 1)])[0]
    assert got["read"] and got["unchanged"]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("word", [15, 16, 23, 367, 391, 418, 421, 422, 503, 764, 769])
def test_empty_record_checks_inactive_fields_and_tail(drivers, dtype, word):
    c = config(k=0)
    baseline = drivers[1].evaluate([request(c, dtype=dtype)])[0]
    assert baseline["read"] == 0
    got = drivers[1].evaluate(
        [request(c, dtype=dtype, mutation=word, value=baseline["record"][word] ^ 1)]
    )[0]
    assert got["read"] and got["unchanged"]


def test_wire_word_count_boundary(drivers):
    limit = (2**32 - 1 - 756) // 7
    for b in (0, 1, limit, limit + 1, -1, 2**63 - 1):
        got = drivers[1].evaluate([[1, b]])[0]
        valid = 0 <= b <= limit
        assert bool(got["ok"]) == valid
        assert got["words"] == (756 + 7 * b if valid else 12345)


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("window", [(-1, -1, 0), (-1, -1, 1), (530, 1, 0)])
def test_nonuniform_empty_prefix_and_used(drivers, dtype, window):
    caps = [0, 65, 0, 7, 2]
    c = config(b=5, q=97, total=sum(caps), k=16384, p=512, h=8)
    check_protocol(drivers, c, [0, 63, 0, 1, 2], [17, 16383, 1, 0, 513], window, dtype, caps)
