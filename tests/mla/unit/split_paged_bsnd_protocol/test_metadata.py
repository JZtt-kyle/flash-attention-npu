"""v23 current-length/page validation, independent queries and corrupt records."""

import math

import pytest

from .test_plan import PAIRS, arrays, build, config


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
        *arrays(c, uq, kv, cu, counts, absent),
    ]


def check_protocol(drivers, c, uq, kv, window, dtype):
    plan = build(drivers[0], c, uq=uq, kv=kv, window=window)
    assert plan["status"] == 0, plan
    p = plan["plan"]
    b, q, k, page, h, s, cores = c[:7]
    targets = sorted({0, max(0, p["tasks"] - 1), p["tasks"] // 2, p["tasks"], 2**64 - 1})
    results = drivers[1].evaluate(
        [request(c, uq=uq, kv=kv, window=window, dtype=dtype, task=t) for t in targets]
    )
    for t, got in zip(targets, results):
        assert got["built"] and got["generated"] == got["read"] == 0, got
        record = got["record"]
        assert len(record) == 414 + 7 * b and record[1:3] == [23, 23]
        assert got["groups"] == p["groups"] and got["tasks"] == p["tasks"]
        assert (
            got["bytes"] == p["layout"]["buffers"][3]
            and got["budget"] == plan["budget"]["bounds"][4]
        )
        assert got["bases"] == plan["tiling"][3:]
        assert got["page_reads"] == sum(math.ceil(v / page) for v in kv)
        for bi in range(b):
            start = bi * q
            assert record[408 + 5 * bi : 413 + 5 * bi] == [
                start % 2**32,
                start // 2**32,
                q,
                q if uq is None else uq[bi],
                kv[bi],
            ]
        base = 408 + 5 * b
        assert [
            record[base + 4 + 2 * i] + 2**32 * record[base + 5 + 2 * i] for i in range(b + 1)
        ] == p["prefix"]
        if t >= p["tasks"]:
            assert got["task"] == [] and got["task_unchanged"]
        else:
            task = drivers[0].evaluate([[2, t]])[0]
            # This compares independently implemented producer and reader
            # ownership; test_plan separately verifies all coordinates.
            assert (
                got["task"]
                == task["task"] + task["whole"][:2] + task["fragment"][:2] + task["offsets"]
            )
    return results[0]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("page,window,splits", PAIRS)
def test_protocol_pairs(drivers, dtype, page, window, splits):
    c = config(p=page, k=math.ceil(1057 / page) * page, s=splits)
    check_protocol(drivers, c, [13, 3], [1057, 1], window, dtype)


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize(
    "options,uq,kv,window",
    [
        ({"b": 0}, [], [], (-1, -1, 0)),
        ({"q": 0}, [0, 0], [1120, 1], (-1, -1, 1)),
        ({"k": 0}, [17, 0], [0, 0], (1, 0, 0)),
        ({}, [0, 0], [1120, 1], (0, 0, 0)),
        ({}, [17, 3], [0, 0], (-1, -1, 1)),
        ({}, None, [1120, 1120], (-1, -1, 0)),
        ({"q": 257, "h": 1, "s": 256}, [257, 1], [1, 0], (-1, -1, 1)),
        ({"q": 208, "h": 8}, None, [1025, 1025], (-1, -1, 0)),
        ({"q": 249, "h": 13}, None, [1025, 1025], (19, 1, 0)),
        ({"b": 1, "q": 1, "p": 1024, "k": 16384, "h": 257}, None, [16384], (-1, -1, 0)),
        ({"cores": 1}, [17, 3], [1057, 1], (530, 1, 1)),
    ],
)
def test_protocol_corners(drivers, dtype, options, uq, kv, window):
    check_protocol(drivers, config(**options), uq, kv, window, dtype)


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize(
    "options,kwargs",
    [
        ({"s": 0}, {}),
        ({"s": 1}, {}),
        ({"s": 257}, {}),
        ({"p": 17}, {}),
        ({"p": 0}, {}),
        ({"k": 111}, {}),
        ({"k": 16464}, {}),
        ({"q": -1}, {}),
        ({"q": 2**30 + 1}, {}),
        ({"total": 1}, {}),
        ({"layout": 1}, {}),
        ({"memory": 1}, {}),
        ({"lib": 2**64 - 1}, {}),
        ({}, {"uq": [-1, 0]}),
        ({}, {"uq": [18, 0]}),
        ({}, {"kv": [1121, 0]}),
        ({}, {"kv": [-1, 0]}),
        ({}, {"counts": [0, 0, 1]}),
        ({}, {"absent": [2]}),
        ({}, {"cu": [0, 17, 34]}),
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
@pytest.mark.parametrize("mode", [*range(1, 20), 22, 23, 24, 26, 27, 28, 29, 30, 31])
def test_descriptors_live_lengths_pages_and_budgets(drivers, dtype, mode):
    got = drivers[1].evaluate([request(uq=[13, 3], kv=[1057, 1], dtype=dtype, mode=mode)])[0]
    assert got["read"] and got["unchanged"] and got["groups"] == 12345, got
    if mode in (1, 17, 18, 19):
        assert not got["built"] and got["intact"]
    # Descriptor/header failures must precede page reads.
    if mode in (1, 5, 6, 7, 8, 11, 12, 13, 14, 15, 17, 18, 19, 22, 23, 24, 26, 27, 30, 31):
        assert got["page_reads"] == 0


# Header, native geometry/resources/endpoints, logical allocation/used/P/C,
# mask/splits and every low/high compact-M prefix endpoint.
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
    (401, 1),
    (402, 0),
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
    (418, 2),
    (419, 0),
    (420, 0),
    (421, 2),
    (422, 1),
    (423, 1),
    (424, 0),
    (425, 1),
    (426, 0),
    (427, 1),
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


def test_page_mapping_is_invocation_state(drivers):
    args = dict(uq=[13, 3], kv=[1057, 1], window=(19, 1, 0))
    a, b = drivers[1].evaluate([request(**args), request(mode=32, **args)])
    assert a["read"] == b["read"] == 0 and a["record"] == b["record"]
    assert a["task"] == b["task"] and a["page_reads"] == b["page_reads"]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("window", [(-1, -1, 0), (-1, -1, 1), (530, 1, 0)])
def test_representative(drivers, dtype, window):
    check_protocol(drivers, config(p=528, k=1584), [13, 3], [1057, 1], window, dtype)


def test_capacity_cost_boundary(drivers):
    c = config(b=65538, q=65536, k=16, p=16, h=1, s=256)
    task = 65538 * 256 * 256 - 1
    got = drivers[1].evaluate([request(c, kv=[0] * c[0], mode=20, task=task)])[0]
    assert got["generated"] == got["read"] == 0, got
    assert got["tasks"] == task + 1 > 2**32 and got["task"][0] == task
    assert got["task"][3:6] == [65537, 255, 255]


def test_consumer_accumulator_and_static_partial_overflow(drivers):
    # Generate safe S=2, present S=256 with matching header/tail. Consumer must
    # independently reject the accumulator even though current used Q is zero.
    c = config(b=16384, q=2**30, k=0, p=16, h=1, s=256)
    got = drivers[1].evaluate([request(c, uq=[0] * c[0], kv=[0] * c[0], mode=25)])[0]
    assert got["generated"] == 0 and got["read"] and got["unchanged"], got

    c = config(b=16200, q=2**30, k=16, p=16, h=1, s=256)
    got = drivers[1].evaluate([request(c, uq=[0] * c[0], kv=[0] * c[0], mode=25)])[0]
    assert got["generated"] == 0 and got["read"] and got["unchanged"], got


@pytest.mark.parametrize("word", range(367, 401))
def test_each_source_workspace_and_local_memory_word(drivers, word):
    baseline = drivers[1].evaluate([request()])[0]
    assert baseline["read"] == 0
    got = drivers[1].evaluate([request(mutation=word, value=baseline["record"][word] ^ 1)])[0]
    assert got["read"] and got["unchanged"]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("word", [15, 16, 23, 367, 391, 418, 421, 422, 427])
def test_empty_record_still_checks_inactive_fields_and_tail(drivers, dtype, word):
    c = config(k=0)
    baseline = drivers[1].evaluate([request(c, dtype=dtype)])[0]
    assert baseline["read"] == 0
    got = drivers[1].evaluate(
        [request(c, dtype=dtype, mutation=word, value=baseline["record"][word] ^ 1)]
    )[0]
    assert got["read"] and got["unchanged"]
