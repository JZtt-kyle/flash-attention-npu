"""v21 compact Q / rectangular KV protocol, independent fragment oracle and corrupt records."""

import itertools

import pytest


def request(
    qa=(7, 0, 65),
    capacity=1057,
    usedq=None,
    usedk=None,
    h=4,
    s=8,
    left=-1,
    right=-1,
    causal=0,
    dtype=1,
    mut=-1,
    value=0,
    mode=0,
    totalq=None,
    totalk=None,
):
    cq = [0, *itertools.accumulate(qa)]
    ck = [0] * (len(qa) + 1)
    q, k = max(qa, default=0), capacity
    return [
        len(qa),
        sum(qa) if totalq is None else totalq,
        len(qa) * capacity if totalk is None else totalk,
        q,
        k,
        h,
        s,
        left,
        right,
        causal,
        dtype,
        int(usedq is not None),
        int(usedk is not None),
        mut,
        value,
        mode,
        *cq,
        *ck,
        *(qa if usedq is None else usedq),
        *([capacity] * len(qa) if usedk is None else usedk),
    ]


CASES = [
    {},
    dict(left=-2),
    dict(right=-2),
    dict(s=2),
    dict(s=192),
    dict(s=193),
    dict(s=255),
    dict(s=256),
    dict(usedq=[2, 0, 63], usedk=[1025, 0, 0], left=530, right=1),
    dict(qa=[9, 5], capacity=3, usedk=[3, 0], causal=1),
    dict(left=0, right=0),
    dict(left=530),
    dict(right=1),
    dict(left=5, right=2, causal=1),
    dict(qa=[1, 1, 1, 1], capacity=3),
    dict(qa=[0, 0, 0]),
    dict(capacity=0),
    dict(qa=[]),
    dict(qa=[0], capacity=0),
    dict(usedq=[0, 0, 0]),
    dict(usedk=[0, 0, 0]),
    dict(usedq=[1, 0, 64]),
    dict(usedk=[1, 0, 512]),
]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("spec", CASES)
def test_protocol(metadata_drivers, dtype, spec):
    r = request(dtype=dtype, **spec)
    a = metadata_drivers[0].evaluate([r])[0]
    assert a["built"] and a["generated"] == a["read"] == 0, a
    b, tq, tk, q, k, h, s, left, right, causal, _, uq, uk = r[:13]
    record = a["record"]
    assert len(record) == 416 + 10 * b and record[1:3] == [21, 21]
    assert a["tasks"] == a["groups"] * s
    if not tq or not tk:
        assert a["groups"] == a["bytes"] == 0
        return
    m, n = record[15:17]
    kind, wl, wr = record[410 + 8 * b : 413 + 8 * b]
    wl, wr = (x if x < 2**31 else x - 2**32 for x in (wl, wr))
    qa = spec.get("qa", [7, 0, 65])
    qs = spec.get("usedq", qa)
    ks = spec.get("usedk", [k] * b)
    expected = []
    for bi, (ql, kl) in enumerate(zip(qs, ks)):
        span = record[410 + 8 * bi : 418 + 8 * bi]
        assert span == [sum(qa[:bi]), 0, (bi * k) % 2**32, (bi * k) // 2**32, qa[bi], k, ql, kl]
        for block in range((ql * h + m - 1) // m):
            keys = {
                j // n
                for row in range(block * m, min((block + 1) * m, ql * h))
                for j in range(kl)
                if (kind == 0 or j <= row // h + kl - ql + (wr if kind == 2 else 0))
                and (kind != 2 or j >= row // h + kl - ql - wl)
            }
            first, stop = (min(keys), max(keys) + 1) if keys else (0, 0)
            native = metadata_drivers[1].evaluate(
                [
                    [
                        0,
                        3 if kind == 1 else 4 if kind == 2 else 0,
                        2**31 - 1 if kind == 1 else wl,
                        0 if kind == 1 else wr,
                        h,
                        m,
                        n,
                        26,
                        2,
                        1,
                        ql,
                        kl,
                        block,
                    ]
                ]
            )[0]
            assert native[2:4] == [first, stop]
            cuts = metadata_drivers[2].evaluate([[1, first, stop, s, i] for i in range(s)])
            expected.extend(
                [bi, block, i, min(cut[0], stop), min(cut[1], stop)] for i, cut in enumerate(cuts)
            )
    assert a["fragments"] == expected
    assert a["groups"] == sum((ql * h + m - 1) // m for ql in qs)


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize(
    "spec",
    [
        dict(s=0),
        dict(s=1),
        dict(s=257),
        dict(usedq=[8, 0, 65]),
        dict(usedk=[1058, 3, 513]),
        dict(usedk=[-1, 3, 513]),
        dict(usedq=[-1, 0, 65]),
        dict(h=0),
        dict(h=2**23),
        dict(qa=(2**30 + 1, 0, 1)),
        dict(totalq=2**31),
        dict(capacity=16385),
        dict(totalk=1),
        dict(capacity=-1),
    ],
)
def test_invalid_generation(metadata_drivers, dtype, spec):
    a = metadata_drivers[0].evaluate([request(dtype=dtype, **spec)])[0]
    assert a["generated"] != 0 and a["read"] != 0 and a["groups"] == 12345 and a["unchanged"]


@pytest.mark.parametrize(
    "mutation",
    [
        (1, 19),
        (2, 19),
        (1, 20),
        (1, 17),
        (2, 18),
        (3, 0),
        (5, 1),
        (15, 0),
        (16, 0),
        (410, 1),
        (412, 1),
        (414, 8),
        (416, 8),
        (434, 3),
        (435, 0),
        (437, 2),
        (438, 1),
        (439, 1),
        (440, 0),
        (441, 2**32 - 1),
        (442, 2**32 - 1),
    ],
)
def test_corruption(metadata_drivers, mutation):
    a = metadata_drivers[0].evaluate([request(mut=mutation[0], value=mutation[1])])[0]
    assert a["read"] != 0 and a["groups"] == 12345 and a["unchanged"]


@pytest.mark.parametrize("mode", [*range(1, 20), 22, 23, 24])
def test_alias_budget_and_live_lengths(metadata_drivers, mode):
    a = metadata_drivers[0].evaluate([request(mode=mode, usedk=[1057, 3, 513])])[0]
    assert a["read"] != 0 and a["groups"] == 12345 and a["unchanged"]
    if mode in [1, 17, 18, 19]:
        assert not a["built"] and a["intact"]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("mask", [(-1, -1, 0), (-1, -1, 1), (530, 1, 0)])
def test_representative_protocol(metadata_drivers, dtype, mask):
    test_protocol(
        metadata_drivers,
        dtype,
        dict(usedq=[2, 0, 63], usedk=[1025, 0, 17], left=mask[0], right=mask[1], causal=mask[2]),
    )


def test_capacity_cost_boundary_wide_serialized_kv(metadata_drivers):
    b, capacity = 262146, 16384
    qa = [0] * (b - 2) + [1, 1]
    got = metadata_drivers[0].evaluate([request(qa=qa, capacity=capacity, s=2, h=1, mode=20)])[0]
    assert got["built"] and got["generated"] == got["read"] == 0
    assert got["groups"] == 2 and got["tasks"] == 4
    assert got["last_span"] == [1, 0, capacity, 1, 1, capacity, 1, capacity]
    assert got["tail"] == [1, 0, 2, 0]


@pytest.mark.parametrize("capacity", [0, 1])
def test_accumulator_capacity_checked_with_zero_used_and_empty_kv(metadata_drivers, capacity):
    # Generate with S=2, then update the valid invocation/record to S=256.
    # This independently checks consumer overflow with no tasks to hide it.
    h, s = 2**23 - 1, 256
    limit = (2**63 - 1) // (h * s * 512 * 4)
    for total, accepted in [(limit, True), (limit + 1, False)]:
        qa = [512] * (total // 512) + ([total % 512] if total % 512 else [])
        spec = dict(
            qa=qa, capacity=capacity, h=h, s=s, usedq=[0] * len(qa), usedk=[0] * len(qa), mode=21
        )
        got = metadata_drivers[0].evaluate([request(**spec)])[0]
        assert got["generated"] == 0
        # Nonempty KV additionally reserves padded persistent slots, so the
        # workspace may overflow before the accumulator's exact byte limit.
        if not capacity and accepted:
            assert got["read"] == 0 and got["groups"] == 0
        elif not accepted:
            assert got["read"] != 0 and got["unchanged"]
        spec["mode"] = 20
        built = metadata_drivers[0].evaluate([request(**spec)])[0]
        assert (built["generated"] == 0) == (got["read"] == 0)


@pytest.mark.parametrize(
    "field,version", [(field, v) for field in [1, 2] for v in [5, 14, 17, 18, 19, 20]]
)
def test_other_protocols_are_rejected(metadata_drivers, field, version):
    got = metadata_drivers[0].evaluate([request(mut=field, value=version)])[0]
    assert got["generated"] == 0 and got["read"] != 0 and got["unchanged"]


@pytest.mark.parametrize("dtype", [0, 3, 2**31 - 1])
def test_invalid_dtype(metadata_drivers, dtype):
    got = metadata_drivers[0].evaluate([request(dtype=dtype)])[0]
    assert got["generated"] != 0 and got["read"] != 0 and got["unchanged"]
