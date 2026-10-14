"""v19 dual-prefix/used protocol, independent fragment oracle and corrupt records."""

import contextlib
import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_split_tnd_build import compile_drivers


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("split-tnd")
    )
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CpuDriver(p)) for p in compile_drivers(directory)]


def request(
    qa=(7, 0, 65),
    ka=(1057, 3, 513),
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
):
    cq = [0, *itertools.accumulate(qa)]
    ck = [0, *itertools.accumulate(ka)]
    q, k = max(qa, default=0), max(ka, default=0)
    return [
        len(qa),
        sum(qa) if totalq is None else totalq,
        sum(ka),
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
        *(ka if usedk is None else usedk),
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
    dict(qa=[9, 5], ka=[3, 0], causal=1),
    dict(left=0, right=0),
    dict(left=530),
    dict(right=1),
    dict(left=5, right=2, causal=1),
    dict(qa=[1, 1, 1, 1], ka=[3] * 4),
    dict(qa=[0, 0, 0]),
    dict(ka=[0, 0, 0]),
    dict(qa=[], ka=[]),
    dict(qa=[0], ka=[0]),
    dict(usedq=[0, 0, 0]),
    dict(usedk=[0, 0, 0]),
    dict(usedq=[1, 0, 64]),
    dict(usedk=[1, 0, 512]),
]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("spec", CASES)
def test_protocol(drivers, dtype, spec):
    r = request(dtype=dtype, **spec)
    a = drivers[0].evaluate([r])[0]
    assert a["built"] and a["generated"] == a["read"] == 0, a
    b, tq, tk, q, k, h, s, left, right, causal, _, uq, uk = r[:13]
    record = a["record"]
    assert len(record) == 416 + 10 * b and record[1:3] == [19, 19]
    assert a["tasks"] == a["groups"] * s
    if not tq or not tk:
        assert a["groups"] == a["bytes"] == 0
        return
    m, n = record[15:17]
    kind, wl, wr = record[410 + 8 * b : 413 + 8 * b]
    wl, wr = (x if x < 2**31 else x - 2**32 for x in (wl, wr))
    qa = spec.get("qa", [7, 0, 65])
    ka = spec.get("ka", [1057, 3, 513])
    qs = spec.get("usedq", qa)
    ks = spec.get("usedk", ka)
    expected = []
    for bi, (ql, kl) in enumerate(zip(qs, ks)):
        span = record[410 + 8 * bi : 418 + 8 * bi]
        assert span == [sum(qa[:bi]), 0, sum(ka[:bi]), 0, qa[bi], ka[bi], ql, kl]
        for block in range((ql * h + m - 1) // m):
            keys = {
                j // n
                for row in range(block * m, min((block + 1) * m, ql * h))
                for j in range(kl)
                if (kind == 0 or j <= row // h + kl - ql + (wr if kind == 2 else 0))
                and (kind != 2 or j >= row // h + kl - ql - wl)
            }
            first, stop = (min(keys), max(keys) + 1) if keys else (0, 0)
            native = drivers[1].evaluate(
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
            cuts = drivers[2].evaluate([[1, first, stop, s, i] for i in range(s)])
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
        dict(usedq=[-1, 0, 65]),
        dict(h=0),
        dict(h=2**23),
        dict(qa=(2**30 + 1, 0, 1)),
        dict(totalq=2**31),
        dict(ka=(16385, 0, 1)),
    ],
)
def test_invalid_generation(drivers, dtype, spec):
    a = drivers[0].evaluate([request(dtype=dtype, **spec)])[0]
    assert a["generated"] != 0 and a["read"] != 0 and a["groups"] == 12345


@pytest.mark.parametrize(
    "mutation",
    [
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
def test_corruption(drivers, mutation):
    a = drivers[0].evaluate([request(mut=mutation[0], value=mutation[1])])[0]
    assert a["read"] != 0 and a["groups"] == 12345


@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5])
def test_alias_budget_and_live_lengths(drivers, mode):
    a = drivers[0].evaluate([request(mode=mode, usedk=[1057, 3, 513])])[0]
    assert a["read"] != 0 and a["groups"] == 12345
    if mode == 1:
        assert not a["built"] and a["intact"]
