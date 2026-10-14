"""v17 serialization, independent consumer, original geometry and failure atomicity."""

import contextlib
import os
from pathlib import Path
import pytest
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_split_bsnd_build import compile_drivers


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("split-bsnd")
    )
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CpuDriver(p)) for p in compile_drivers(directory)]


def request(
    b=2,
    q=9,
    k=1057,
    h=4,
    s=8,
    left=-1,
    right=-1,
    causal=0,
    dtype=1,
    uq=1,
    uk=1,
    mut=-1,
    value=0,
    mode=0,
    qs=None,
    ks=None,
):
    return [
        b,
        q,
        k,
        h,
        s,
        left,
        right,
        causal,
        dtype,
        uq,
        uk,
        mut,
        value,
        mode,
        *(qs if qs is not None else [q] * b),
        *(ks if ks is not None else [k] * b),
    ]


CASES = [
    dict(),
    dict(s=2),
    dict(s=192),
    dict(s=193),
    dict(s=255),
    dict(s=256),
    dict(q=65, h=5, qs=[0, 61], ks=[0, 1055], left=530, right=1),
    dict(q=9, k=3, causal=1, qs=[9, 5], ks=[3, 0]),
    dict(left=0, right=0),
    dict(left=530, right=-1),
    dict(left=-1, right=1),
    dict(left=5, right=2, causal=1),
    dict(q=1, causal=1),
    dict(q=0),
    dict(k=0),
    dict(b=0),
    dict(q=0, k=0),
    dict(qs=[0, 0]),
    dict(ks=[0, 0]),
    dict(uq=0, uk=0),
    dict(uq=0),
    dict(uk=0),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [1, 2])
def test_roundtrip_and_fragment_oracle(drivers, case, dtype):
    case = dict(case, dtype=dtype)
    r = request(**case)
    a = drivers[0].evaluate([r])[0]
    assert a["built"] and a["generated"] == a["read"] == 0, a
    b, q, k, h, s, left, right, causal, *_ = r
    assert len(a["record"]) == 416 + 10 * b and a["record"][1:3] == [17, 17]
    assert a["tasks"] == a["groups"] * s
    # Each sample keeps original effective Q, while static capacity normalizes the window.
    if b and q and k:
        from tests.mla.support.mla_ref import normalize_fa3_window

        wl, wr = normalize_fa3_window(q, k, causal=bool(causal), window_size=(left, right))
        base = 410 + 8 * b
        kind = a["record"][base]
        m = a["record"][15]
        n = a["record"][16]
        # Explicit source FA3 range oracle receives independently enumerated visible tiles.
        actual = {(x[0], x[1], x[2]): x[3:] for x in a["fragments"]}
        for batch in range(b):
            qlen = r[14 + batch] if r[9] else q
            klen = r[14 + b + batch] if r[10] else k
            for block in range((qlen * h + m - 1) // m):
                keys = set()
                for row in range(block * m, min((block + 1) * m, qlen * h)):
                    center = row // h + klen - qlen
                    keys.update(
                        j // n
                        for j in range(klen)
                        if (
                            kind == 0
                            or j <= center + (0 if kind == 1 else (q - 1 if wr < 0 else wr))
                        )
                        and (kind != 2 or j >= center - (k - 1 if wl < 0 else wl))
                    )
                first = min(keys) if keys else 0
                stop = max(keys) + 1 if keys else 0
                mask = 3 if kind == 1 else 4 if kind == 2 else 0
                src_left = 2**31 - 1 if kind == 1 else k - 1 if wl < 0 else wl
                src_right = 0 if kind == 1 else q - 1 if wr < 0 else wr
                native = drivers[1].evaluate(
                    [[0, mask, src_left, src_right, h, m, n, 26, 2, 1, qlen, klen, block]]
                )[0]
                assert native[2:4] == [first, stop]
                cuts = drivers[2].evaluate([[1, first, stop, s, i] for i in range(s)])
                for split in range(s):
                    assert [min(stop, cuts[split][0]), cuts[split][1]] == [
                        min(stop, first + split * ((stop - first + s - 1) // s)),
                        min(stop, first + (split + 1) * ((stop - first + s - 1) // s)),
                    ]
                    width = (stop - first + s - 1) // s
                    lo = min(stop, first + split * width)
                    hi = min(stop, lo + width)
                    assert actual[batch, block, split] == [lo, hi]


@pytest.mark.parametrize(
    "change",
    [
        dict(s=1),
        dict(s=257),
        dict(s=-1),
        dict(qs=[-1, 9]),
        dict(qs=[10, 9]),
        dict(ks=[-1, 0]),
        dict(ks=[1058, 0]),
        dict(dtype=3),
        dict(s=2**40),
        dict(b=0, q=2**32),
        dict(q=65536, h=65536, qs=[0, 0]),
    ],
)
def test_failed_producer_clears_payload(drivers, change):
    a = drivers[0].evaluate([request(**change)])[0]
    assert a["built"] and a["generated"] != 0 and a["read"] != 0
    assert not any(a["record"][8:]) and a["groups"] == 12345


@pytest.mark.parametrize(
    "word", [0, 1, 2, 3, 4, 5, 7, 8, 9, 18, 19, 409, 410, 414, 416, 426, 429, 430, 431, 434, 435]
)
def test_corruption_rejected_before_output(drivers, word):
    a = drivers[0].evaluate([request(mut=word, value=123456789)])[0]
    assert a["read"] != 0 and a["groups"] == 12345


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_overlap_workspace_and_stale_lengths(drivers, mode):
    a = drivers[0].evaluate([request(mode=mode)])[0]
    if mode == 1:
        assert not a["built"] and a["intact"]
    assert a["read"] != 0 and a["groups"] == 12345
