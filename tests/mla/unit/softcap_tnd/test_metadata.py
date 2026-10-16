"""Feature binding, allocation safety and unchanged pinned native partitions."""

import contextlib
import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_prefill_bsnd_build import compile_drivers as prefill_build
from tests.mla.support.mla_split_tnd_build import compile_drivers as explicit_build
from tests.mla.support.mla_auto_split_tnd_metadata_build import compile_drivers as auto_build


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    root = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("softcap-tnd")
    )
    with contextlib.ExitStack() as stack:
        result = {}
        for name, build in (
            ("prefill", prefill_build),
            ("explicit", explicit_build),
            ("auto", auto_build),
        ):
            paths = build(root / name)
            result[name] = stack.enter_context(
                CpuDriver(paths[0], args=["--softcap-tnd" if name == "prefill" else "--softcap"])
            )
            result[name + "-native"] = stack.enter_context(CpuDriver(paths[1]))
            if name == "prefill":
                result["tiling"] = stack.enter_context(CpuDriver(paths[2], args=["--tnd"]))
            if name == "explicit":
                result["plan"] = stack.enter_context(CpuDriver(root / name / "plan"))
        yield result


def request(
    kind,
    *,
    q=(7, 0, 17),
    k=(1057, 31, 3),
    splits=2,
    mask=0,
    dtype=1,
    cap=1,
    fault=-1,
    value=0,
    mode=0,
    uq=None,
    uk=None,
    heads=8,
    maxima=None,
):
    b, tq, tk = len(q), sum(q), sum(k)
    maxq, maxk = maxima or (max(q, default=0), max(k, default=0))
    cq, ck = [0, *itertools.accumulate(q)], [0, *itertools.accumulate(k)]
    left, right = (
        (-1, -1) if mask == 0 else (-1, 0) if mask == 1 else (min(7, maxk - 1), min(2, maxq - 1))
    )
    flags = int(uq is not None) + 2 * int(uk is not None)
    if kind == "prefill":
        left, right = (
            (maxk - 1, maxq - 1) if mask == 0 else (maxk - 1, 0) if mask == 1 else (left, right)
        )
        header = [
            b,
            maxq,
            maxk,
            heads,
            2,
            splits,
            26,
            dtype,
            flags,
            mask,
            left,
            right,
            fault,
            value,
            cap,
            tq,
            tk,
        ]
        arrays = [*(q if uq is None else uq), *(k if uk is None else uk), *cq, *ck]
    elif kind == "auto":
        header = [
            1,
            b,
            maxq,
            maxk,
            heads,
            26,
            2,
            dtype,
            flags & 1,
            flags >> 1,
            left,
            right,
            int(mask == 1),
            mode,
            tq,
            tk,
            cap,
        ]
        arrays = [*cq, *ck, *(q if uq is None else uq), *(k if uk is None else uk)]
    else:
        header = [
            b,
            tq,
            tk,
            maxq,
            maxk,
            heads,
            splits,
            left,
            right,
            int(mask == 1),
            dtype,
            flags & 1,
            flags >> 1,
            fault,
            value,
            mode,
            cap,
        ]
        arrays = [*cq, *ck, *(q if uq is None else uq), *(k if uk is None else uk)]
    return [*header, *arrays]


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("mask,splits", [(0, 1), (1, 2), (2, 0), (0, 256)])
def test_feature_and_source_partitions(drivers, kind, dtype, mask, splits):
    if kind == "explicit" and splits == 0:
        splits = 4
    if kind == "auto":
        splits = 0
    r = drivers[kind].evaluate([request(kind, splits=splits, mask=mask, dtype=dtype)])[0]
    assert r["built"] and r.get("status", r.get("read")) == 0, r
    rec = r["record"]
    assert rec[1:3] == [dict(prefill=37, explicit=35, auto=36)[kind]] * 2
    if kind != "explicit" or splits != 1:
        old = drivers[kind].evaluate([request(kind, splits=splits, mask=mask, dtype=dtype, cap=0)])[
            0
        ]
        assert rec[:1] + rec[3:] == old["record"][:1] + old["record"][3:]
    if kind == "explicit":
        assert r["tasks"] == r["groups"] * splits
        assert all(
            part < splits and 0 <= first <= last <= 3 for _, _, part, first, last in r["fragments"]
        )
        return
    q, k = (7, 0, 17), (1057, 31, 3)
    hk = 2 if kind == "prefill" else 1
    base = 24 + 8 * 3 if kind == "prefill" else 410 + 8 * 3
    m, n = rec[13:15] if kind == "prefill" else rec[15:17]
    mk, left, right = rec[base : base + 3]
    if kind == "prefill":
        assert [m, n] == drivers["tiling"].evaluate([[17, 1057, 8 // hk, mask]])[0]
    source = drivers[kind + "-native"].evaluate(
        [
            [
                1,
                3 if mk == 1 else 4 if mk == 2 else 0,
                2**31 - 1 if mk == 1 else left,
                0 if mk == 1 else right,
                8 // hk,
                m,
                n,
                26,
                2,
                3 * hk,
                *itertools.chain.from_iterable(pair for pair in zip(q, k) for _ in range(hk)),
            ]
        ]
    )[0]
    assert rec[base + 3] == source[0]
    assert rec[base + 82 : base + 86] == [*source[2:5], 8]
    pos = 7
    for offset, count in (
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
        assert rec[base + offset : base + offset + count] == source[pos : pos + count]
        pos += count


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("flags", [0, 1, 2, 3])
def test_used_allocation_spans(drivers, kind, dtype, flags):
    uq = (3, 0, 1) if flags & 1 else None
    uk = (513, 0, 0) if flags & 2 else None
    r = drivers[kind].evaluate([request(kind, dtype=dtype, uq=uq, uk=uk)])[0]
    assert r.get("status", r.get("read")) == 0, r
    base = 24 if kind == "prefill" else 410
    for i, (qs, ks) in enumerate(zip((0, 7, 7), (0, 1057, 1088))):
        assert r["record"][base + 8 * i : base + 8 * i + 8] == [
            qs,
            0,
            ks,
            0,
            (7, 0, 17)[i],
            (1057, 31, 3)[i],
            (uq or (7, 0, 17))[i],
            (uk or (1057, 31, 3))[i],
        ]


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize(
    "word,value", [(1, 19), (2, 28), (3, 1), (4, 99), (5, 1), (7, 0xFFFFFFFF), (19, 0xFFFFFFFF)]
)
def test_corrupt_record_rejected(drivers, kind, word, value):
    if kind == "auto":
        drivers[kind].evaluate([request(kind)])
        r = drivers[kind].evaluate([[5, word, value]])[0]
    else:
        r = drivers[kind].evaluate([request(kind, fault=word, value=value)])[0]
    assert r.get("status", r.get("read")) != 0, r


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize(
    "q,k", [((), ()), ((0, 0), (31, 3)), ((7, 2), (0, 0)), ((1, 1), (16384, 513))]
)
def test_empty_and_capacity(drivers, kind, q, k):
    r = drivers[kind].evaluate([request(kind, q=q, k=k, splits=1)])[0]
    assert r["built"] and r.get("status", r.get("read")) == 0, r


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"cap": 2},
        {"cap": 0xFFFFFFFF},
        {"uq": (8, 0, 1)},
        {"uk": (1058, 0, 0)},
        {"maxima": (16, 1057)},
    ],
)
def test_feature_and_length_bounds(drivers, kind, kwargs):
    r = drivers[kind].evaluate([request(kind, **kwargs)])[0]
    assert not r.get("layout", True) or not r["built"] or r.get("status", r.get("read")) != 0, r


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("explicit", {"splits": 257}),
        ("explicit", {"mode": 2}),
        ("explicit", {"mode": 1, "uq": (7, 0, 17), "uk": (1057, 31, 3)}),
        ("prefill", {"fault": -12}),
        ("auto", {"mode": 1, "uk": (1057, 31, 3)}),
    ],
)
def test_workspace_and_overlap(drivers, kind, kwargs):
    r = drivers[kind].evaluate([request(kind, **kwargs)])[0]
    assert not r.get("layout", True) or not r["built"] or r.get("status", r.get("read")) != 0, r


@pytest.mark.parametrize("q,k", [(7, 33), (0, 33), (7, 0)])
def test_single_split_plan_workspace_and_rows(drivers, q, k):
    d = drivers["plan"]
    h = 4
    m = 16
    n = 32
    config = [2, q, k, h, m, n, 1, 3, 2, 2, 0, q, k]
    arrays = [[0, q, q], [0, k, k], None, None]
    encoded = [v for a in arrays for v in [int(a is not None), len(a or []), *(a or [])]]
    r = d.evaluate([[1, *config, q, k, -1, -1, 0, *encoded]])[0]
    assert r["status"] == 0, r
    r = r["plan"]
    groups = (q * h + m - 1) // m if q * k else 0
    active = min(2, q)
    bound = min(active * ((q * h + m - 1) // m), (q * h + active * (m - 1)) // m) if q * k else 0
    assert r["layout"]["geometry"][:2] == [bound, bound]
    assert r["groups"] == r["tasks"] == groups
    assert r["layout"]["accum"]["rank"] == 0
    if groups:
        for group in range(groups):
            task = d.evaluate([[2, group]])[0]
            merge = d.evaluate([[3, group]])[0]
            assert task["status"] == merge["status"] == 0
            assert task["offsets"] == merge["offsets"]
            row = d.evaluate([[4, group, 0]])[0]
            assert row["status"] != 0 and row["unchanged"]
        assert d.evaluate([[2, groups]])[0]["status"] != 0
