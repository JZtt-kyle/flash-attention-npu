"""PA softcap feature binding, native partitions and independent safety boundaries."""

import contextlib
import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_prefill_bsnd_build import compile_drivers as prefill_build
from tests.mla.support.mla_split_paged_bsnd_build import compile_drivers as explicit_build
from tests.mla.support.mla_auto_split_paged_bsnd_build import compile_drivers as auto_build


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    root = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("softcap-pa")
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
                CpuDriver(
                    paths[0 if name == "prefill" else 1],
                    args=["--softcap-paged" if name == "prefill" else "--softcap"],
                )
            )
            result[name + "-native"] = stack.enter_context(
                CpuDriver(paths[1 if name == "prefill" else 2])
            )
            if name == "prefill":
                result["tiling"] = stack.enter_context(CpuDriver(paths[2], args=["--paged"]))
        yield result


def request(
    kind,
    *,
    b=2,
    q=17,
    k=1057,
    page=48,
    splits=3,
    mask=0,
    dtype=1,
    cap=1,
    uq=None,
    uk=None,
    fault=-1,
    value=0,
    mode=0,
    heads=8,
):
    capacity = (k + page - 1) // page * page
    qlens = [q] * b if uq is None else uq
    klens = [k] * b if uk is None else uk
    left, right, causal = (-1, -1, 0) if mask == 0 else (-1, 0, 1) if mask == 1 else (7, 2, 0)
    if kind == "prefill":
        left, right = (
            (capacity - 1, q - 1) if mask == 0 else (capacity - 1, 0) if mask == 1 else (7, 2)
        )
        return [
            b,
            q,
            capacity,
            heads,
            2,
            splits,
            26,
            dtype,
            2 + int(uq is not None),
            mask,
            left,
            right,
            fault,
            value,
            cap,
            page,
            *qlens,
            *klens,
        ]
    config = [b, q, capacity, page, heads, 2 if kind == "auto" else splits, 26, 0, b * q, 0, -1, 0]
    control = [left, right, causal, dtype, mode, fault, value, 0]
    if kind == "auto":
        control += [0, 0, 0, 0]
    arrays = [
        0,
        0,
        0,
        int(uq is not None),
        b if uq is not None else 0,
        b if uq is not None else 0,
        *(qlens if uq is not None else []),
        1,
        b,
        b,
        *klens,
    ]
    return [0, *config, *control, cap, *arrays]


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("page,mask,splits", [(16, 0, 1), (48, 1, 3), (1024, 2, 0), (48, 0, 256)])
def test_feature_source_partitions(drivers, kind, dtype, page, mask, splits):
    splits = 2 if kind == "explicit" and splits == 0 else splits
    r = drivers[kind].evaluate([request(kind, page=page, mask=mask, splits=splits, dtype=dtype)])[0]
    assert r["built"] and r.get("status", r.get("read")) == 0, r
    rec = r["record"]
    assert rec[1:3] == [dict(prefill=43, explicit=41, auto=42)[kind]] * 2
    if kind != "explicit" or splits != 1:
        old = drivers[kind].evaluate(
            [request(kind, page=page, mask=mask, splits=splits, dtype=dtype, cap=0)]
        )[0]
        assert rec[:1] + rec[3:] == old["record"][:1] + old["record"][3:]
    if kind == "explicit":
        assert r["tasks"] == r["groups"] * splits and r["bytes"] <= r["budget"]
        assert r["page_reads"] == 2 * ((1057 + page - 1) // page)
        return
    base = 28 if kind == "prefill" else 418
    m, n = rec[13:15] if kind == "prefill" else rec[15:17]
    hk = 2 if kind == "prefill" else 1
    if kind == "prefill":
        assert [m, n] == drivers["tiling"].evaluate(
            [[17, (1057 + page - 1) // page * page, 8 // hk, mask, page]]
        )[0]
    mk, left, right = rec[base : base + 3]
    native = drivers[kind + "-native"].evaluate(
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
                2 * hk,
                *itertools.chain.from_iterable([(17, 1057)] * (2 * hk)),
            ]
        ]
    )[0]
    assert rec[base + 3] == native[0]
    assert rec[base + 82 : base + 86] == [*native[2:5], 8]
    pos = 7
    for off, count in (
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
        assert rec[base + off : base + off + count] == native[pos : pos + count]
        pos += count


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize(
    "kwargs",
    [dict(b=0), dict(q=0), dict(k=0), dict(uq=[0, 3], uk=[1057, 0]), dict(k=16384, page=1024, q=1)],
)
def test_empty_used_and_capacity(drivers, kind, dtype, kwargs):
    r = drivers[kind].evaluate([request(kind, dtype=dtype, **kwargs)])[0]
    assert r["built"] and r.get("status", r.get("read")) == 0, r


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize(
    "fault,value", [(1, 30), (2, 24), (3, 1), (4, 99), (5, 1), (7, 0xFFFFFFFF), (19, 0xFFFFFFFF)]
)
def test_corrupt_feature_and_geometry(drivers, kind, dtype, fault, value):
    r = drivers[kind].evaluate([request(kind, dtype=dtype, fault=fault, value=value)])[0]
    assert r.get("status", r.get("read")) != 0


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(cap=2),
        dict(cap=0xFFFFFFFF),
        dict(uq=[18, 3]),
        dict(uk=[1105, 0]),
        dict(page=17),
        dict(heads=0),
    ],
)
def test_invalid_feature_lengths_and_bounds(drivers, kind, kwargs):
    r = drivers[kind].evaluate([request(kind, **kwargs)])[0]
    assert not r.get("layout", True) or not r["built"] or r.get("status", r.get("read")) != 0, r


@pytest.mark.parametrize("kind", ["explicit", "auto"])
@pytest.mark.parametrize(
    "mode", [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 28, 29, 30]
)
def test_page_address_workspace_and_overlap(drivers, kind, mode):
    r = drivers[kind].evaluate([request(kind, mode=mode)])[0]
    assert not r["built"] or r["read"] != 0, r
    assert r["unchanged"]
    if not r["built"]:
        assert r["intact"]
