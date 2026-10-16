"""Softcap BSND protocol binding, native partition parity and allocation bounds."""

import contextlib
import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_prefill_bsnd_build import compile_drivers as prefill_build
from tests.mla.support.mla_split_bsnd_build import compile_drivers as explicit_build
from tests.mla.support.mla_auto_split_bsnd_metadata_build import compile_drivers as auto_build


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    root = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("softcap-bsnd")
    )
    with contextlib.ExitStack() as stack:
        result = {}
        for name, build in (
            ("prefill", prefill_build),
            ("explicit", explicit_build),
            ("auto", auto_build),
        ):
            paths = build(root / name)
            result[name] = stack.enter_context(CpuDriver(paths[0], args=["--softcap"]))
            result[name + "-native"] = stack.enter_context(CpuDriver(paths[1]))
            if name == "prefill":
                result["tiling"] = stack.enter_context(CpuDriver(paths[2]))
        yield result


def request(
    kind, *, shape=(2, 17, 1057, 8), splits=2, mask=0, dtype=1, cap=1, fault=-1, value=0, mode=0
):
    b, q, k, h = shape
    left, right = (-1, -1) if mask == 0 else (-1, 0) if mask == 1 else (7, 2)
    if kind == "prefill":
        left, right = (k - 1, q - 1) if mask == 0 else (k - 1, 0) if mask == 1 else (7, 2)
        header = [b, q, k, h, 2, splits, 26, dtype, 3, mask, left, right, fault, value, cap]
    elif kind == "auto":
        header = [1, b, q, k, h, 26, 2, dtype, 1, 1, left, right, int(mask == 1), mode, cap]
    else:
        header = [
            b,
            q,
            k,
            h,
            splits,
            left,
            right,
            int(mask == 1),
            dtype,
            1,
            1,
            fault,
            value,
            mode,
            cap,
        ]
    return [*header, *([q] * b), *([k] * b)]


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("mask,splits", [(0, 1), (1, 2), (2, 0), (0, 256)])
def test_feature_and_source_partitions(drivers, kind, dtype, mask, splits):
    if kind == "explicit" and splits == 0:
        splits = 4
    if kind == "auto":
        splits = 0
    d = drivers[kind]
    r = d.evaluate([request(kind, splits=splits, mask=mask, dtype=dtype)])[0]
    assert r["built"] and r.get("status", r.get("read")) == 0, r
    rec = r["record"]
    assert rec[1:3] == [dict(prefill=34, explicit=32, auto=33)[kind]] * 2
    # The feature carries no scale/cap value; every scheduling word retains
    # the established schedule. Single explicit split is the one new boundary.
    if kind != "explicit" or splits != 1:
        old = d.evaluate([request(kind, splits=splits, mask=mask, dtype=dtype, cap=0)])[0]
        assert rec[:1] + rec[3:] == old["record"][:1] + old["record"][3:]
    if kind == "explicit":
        assert r["tasks"] == r["groups"] * splits
        for _, _, part, first, last in r["fragments"]:
            assert part < splits and 0 <= first <= last <= 3
        return
    b, q, k, h = (2, 17, 1057, 8)
    hk = 2 if kind == "prefill" else 1
    base = 20 + 2 * b if kind == "prefill" else 410 + 8 * b
    m, n = rec[13:15] if kind == "prefill" else rec[15:17]
    mk, left, right = rec[base : base + 3]
    if kind == "prefill":
        assert [m, n] == drivers["tiling"].evaluate([[q, k, h // hk, mask]])[0]
    source = drivers[kind + "-native"].evaluate(
        [
            [
                1,
                3 if mk == 1 else 4 if mk == 2 else 0,
                2**31 - 1 if mk == 1 else left,
                0 if mk == 1 else right,
                h // hk,
                m,
                n,
                26,
                2,
                b * hk,
                *itertools.chain.from_iterable([(q, k)] * (b * hk)),
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
@pytest.mark.parametrize(
    "word,value", [(1, 17), (2, 27), (3, 1), (4, 99), (5, 1), (7, 0xFFFFFFFF), (19, 0xFFFFFFFF)]
)
def test_corrupt_record_rejected(drivers, kind, word, value):
    if kind == "auto":
        drivers[kind].evaluate([request(kind)])
        r = drivers[kind].evaluate([[5, word, value]])[0]
    else:
        r = drivers[kind].evaluate([request(kind, fault=word, value=value)])[0]
    assert r.get("status", r.get("read")) != 0, r


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize("shape", [(0, 7, 31, 8), (2, 0, 31, 8), (2, 7, 0, 8), (1, 1, 16384, 128)])
def test_empty_and_capacity(drivers, kind, shape):
    r = drivers[kind].evaluate([request(kind, shape=shape, splits=1)])[0]
    assert r["built"] and r.get("status", r.get("read")) == 0, r


@pytest.mark.parametrize("kind", ["prefill", "explicit", "auto"])
@pytest.mark.parametrize("cap", [2, 0xFFFFFFFF])
def test_invalid_feature(drivers, kind, cap):
    r = drivers[kind].evaluate([request(kind, cap=cap)])[0]
    assert not r.get("layout", True) or not r["built"] or r.get("status", r.get("read")) != 0


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("explicit", {"splits": 257}),
        ("prefill", {"shape": (1, 2**30, 31, 8)}),
        ("explicit", {"mode": 2}),
        ("prefill", {"fault": -4}),
        ("auto", {"mode": 1}),
    ],
)
def test_bounds_and_overlap(drivers, kind, kwargs):
    r = drivers[kind].evaluate([request(kind, **kwargs)])[0]
    assert not r.get("layout", True) or not r["built"] or r.get("status", r.get("read")) != 0
