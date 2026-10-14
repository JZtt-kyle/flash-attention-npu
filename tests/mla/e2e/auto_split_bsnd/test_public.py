"""Native automatic BSND split: public results, static capacity and lifecycle."""

import contextlib
import gc
import os
from pathlib import Path
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.e2e.split_kv_bsnd.test_public import make_case as explicit_case, metadata, invoke
from tests.mla.support.mla_auto_split_bsnd_reference import references


def make_case(dtype="fp16", **kwargs):
    return explicit_case(dtype, splits=0, **kwargs)


def geometry(record, batch):
    words = record.cpu().tolist()
    base = 410 + 8 * batch
    count = words[base + 3]
    ends = list(
        zip(
            words[base + 4 : base + 4 + count],
            words[base + 30 : base + 30 + count],
            words[base + 56 : base + 56 + count],
        )
    )
    return words[15], words[16], ends


def compare(c, result, record=None, tile=None):
    if tile is None:
        tile = geometry(metadata(c) if record is None else record, c["shape"][0])
    m, n, ends = tile
    o = c["options"]
    refs = references(
        c["raw"],
        c["qlens"],
        c["klens"],
        ends,
        m or 256,
        n or 512,
        o["causal"],
        o["window_size"],
        o["softmax_scale"],
    )
    actual = [x.cpu() for x in result]
    for value, name in zip(actual, ("out", "lse", "accum", "accum_lse")):
        assert value.shape == refs[name + "_high"].shape
        assert_fa_close(value, refs[name + "_high"], refs[name + "_low"], name="auto-" + name)
    return refs


SPECS = [
    dict(shape=(2, 3, 2, 1025), mask="none"),
    dict(shape=(2, 65, 4, 1057), mask="window", strided=True),
    dict(shape=(2, 7, 2, 3), mask="causal"),
    dict(shape=(1, 1, 128, 16384), mask="none"),
    dict(shape=(3, 65, 4, 8193), mask="causal"),
]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("spec", SPECS)
def test_representative(dtype, spec):
    c = make_case(dtype, **spec)
    record = metadata(c)
    result = invoke(c, record)
    compare(c, result, record)
    if os.environ.get("MLA_VALIDATION_RESULT"):
        shape = "-".join(map(str, c["shape"]))
        torch.save(
            dict(
                raw=c["raw"],
                result=[x.cpu() for x in result],
                record=record.cpu(),
                options=c["options"],
            ),
            Path(os.environ["MLA_VALIDATION_RESULT"]) / f"auto-{dtype}-{spec['mask']}-{shape}.pt",
        )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("mask", ["none", "causal", "window"])
def test_daily(dtype, mask):
    c = make_case(dtype, mask=mask, strided=True, shape=(2, 7, 3, 1057))
    record = metadata(c)
    assert record.numel() == 778 and record.cpu()[1:3].tolist() == [18, 18]
    first = invoke(c, record)
    compare(c, first, record)
    for x, y in zip(first, invoke(c, record)):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    c["options"]["softmax_scale"] = 0.0625
    for name, value in c["raw"].items():
        value.mul_(0.5)
        c["t"][name].copy_(value)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", [(0, 3, 2, 33), (2, 0, 2, 33), (2, 3, 2, 0), (2, 0, 2, 0)])
def test_empty(dtype, shape):
    c = make_case(dtype, shape=shape)
    compare(c, invoke(c, metadata(c)))


def test_fake_and_compile():
    with FakeTensorMode():
        c = make_case()
        r = invoke(c, metadata(c))
        assert r[2].shape == (26, 2, 2, 7, 512) and r[3].shape == (26, 2, 2, 7)
    c = make_case("bf16")
    record = metadata(c)
    compiled = torch.compile(
        lambda q, lengths: api.flash_attn_with_kvcache(
            q,
            c["t"]["k"],
            c["t"]["v"],
            qv=c["t"]["qv"],
            cache_seqlens=lengths,
            return_softmax_lse=True,
            **c["options"],
        ),
        backend="eager",
        fullgraph=True,
    )
    compare(c, compiled(c["t"]["q"], c["lengths"]), record)
    c["lengths"].zero_()
    c["klens"] = [0, 0]
    compare(c, compiled(c["t"]["q"], c["lengths"]))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_public_entries(dtype):
    from tests.mla.e2e.window_bsnd.test_flash_attn_npu_v3_mla_window_public import _forward

    c = make_case(dtype, shape=(1, 7, 3, 1057), mask="window")
    t = c["t"]
    o = c["options"]
    baseline = invoke(c, metadata(c))
    compare(c, baseline)
    for lengths in (None, 1057, c["lengths"]):
        r = api.flash_attn_with_kvcache(
            t["q"], t["k"], t["v"], qv=t["qv"], cache_seqlens=lengths, return_softmax_lse=True, **o
        )
        for x, y in zip(r, baseline):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    for x, y in zip(api.flash_attn_func(**t, return_attn_probs=True, **o), baseline[:2]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for entry in ("eager", "cpp"):
        r = _forward(
            entry,
            dict(
                **t,
                seqused_k=c["lengths"],
                num_splits=0,
                softmax_scale=o["softmax_scale"],
                window_size_left=2,
                window_size_right=1,
            ),
        )
        for x, y in zip(r, baseline):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_length_updates_streams_lifetime(dtype):
    from tests.mla.support.mla_metadata_error_probe import forbid_host_readback

    parent = torch.npu.current_stream()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    kept = []
    for stream in streams:
        with torch.npu.stream(stream):
            stream.wait_stream(parent)
            c = make_case(dtype, strided=True)
            record = metadata(c)
            tile = geometry(record, c["shape"][0])
            with forbid_host_readback():
                first = invoke(c, record)
                c["lengths"].zero_()
                second = invoke(c, metadata(c))
            kept.append(
                (
                    {k: c[k] for k in ("raw", "shape", "qlens", "klens", "options")},
                    first,
                    second,
                    tile,
                )
            )
            del c, record
    gc.collect()
    pressure = [torch.empty((4096,), device="npu") for _ in range(64)]
    for stream in streams:
        stream.synchronize()
    for c, first, second, tile in kept:
        compare(c, first, tile=tile)
        assert (second[0] == 0).all() and second[1].isneginf().all()
        assert (second[2] == 0).all() and second[3].isneginf().all()
    assert len(pressure) == 64


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
def test_static_overflow(dtype, fake):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        b, q, h = 60000, 2**30, 3
        qr = torch.empty((1, 1, 1, 64), dtype=dtype, device="npu").expand(b, q, h, 64)
        qc = torch.empty((1, 1, 1, 512), dtype=dtype, device="npu").expand(b, q, h, 512)
        with pytest.raises(RuntimeError, match="overflow"):
            torch.ops.flash_attn_npu_3._mla_split_bsnd_forward(
                qr, qr[:, :1, :1], qc[:, :1, :1], qc, 0
            )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("window,causal", [((-1, 1), False), ((2, -1), False), ((5, 2), True)])
def test_normalized_window(dtype, window, causal):
    c = make_case(dtype, shape=(2, 7, 3, 1057))
    c["options"].update(window_size=window, causal=causal)
    record = metadata(c)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "fault", ["negative", "too-many", "metadata-shape", "q-used-record", "scale", "grad"]
)
def test_static_rejections(fake, fault):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        c = make_case()
        record = metadata(c)
        if fault == "negative":
            c["options"]["num_splits"] = -1
        elif fault == "too-many":
            c["options"]["num_splits"] = 257
        elif fault == "metadata-shape":
            record = torch.empty((436,), dtype=torch.int32, device="npu")
        elif fault == "q-used-record":
            c["usedq"] = c["lengths"]
        elif fault == "scale":
            c["options"]["softmax_scale"] = float("inf")
        elif fault == "grad":
            c["t"]["q"].requires_grad_(True)
        with pytest.raises((RuntimeError, ValueError)):
            invoke(c, record)


@pytest.mark.parametrize("batch", [(2**32 - 1 - 758) // 10 + 1, 2**32])
def test_metadata_fake_overflow(batch):
    with FakeTensorMode():
        lengths = torch.empty((batch,), dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="overflow|shape bounds"):
            api.get_scheduler_metadata(batch, 0, 0, 1, 1, 64, lengths, headdim_v=512, num_splits=0)
