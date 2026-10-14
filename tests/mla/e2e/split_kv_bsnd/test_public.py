"""Public BSND explicit split, allocation/used, strides, empty slots and accumulators."""

import os
from pathlib import Path
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_split_bsnd_reference import references


def make_case(
    dtype_name="fp16", shape=(2, 7, 2, 33), splits=8, mask="window", strided=False, used=False
):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    b, q, h, k = shape
    g = torch.Generator().manual_seed(20261014)
    raw = {
        name: torch.randn(
            (b, q if name in ("q", "qv") else k, h if name in ("q", "qv") else 1, dim), generator=g
        ).to(dtype)
        for name, dim in (("q", 64), ("qv", 512), ("k", 64), ("v", 512))
    }
    qlens = [q] * b
    klens = [k] * b
    if b > 1:
        klens[-1] = 0 if mask == "causal" else min(k, 2)
    if used and b > 1:
        qlens[-1] = max(0, q - 3)
    tensors = {}
    for name, t in raw.items():
        if strided:
            # Independent head/sequence/batch strides and nonzero storage offsets.
            dims = list(t.shape)
            dims[0] += 1
            dims[1] = dims[1] * 2 + 1
            dims[2] *= 2
            base = torch.empty(dims, dtype=dtype, device="npu")
            dev = base[1:, : 2 * t.shape[1] : 2, ::2, :]
            dev.copy_(t)
        else:
            dev = t.npu()
        tensors[name] = dev
    options = dict(
        num_splits=splits,
        causal=mask == "causal",
        window_size=(2, 1) if mask == "window" else (-1, -1),
        softmax_scale=0.125,
    )
    lengths = torch.tensor(klens, dtype=torch.int32, device="npu")
    usedq = torch.tensor(qlens, dtype=torch.int32, device="npu") if used else None
    return dict(
        raw=raw,
        t=tensors,
        options=options,
        lengths=lengths,
        usedq=usedq,
        qlens=qlens,
        klens=klens,
        shape=shape,
    )


def metadata(c):
    b, q, h, k = c["shape"]
    return api.get_scheduler_metadata(
        b,
        q,
        k,
        h,
        1,
        64,
        c["lengths"],
        qkv_dtype=c["t"]["q"].dtype,
        headdim_v=512,
        num_splits=c["options"]["num_splits"],
        causal=c["options"]["causal"],
        window_size=c["options"]["window_size"],
    )


def invoke(c, record=None):
    t = c["t"]
    o = c["options"]
    if c["usedq"] is not None:
        return torch.ops.flash_attn_npu_3._mla_split_bsnd_forward(
            t["q"],
            t["k"],
            t["v"],
            t["qv"],
            o["num_splits"],
            *o["window_size"],
            o["causal"],
            o["softmax_scale"],
            c["lengths"],
            c["usedq"],
            record,
        )
    return api.flash_attn_with_kvcache(
        t["q"],
        t["k"],
        t["v"],
        qv=t["qv"],
        cache_seqlens=c["lengths"],
        scheduler_metadata=record,
        return_softmax_lse=True,
        **o,
    )


def compare(c, result, tile=None):
    # Fixed-source base geometry: BSND Calculate uses 128..256 divisor or 256.
    b, q, h, k = c["shape"]
    rows = q * h
    m = next((i for i in range(128, 257) if rows % i == 0), 256) if rows else 256
    # CPU scheduler may choose a different source divisor. Read metadata only to
    # select the oracle tile; CPU source/fragment unit tests verify this choice.
    if tile is None:
        md = metadata(c).cpu()
        m = int(md[15]) if q and k and b else 256
        n = int(md[16]) if q and k and b else 512
    else:
        m, n = tile
    ref = references(
        c["raw"],
        c["qlens"],
        c["klens"],
        c["options"]["num_splits"],
        m,
        n,
        c["options"]["causal"],
        c["options"]["window_size"],
        c["options"]["softmax_scale"],
    )
    got = [x.cpu() for x in result]
    for i, name in enumerate(("out", "lse", "accum", "accum_lse")):
        assert got[i].dtype == (c["raw"]["q"].dtype if i == 0 else torch.float32)
        assert got[i].shape == ref[name + "_high"].shape
        for batch, used in enumerate(c["qlens"]):
            idx = (
                (batch, slice(used))
                if i == 0
                else (batch, slice(None), slice(used))
                if i == 1
                else (slice(None), batch, slice(None), slice(used))
            )
            assert_fa_close(
                got[i][idx], ref[name + "_high"][idx], ref[name + "_low"][idx], name="split_" + name
            )
    return ref


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "spec",
    [
        dict(shape=(2, 3, 2, 1025), splits=2, mask="none"),
        dict(shape=(2, 65, 4, 1057), splits=8, mask="window", strided=True, used=True),
        dict(shape=(2, 7, 2, 3), splits=256, mask="causal"),
        dict(shape=(1, 1, 2, 16384), splits=2, mask="none"),
        dict(shape=(1, 65, 4, 513), splits=256, mask="window"),
    ],
)
def test_representative(dtype, spec):
    c = make_case(dtype, **spec)
    r = invoke(c)
    compare(c, r)
    torch.npu.synchronize()
    if os.environ.get("MLA_VALIDATION_RESULT"):
        shape_id = "-".join(map(str, spec["shape"]))
        path = (
            Path(os.environ["MLA_VALIDATION_RESULT"])
            / f"split-{dtype}-{spec['splits']}-{shape_id}.pt"
        )
        torch.save(dict(raw=c["raw"], result=[x.cpu() for x in r], options=c["options"]), path)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("mask", ["none", "causal", "window"])
def test_daily(dtype, mask):
    c = make_case(dtype, mask=mask, strided=True)
    record = metadata(c)
    assert record.numel() == 436
    r = invoke(c, record)
    compare(c, r)
    repeated = invoke(c, record)
    for x, y in zip(r, repeated):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 3, 2, 33), (2, 0, 2, 33), (2, 3, 2, 0), (2, 0, 2, 0)])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_empty(dtype, shape):
    c = make_case(dtype, shape=shape)
    compare(c, invoke(c, metadata(c)))


def test_fake_and_compile():
    with FakeTensorMode():
        c = make_case()
        r = invoke(c, metadata(c))
        assert r[2].shape == (8, 2, 2, 7, 512)
    c = make_case("bf16")
    record = metadata(c)
    compiled = torch.compile(
        lambda q: api.flash_attn_with_kvcache(
            q,
            c["t"]["k"],
            c["t"]["v"],
            qv=c["t"]["qv"],
            cache_seqlens=c["lengths"],
            scheduler_metadata=record,
            return_softmax_lse=True,
            **c["options"],
        ),
        backend="eager",
        fullgraph=True,
    )
    compare(c, compiled(c["t"]["q"]))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits", [193, 255, 256])
def test_extended_split_domain(dtype, splits):
    c = make_case(dtype, shape=(1, 3, 2, 1057), splits=splits, mask="window")
    compare(c, invoke(c, metadata(c)))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_func_integer_none_and_raw(dtype):
    from tests.mla.e2e.window_bsnd.test_flash_attn_npu_v3_mla_window_public import _forward

    c = make_case(dtype, shape=(1, 7, 3, 1057), mask="window", splits=2)
    t = c["t"]
    options = c["options"]
    baseline = invoke(c, metadata(c))
    for lengths in (None, 1057, c["lengths"]):
        got = api.flash_attn_with_kvcache(
            t["q"],
            t["k"],
            t["v"],
            qv=t["qv"],
            cache_seqlens=lengths,
            return_softmax_lse=True,
            **options,
        )
        for x, y in zip(got, baseline):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    got = api.flash_attn_func(**t, return_attn_probs=True, **options)
    for x, y in zip(got, baseline[:2]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    args = dict(
        **t,
        seqused_k=c["lengths"],
        num_splits=2,
        softmax_scale=options["softmax_scale"],
        window_size_left=2,
        window_size_right=1,
    )
    for entry in ("eager", "cpp"):
        got = _forward(entry, args)
        for x, y in zip(got, baseline):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_length_updates_streams_and_lifetime(dtype):
    import gc
    from tests.mla.support.mla_metadata_error_probe import forbid_host_readback

    parent = torch.npu.current_stream()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    kept = []
    for stream in streams:
        with torch.npu.stream(stream):
            stream.wait_stream(parent)
            c = make_case(dtype, strided=True)
            record = metadata(c)
            host_record = record.cpu()
            tile = (int(host_record[15]), int(host_record[16]))
            with forbid_host_readback():
                first = invoke(c, record)
                c["lengths"].fill_(0)
                second = invoke(c, metadata(c))
            reference = {key: c[key] for key in ("raw", "shape", "qlens", "klens", "options")}
            kept.append((reference, first, second, tile))
            del c, record
    gc.collect()
    pressure = [torch.empty((4096,), device="npu") for _ in range(64)]
    for stream in streams:
        stream.synchronize()
    for c, first, second, tile in kept:
        compare(c, first, tile)
        assert (second[0] == 0).all() and second[1].isneginf().all()
        assert (second[2] == 0).all() and second[3].isneginf().all()
    assert len(pressure) == 64


@pytest.mark.parametrize(
    "option", ["too-many", "negative", "q-used-record", "metadata-shape", "scale", "grad"]
)
def test_static_rejections(option):
    c = make_case()
    t = c["t"]
    if option in ("too-many", "negative"):
        c["options"]["num_splits"] = {"too-many": 257, "negative": -1}[option]
    elif option == "q-used-record":
        c["usedq"] = torch.tensor([7, 7], dtype=torch.int32, device="npu")
    elif option == "scale":
        c["options"]["softmax_scale"] = float("inf")
    elif option == "grad":
        t["q"].requires_grad_(True)
    record = torch.empty((436,), dtype=torch.int32, device="npu")
    if option == "metadata-shape":
        record = record[:-1]
    with pytest.raises((RuntimeError, ValueError)):
        invoke(c, record)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
def test_static_workspace_overflow(dtype, fake):
    import contextlib

    with FakeTensorMode() if fake else contextlib.nullcontext():
        # Read-only broadcast views have tiny storage. Static split budgeting
        # must reject before attempting to allocate any output/workspace.
        b, q, h = 6000, 2**30, 3
        qr = torch.empty((1, 1, 1, 64), dtype=dtype, device="npu").expand(b, q, h, 64)
        qc = torch.empty((1, 1, 1, 512), dtype=dtype, device="npu").expand(b, q, h, 512)
        kr = qr[:, :1, :1]
        kv = qc[:, :1, :1]
        with pytest.raises(RuntimeError, match="overflow"):
            torch.ops.flash_attn_npu_3._mla_split_bsnd_forward(qr, kr, kv, qc, 256)


@pytest.mark.parametrize("batch", [(2**32 - 1 - 416) // 10 + 1, 2**32])
def test_fake_metadata_word_overflow(batch):
    with FakeTensorMode():
        lengths = torch.empty((batch,), device="npu", dtype=torch.int32)
        with pytest.raises(RuntimeError, match="overflow|shape bounds"):
            api.get_scheduler_metadata(batch, 0, 0, 1, 1, 64, lengths, headdim_v=512, num_splits=2)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("window,causal", [((-1, 1), False), ((2, -1), False), ((5, 2), True)])
def test_normalization_and_cross_layer_reuse(dtype, window, causal):
    c = make_case(dtype, shape=(2, 7, 3, 1057), splits=2)
    c["options"].update(window_size=window, causal=causal)
    record = metadata(c)
    compare(c, invoke(c, record))
    c["options"]["softmax_scale"] = 0.0625
    for name, value in c["raw"].items():
        value.mul_(0.5)
        c["t"][name].copy_(value)
    compare(c, invoke(c, record))
