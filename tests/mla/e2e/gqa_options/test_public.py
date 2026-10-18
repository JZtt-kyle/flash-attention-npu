"""GQA options at public, raw, fake and compile entries for every physical layout."""

import contextlib
import gc
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from tests.mla.support.public_cases.gqa_options import (
    LAYOUTS,
    CASES,
    DAILY,
    REPRESENTATIVE,
    make_case,
    metadata,
    invoke,
    compare,
)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("spec", [pytest.param(s, id=s["id"]) for s in CASES])
def test_behavior(layout, representation, dtype, spec):
    c = make_case(layout, representation, dtype, **spec)
    record = metadata(c, geometry=layout == "tnd")
    compare(c, invoke(c), record)
    compare(c, invoke(c, None if layout == "tnd" else record, entry="op"), record)
    for name, value in c.get("physical", {}).items():
        torch.testing.assert_close(c["t"][name].cpu(), value, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "spec", [pytest.param(s, id=s["id"]) for s in CASES if s["id"] in REPRESENTATIVE]
)
def test_representative(layout, representation, dtype, spec):
    c = make_case(layout, representation, dtype, **spec)
    compare(c, invoke(c), metadata(c, geometry=layout == "tnd"))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("spec", [pytest.param(s, id=s["id"]) for s in CASES if s["id"] in DAILY])
def test_daily(layout, representation, dtype, spec):
    c = make_case(layout, representation, dtype, **spec)
    record = metadata(c, geometry=layout == "tnd")
    compare(c, invoke(c, None if layout == "tnd" else record), record)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_prefill_paged_tail_softcap(dtype):
    # PA48 yields N=480; Q17 leaves one row on AIV1, KV513 a 33-column tail.
    c = make_case(
        "paged_bsnd",
        "prefill",
        dtype,
        pack=False,
        margin=1,
        splits=0,
        mask="causal",
        cap=15,
        page=48,
        q=(17, 3),
        k=(1057, 513),
    )
    record = metadata(c)
    for entry, reuse in (("public", False), ("raw", True), ("public", True)):
        compare(c, invoke(c, record if reuse else None, entry=entry), record)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "representation,dtype,splits,q,k",
    [
        ("compressed", "fp16", 1, (), ()),
        ("prefill", "bf16", 0, (0, 0), (33, 3)),
        ("compressed", "bf16", 3, (7, 3), (0, 0)),
        ("prefill", "fp16", 1, (7, 3), (0, 0)),
    ],
)
def test_empty(layout, representation, dtype, splits, q, k):
    c = make_case(layout, representation, dtype, splits=splits, q=q, k=k, pack=False, margin=1)
    compare(c, invoke(c), metadata(c, geometry=layout == "tnd"))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "representation,splits,pack,margin",
    [
        ("compressed", 0, False, 1),
        ("prefill", 3, True, 2),
        ("prefill", 1, False, 0),
    ],
)
def test_fake(layout, representation, splits, pack, margin):
    spec = dict(splits=splits, pack=pack, margin=margin, mask="window")
    c = make_case(layout, representation, "bf16", **spec)
    real = invoke(c, entry="raw")
    with FakeTensorMode():
        c = make_case(layout, representation, "bf16", **spec)
        record = metadata(c)
        for entry in ("public", "op", "fake"):
            result = invoke(c, record if layout != "tnd" else None, entry=entry)
            for x, y in zip(real, result):
                assert (x.shape, x.stride(), x.dtype, x.device) == (
                    y.shape,
                    y.stride(),
                    y.dtype,
                    y.device,
                )


@pytest.mark.parametrize(
    "layout,representation,pack,margin",
    [
        ("bsnd", "compressed", False, 0),
        ("tnd", "prefill", True, 1),
        ("mixed", "compressed", False, 1),
        ("paged_bsnd", "prefill", False, 0),
        ("paged_tnd", "compressed", True, 2),
    ],
)
def test_compile(layout, representation, pack, margin):
    c = make_case(layout, representation, "bf16", splits=3, mask="window", pack=pack, margin=margin)

    def call(q):
        return invoke(dict(c, t=dict(c["t"], q=q)))

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    compare(c, compiled(c["t"]["q"]), metadata(c, geometry=layout == "tnd"))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "representation,dtype,entry", [("compressed", "fp16", "cpp"), ("prefill", "bf16", "raw")]
)
def test_reuse_lifetime(layout, representation, dtype, entry):
    spec = dict(splits=3, mask="window", pack=False, margin=1, strided=True)
    c = make_case(layout, representation, dtype, **spec)
    record = metadata(c, geometry=layout == "tnd")
    if layout != "tnd":
        torch.testing.assert_close(record, metadata(c, entry="cpp"), rtol=0, atol=0)
    compare(c, invoke(c, None if layout == "tnd" else record, entry=entry), record)
    second = make_case(layout, representation, dtype, **dict(spec, scale=0))
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        result = invoke(second, None if layout == "tnd" else record, entry=entry)
        event = torch.npu.Event()
        event.record()
    del c
    gc.collect()
    event.synchronize()
    compare(second, result, record)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("fake", [False, True])
def test_rejections(layout, representation, fake):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        c = make_case(layout, representation, mask="window")
        record = metadata(c)
        for bad in (record[:-1], record.to(torch.int64), record.repeat(2)[::2]):
            with pytest.raises(RuntimeError, match="metadata"):
                invoke(c, bad, entry="op")
        for margin in (-1, 26):
            c["options"]["sm_margin"] = margin
            for entry in ("public", "op", "raw"):
                with pytest.raises(RuntimeError, match="sm_margin"):
                    invoke(c, entry=entry)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation,dtype", [("compressed", "fp16"), ("prefill", "bf16")])
def test_resources_and_default(layout, representation, dtype):
    c = make_case(layout, representation, dtype, mask="causal", splits=0)
    physical = int(metadata(c).cpu()[12])
    c["options"]["sm_margin"] = physical - 1
    record = metadata(c, geometry=layout == "tnd")
    assert int(record.cpu()[27]) == 1
    compare(c, invoke(c), record)
    c["options"]["sm_margin"] = physical
    with pytest.raises(RuntimeError, match="sm_margin"):
        invoke(c)
    c["options"].update(sm_margin=0, pack_gqa=None)
    implicit = invoke(c)
    c["options"]["pack_gqa"] = True
    explicit = invoke(c)
    for x, y in zip(implicit, explicit):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    one = make_case(
        layout,
        representation,
        dtype,
        heads=1 if representation == "compressed" else 2,
        margin=1,
        mask="window",
        splits=3,
    )
    compare(one, invoke(one), metadata(one, geometry=layout == "tnd"))


@pytest.mark.parametrize("layout", ["tnd", "mixed", "paged_tnd"])
@pytest.mark.parametrize("representation,dtype", [("compressed", "bf16"), ("prefill", "fp16")])
def test_query_used(layout, representation, dtype):
    c = make_case(
        layout,
        representation,
        dtype,
        q=(7, 0, 3),
        k=(65, 7, 3),
        used_q=(3, 0, 1),
        used_k=(33, 0, 2),
        mask="window",
        splits=0,
        pack=False,
        margin=1,
        strided=True,
    )
    compare(c, invoke(c, entry="op"), metadata(c, geometry=True))


@pytest.mark.parametrize("layout", ["bsnd", "mixed", "paged_bsnd", "paged_tnd"])
@pytest.mark.parametrize("representation,dtype", [("compressed", "fp16"), ("prefill", "bf16")])
def test_length_update(layout, representation, dtype):
    c = make_case(
        layout,
        representation,
        dtype,
        q=(7, 3),
        k=(65, 7),
        mask="window",
        splits=3,
        pack=False,
        margin=1,
    )
    old = metadata(c)
    c["klens"] = [33, 0]
    lengths = c["lengths"]["seqused_k"] if isinstance(c["lengths"], dict) else c["lengths"]
    lengths.copy_(lengths.new_tensor(c["klens"]))
    record = metadata(c)
    assert not torch.equal(old.cpu(), record.cpu())
    compare(c, invoke(c, record), record)
    compare(c, invoke(c), record)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation,dtype", [("compressed", "fp16"), ("prefill", "bf16")])
def test_head_broadcast(layout, representation, dtype):
    c = make_case(layout, representation, dtype, splits=0, mask="window", pack=False, margin=1)
    for name, value in c["t"].items():
        c["t"][name] = value[..., :1, :].expand_as(value)
        c["raw"][name] = c["raw"][name][..., :1, :].expand_as(c["raw"][name])
    compare(c, invoke(c), metadata(c, geometry=layout == "tnd"))


@pytest.mark.parametrize("representation,dtype", [("compressed", "bf16"), ("prefill", "fp16")])
def test_tnd_empty_k_declared_capacity(representation, dtype):
    c = make_case("tnd", representation, dtype, q=(7, 3), k=(0, 0), pack=False, margin=1, splits=3)
    c["shape"] = (*c["shape"][:3], 65, c["shape"][4])
    c["lengths"]["max_seqlen_k"] = 65
    compare(c, invoke(c), metadata(c, geometry=True))


@pytest.mark.parametrize("fake", [False, True])
def test_private_requires_mla_dimensions(fake):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        c = make_case("bsnd", "compressed", pack=False)
        c["t"]["v"] = c["t"]["k"]
        c["t"].pop("qv")
        with pytest.raises(RuntimeError, match="MLA|192|compressed"):
            invoke(c, entry="op")


@pytest.mark.parametrize("layout", ["mixed", "paged_tnd"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("fake", [False, True])
def test_prefill_query_used_record(layout, dtype, fake):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        c = make_case(
            layout,
            "prefill",
            dtype,
            q=(7, 3),
            k=(65, 7),
            used_q=(3, 1),
            splits=0,
            mask="window",
            pack=False,
            margin=1,
            strided=True,
        )
        record = metadata(c, entry="op")
        result = invoke(c, record.clone(), entry="op")
        if fake:
            assert result[0].shape == (*c["t"]["q"].shape[:-1], 128)
        else:
            assert record.cpu()[19] == 3
            compare(c, result, record)


@pytest.mark.parametrize(
    "layout,representation,dtype",
    [("mixed", "compressed", "fp16"), ("paged_bsnd", "prefill", "bf16")],
)
def test_record_copy_offset(layout, representation, dtype):
    c = make_case(layout, representation, dtype, splits=3, mask="window", pack=False, margin=1)
    record = metadata(c)
    storage = torch.empty(record.numel() + 8, dtype=record.dtype, device=record.device)
    copied = storage[8:].copy_(record)
    compare(c, invoke(c, copied), record)
