"""Public TND softcap numerics, allocation/used, records and lifetime."""

import contextlib
import gc
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.softcap_tnd import (
    CASES,
    DAILY,
    REPRESENTATIVE,
    make_case,
    metadata,
    invoke,
    compare,
)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("spec", [pytest.param(s, id=s["id"]) for s in CASES])
def test_behavior(representation, dtype, spec):
    c = make_case(representation, dtype, **spec)
    record = metadata(c)
    version = 37 if representation == "prefill" else 36 if spec.get("splits", 1) == 0 else 35
    assert record.cpu()[1:3].tolist() == [version] * 2
    compare(c, invoke(c, record), record)
    compare(c, invoke(c, entry="public"), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "spec", [pytest.param(s, id=s["id"]) for s in CASES if s["id"] in REPRESENTATIVE]
)
def test_representative(representation, dtype, spec):
    c = make_case(representation, dtype, **spec)
    record = metadata(c)
    compare(c, invoke(c, entry="public"), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("spec", [pytest.param(s, id=s["id"]) for s in CASES if s["id"] in DAILY])
def test_daily(representation, dtype, spec):
    c = make_case(representation, dtype, **spec)
    record = metadata(c)
    compare(c, invoke(c, entry="public"), record)
    c["options"].update(softcap=30, softmax_scale=-0.23)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits", [0, 1, 3])
def test_representative_length_update(representation, dtype, splits):
    c = make_case(
        representation,
        dtype,
        q=(7, 0, 17),
        k=(513, 31, 3),
        used_q=(7, 0, 17),
        used_k=(513, 31, 3),
        mask="window",
        splits=splits,
        strided=True,
    )
    old = metadata(c)
    for key, lengths in (("q", [3, 0, 1]), ("k", [511, 0, 2])):
        c[key + "lens"] = lengths
        value = c["lengths"]["seqused_" + key]
        value.copy_(value.new_tensor(lengths))
    record = metadata(c)
    assert not torch.equal(record.cpu(), old.cpu())
    compare(c, invoke(c, record), record)
    compare(c, invoke(c, entry="public"), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "splits,q,k",
    [
        (1, (), ()),
        (3, (0, 0), (31, 3)),
        (0, (7, 2), (0, 0)),
        (1, (7, 2), (0, 0)),
        (3, (7, 2), (0, 0)),
    ],
)
def test_empty(representation, dtype, splits, q, k):
    c = make_case(representation, dtype, q=q, k=k, splits=splits)
    record = metadata(c)
    compare(c, invoke(c, entry="public"), record)


@pytest.mark.parametrize(
    "representation,splits", [(r, s) for r in ("compressed", "prefill") for s in (0, 1, 3)]
)
def test_fake(representation, splits):
    c = make_case(representation, "bf16", splits=splits, mask="window")
    real = invoke(c, entry="raw")
    with FakeTensorMode():
        fake = make_case(representation, "bf16", splits=splits, mask="window")
        record = metadata(fake)
        for entry in ("public", "op", "fake"):
            result = invoke(fake, record if entry != "public" else None, entry=entry)
            for x, y in zip(real, result):
                assert (x.shape, x.stride(), x.dtype, x.device) == (
                    y.shape,
                    y.stride(),
                    y.dtype,
                    y.device,
                )


@pytest.mark.parametrize(
    "representation,entry,splits",
    [
        ("compressed", "public", 0),
        ("prefill", "public", 3),
        ("compressed", "op", 1),
        ("prefill", "op", 1),
    ],
)
def test_compile(representation, entry, splits):
    c = make_case(representation, "bf16", splits=splits, mask="window")
    record = metadata(c)

    def call(q):
        return invoke(dict(c, t=dict(c["t"], q=q)), record if entry == "op" else None, entry=entry)

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    compare(c, compiled(c["t"]["q"]), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("entry", ["cpp", "raw"])
def test_raw_reuse_and_lifetime(representation, dtype, entry):
    spec = dict(
        splits=3, mask="window", q=(7, 0, 3), k=(65, 7, 3), used_q=(3, 0, 1), used_k=(33, 0, 2)
    )
    c = make_case(representation, dtype, strided=True, **spec)
    record = metadata(c)
    compare(c, invoke(c, record, entry=entry), record)
    second = make_case(representation, dtype, cap=30, scale=0, **spec)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        result = invoke(second, record, entry=entry)
        event = torch.npu.Event()
        event.record()
    del c
    del second["t"]
    gc.collect()
    event.synchronize()
    compare(second, result, record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("cap", [-1, float("nan"), float("inf"), 1e100, 1e-100])
def test_invalid_cap(representation, fake, cap):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        c = make_case(representation, cap=cap)
        for entry in ("public", "op"):
            with pytest.raises(RuntimeError, match="softcap"):
                invoke(c, entry=entry)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("fake", [False, True])
def test_metadata_and_length_static_rejections(representation, fake):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        c = make_case(representation)
        record = metadata(c)
        for bad in (record[:-1], record.to(torch.int64), record.repeat(2)[::2]):
            with pytest.raises(RuntimeError, match="metadata"):
                invoke(c, bad)
        for name, bad in (
            ("cu_seqlens_k", c["lengths"]["cu_seqlens_k"][:-1]),
            ("seqused_q", torch.zeros(2, dtype=torch.int32, device="npu")),
            ("max_seqlen_q", -1),
        ):
            invalid = dict(c, lengths=dict(c["lengths"], **{name: bad}))
            with pytest.raises(RuntimeError):
                invoke(invalid, entry="public")
        with pytest.raises(RuntimeError, match="not implemented"):
            api.get_scheduler_metadata(
                3,
                7,
                65,
                8,
                2 if representation == "prefill" else 1,
                192 if representation == "prefill" else 64,
                torch.tensor([65, 7, 3], device="npu", dtype=torch.int32),
                headdim_v=128 if representation == "prefill" else 512,
                cu_seqlens_q=c["lengths"]["cu_seqlens_q"],
                has_softcap=True,
            )
