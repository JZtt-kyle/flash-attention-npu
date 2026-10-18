"""Complete positive-cap BSND PA public behavior and independent page semantics."""

import contextlib
import gc
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.softcap_paged_bsnd import (
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
    version = 43 if representation == "prefill" else 42 if spec.get("splits", 1) == 0 else 41
    assert record.cpu()[1:3].tolist() == [version] * 2
    for entry, supplied in (("op", record), ("public", None), ("public", record)):
        compare(c, invoke(c, supplied, entry=entry), record)
    for name, raw in c["physical"].items():
        torch.testing.assert_close(c["t"][name].cpu(), raw, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(c["page_table"].cpu(), c["raw_pages"], rtol=0, atol=0)


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
    compare(c, invoke(c, record, entry="public"), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits", [0, 1, 3])
def test_representative_length_update(representation, dtype, splits):
    c = make_case(
        representation, dtype, splits=splits, mask="window", shape=(2, 7, 8, 513), strided=True
    )
    old = metadata(c)
    c["klens"] = [511, 0]
    c["lengths"].copy_(c["lengths"].new_tensor(c["klens"]))
    record = metadata(c)
    assert not torch.equal(old.cpu(), record.cpu())
    for supplied in (None, record):
        compare(c, invoke(c, supplied, entry="public"), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "splits,shape",
    [
        (1, (0, 7, 8, 65)),
        (3, (2, 0, 8, 65)),
        (0, (2, 7, 8, 0)),
        (1, (2, 7, 8, 0)),
        (3, (2, 7, 8, 0)),
    ],
)
def test_empty(representation, dtype, splits, shape):
    c = make_case(representation, dtype, splits=splits, shape=shape)
    record = metadata(c)
    compare(c, invoke(c, entry="public"), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("splits", [0, 1, 3])
def test_fake(representation, splits):
    c = make_case(representation, "bf16", splits=splits, mask="window")
    real = invoke(c, entry="raw")
    with FakeTensorMode():
        fake = make_case(representation, "bf16", splits=splits, mask="window")
        record = metadata(fake)
        for entry in ("public", "op", "fake"):
            result = invoke(fake, record, entry=entry)
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
    spec = dict(splits=3, mask="window", shape=(2, 7, 8, 65), lengths=[65, 3], page=16)
    c = make_case(representation, dtype, strided=True, **spec)
    record = metadata(c)
    compare(c, invoke(c, record, entry=entry), record)
    second = make_case(representation, dtype, cap=30, scale=0, repeat_pages=True, **spec)
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
def test_metadata_and_static_rejections(representation, fake):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        c = make_case(representation)
        record = metadata(c)
        for bad in (record[:-1], record.to(torch.int64), record.repeat(2)[::2]):
            with pytest.raises(RuntimeError, match="metadata"):
                invoke(c, bad)
        for name, bad in [
            ("lengths", torch.zeros(1, dtype=torch.int32, device="npu")),
            ("page_table", c["page_table"].to(torch.int64)),
            ("page_table", c["page_table"].repeat_interleave(2, dim=1)[:, ::2]),
        ]:
            with pytest.raises(RuntimeError):
                invoke(dict(c, **{name: bad}), entry="public")
        with pytest.raises(RuntimeError, match="not implemented"):
            api.get_scheduler_metadata(
                2,
                7,
                c["shape"][3],
                8,
                2 if representation == "prefill" else 1,
                192 if representation == "prefill" else 64,
                c["lengths"],
                headdim_v=128 if representation == "prefill" else 512,
                page_size=c["page"],
                cu_seqlens_q=torch.tensor([0, 7, 14], dtype=torch.int32, device="npu"),
                has_softcap=True,
            )


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits", [0, 1, 3])
def test_public_metadata_and_length_forms(representation, dtype, splits):
    c = make_case(representation, dtype, splits=splits, mask="window", shape=(2, 7, 8, 96), page=48)
    record = metadata(c)
    torch.testing.assert_close(record, metadata(c, entry="cpp"), rtol=0, atol=0)
    for lengths in (None, 96):
        compare(c, invoke(dict(c, lengths=lengths), record, entry="public"), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_internal_query_used(representation, dtype):
    c = make_case(representation, dtype, splits=3, mask="window", used_q=[3, 0], strided=True)
    record = metadata(c)
    compare(c, invoke(c), record)
    with pytest.raises(RuntimeError, match="Q-used"):
        invoke(c, record)
