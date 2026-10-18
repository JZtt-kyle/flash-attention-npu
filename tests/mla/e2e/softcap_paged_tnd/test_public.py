"""Public TND softcap numerics, allocation/used, records and lifetime."""

import contextlib
import itertools
import gc
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from tests.mla.support.public_cases.softcap_paged_tnd import (
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
    version = 46 if representation == "prefill" else 45 if spec.get("splits", 1) == 0 else 44
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
        used_k=(513, 31, 3),
        mask="window",
        splits=splits,
        strided=True,
    )
    old = metadata(c)
    for key, lengths in (("k", [511, 0, 2]),):
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
    spec = dict(splits=3, mask="window", q=(7, 0, 3), k=(65, 7, 3), used_k=(33, 0, 2))
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
def test_metadata_and_static_rejections(representation, fake):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        c = make_case(representation)
        record = metadata(c)
        for bad in (record[:-1], record.to(torch.int64), record.repeat(2)[::2]):
            with pytest.raises(RuntimeError, match="metadata"):
                invoke(c, bad)
        for name, bad in [
            ("cu_seqlens_q", c["lengths"]["cu_seqlens_q"][:-1]),
            ("seqused_k", torch.zeros(1, dtype=torch.int32, device="npu")),
            ("max_seqlen_q", -1),
        ]:
            with pytest.raises(RuntimeError):
                invoke(dict(c, lengths=dict(c["lengths"], **{name: bad})), entry="public")
        for bad in (
            c["page_table"].to(torch.int64),
            c["page_table"].repeat_interleave(2, dim=1)[:, ::2],
        ):
            with pytest.raises(RuntimeError):
                invoke(dict(c, page_table=bad), entry="public")


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits", [0, 1, 3])
def test_public_metadata_and_length_forms(representation, dtype, splits):
    c = make_case(
        representation, dtype, splits=splits, mask="window", q=(7, 0, 3), k=(96, 96, 96), page=48
    )
    record = metadata(c)
    torch.testing.assert_close(record, metadata(c, entry="cpp"), rtol=0, atol=0)
    for lengths in (None, 96):
        compare(
            c,
            invoke(dict(c, lengths=dict(c["lengths"], seqused_k=lengths)), record, entry="public"),
            record,
        )


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits,used", [(1, (3, 0, 1)), (3, (0, 0, 0)), (0, (3, 0, 1))])
def test_internal_query_used(representation, dtype, splits, used):
    c = make_case(representation, dtype, splits=splits, mask="window", used_q=used, strided=True)
    # Only reference tile partitions are needed here: effective lengths choose
    # the native schedule, while the oracle retains the original allocation.
    full = dict(
        c,
        lengths=dict(
            c["lengths"],
            seqused_q=None,
            cu_seqlens_q=torch.tensor(
                [0, *itertools.accumulate(used)], dtype=torch.int32, device="npu"
            ),
        ),
    )
    record = metadata(c, entry="op") if representation == "prefill" else metadata(full)
    compare(c, invoke(c), record)
    for entry in ("op", "raw", "cpp"):
        if representation == "prefill":
            compare(c, invoke(c, record, entry=entry), record)
        else:
            with pytest.raises(RuntimeError, match="Q-used"):
                invoke(c, record, entry=entry)
