"""Complete public BSND softcap dispatch, numerics, metadata and lifecycle."""

import contextlib
import gc
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.softcap_bsnd import (
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
    expected_version = (
        34 if representation == "prefill" else 33 if spec.get("splits", 1) == 0 else 32
    )
    assert record.cpu()[1:3].tolist() == [expected_version] * 2
    result = invoke(c, record)
    compare(c, result, record)
    for x, y in zip(result, invoke(c)):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    if all(k == c["shape"][3] for k in c["klens"]):
        for x, y in zip(result, invoke(c, entry="func")):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "spec", [pytest.param(s, id=s["id"]) for s in CASES if s["id"] in REPRESENTATIVE]
)
def test_representative(representation, dtype, spec):
    c = make_case(representation, dtype, **spec)
    record = metadata(c)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("spec", [pytest.param(s, id=s["id"]) for s in CASES if s["id"] in DAILY])
def test_daily(representation, dtype, spec):
    c = make_case(representation, dtype, **spec)
    record = metadata(c)
    compare(c, invoke(c, record), record)
    c["options"].update(softcap=30, softmax_scale=-0.23)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "splits,shape",
    [
        (1, (0, 7, 8, 31)),
        (3, (2, 0, 8, 31)),
        (0, (2, 7, 8, 0)),
        (1, (2, 7, 8, 0)),
        (3, (2, 7, 8, 0)),
    ],
)
def test_empty(representation, dtype, splits, shape):
    c = make_case(representation, dtype, shape=shape, splits=splits)
    record = metadata(c)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize(
    "representation,splits", [(r, s) for r in ("compressed", "prefill") for s in (0, 1, 3)]
)
def test_fake(representation, splits):
    c = make_case(representation, "bf16", splits=splits, mask="window")
    real = invoke(c)
    with FakeTensorMode():
        fake = make_case(representation, "bf16", splits=splits, mask="window")
        record = metadata(fake)
        result = invoke(fake, record)
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
        ("compressed", "cache", 0),
        ("prefill", "cache", 3),
        ("compressed", "func", 1),
        ("prefill", "func", 1),
    ],
)
def test_compile(representation, entry, splits):
    c = make_case(representation, "bf16", splits=splits, mask="window")
    record = metadata(c)

    def call(q):
        case = dict(c, t=dict(c["t"], q=q))
        return invoke(case, record if entry == "cache" else None, entry=entry)

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    compare(c, compiled(c["t"]["q"]), record)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("entry", ["cpp", "raw"])
def test_raw_used_and_lifetime(representation, dtype, entry):
    c = make_case(representation, dtype, splits=3, mask="window", strided=True)
    record = metadata(c)
    c["qlens"][-1] = 2
    c["usedq"] = torch.tensor(c["qlens"], dtype=torch.int32, device="npu")
    compare(c, invoke(c, entry=entry), record)
    c.pop("usedq")
    c["qlens"][-1] = c["shape"][1]
    # Metadata does not retain input addresses, cap value, scale or stride.
    second = make_case(representation, dtype, splits=3, mask="window", cap=30, scale=0)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        result = invoke(second, record)
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
        with pytest.raises(RuntimeError, match="softcap"):
            invoke(c)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("fake", [False, True])
def test_metadata_static_rejections(representation, fake):
    with FakeTensorMode() if fake else contextlib.nullcontext():
        c = make_case(representation)
        record = metadata(c)
        for invalid in (record[:-1], record.to(torch.int64), record.repeat(2)[::2]):
            with pytest.raises(RuntimeError, match="metadata"):
                invoke(c, invalid)
        with pytest.raises(RuntimeError, match="not implemented"):
            api.get_scheduler_metadata(
                2,
                7,
                64,
                8,
                2 if representation == "prefill" else 1,
                192 if representation == "prefill" else 64,
                c["lengths"],
                headdim_v=128 if representation == "prefill" else 512,
                has_softcap=True,
                page_size=16,
            )


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("layout", ["paged", "paged_tnd"])
@pytest.mark.parametrize("fake", [False, True])
def test_other_layouts_rejected(representation, layout, fake):
    from tests.mla.support.public_cases.contract import _inputs, _forward

    with FakeTensorMode() if fake else contextlib.nullcontext():
        args = _inputs(representation, layout)
        args["softcap"] = 15
        with pytest.raises(RuntimeError, match="softcap"):
            _forward("raw", args)
