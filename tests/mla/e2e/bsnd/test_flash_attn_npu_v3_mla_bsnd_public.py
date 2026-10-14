"""Public BSND lengths: source accuracy, snapshot reuse and stream lifetime."""

from contextlib import nullcontext
import json

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_attention import (
    CASES,
    bsnd_attention as bsnd_attention,
    launch_bsnd,
    make_bsnd_case,
    references,
    values_for,
)
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_metadata import check_source
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_metadata_aicpu import (
    bsnd_probe as bsnd_probe,
)
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    evaluate,
    length_artifacts as length_artifacts,
    length_oracle as length_oracle,
)
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import _forward, _inputs
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_kvcache import cached, generate, metadata_args
from tests.test_flash_attn_npu_v3_metadata_signature import metadata as call_metadata
from tests.mla.support.mla_profile_cases import (
    BSND_CASES,
    CASES as DENSE_CASES,
    S1G_CASES,
    KV_CASES,
    LONG_KV_CASES,
)


# Full real platform for public launch. Include a shape whose dense M=128
# grows to M=256 with unequal lengths, and the maximum single-row KV capacity.
PUBLIC_CASES = [(shape, lengths, None) for shape, lengths, _ in CASES] + [
    ((2, 10, 128, 513), [17, 513], None),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_performance_schedule_matrix(bsnd_probe, length_oracle, dtype):
    specs = [
        (shape, scale, None) for shape, scale in DENSE_CASES + S1G_CASES + KV_CASES + LONG_KV_CASES
    ] + BSND_CASES
    cases, records = [], []
    for shape, _, lengths in specs:
        case = values_for(bsnd_probe, shape, lengths)
        case[12] = bsnd_probe["platform"].block_dims[bsnd_probe["platform"].aic - 1]
        args = metadata_args(dtype, shape)
        if lengths is not None:
            args["cache_seqlens"] = torch.tensor(lengths, dtype=torch.int32, device="npu")
        record = api.get_scheduler_metadata(**args)
        records.append([x & 0xFFFFFFFF for x in record.cpu().tolist()])
        cases.append(case)
    name = "public-schedules-" + str(dtype).removeprefix("torch.")
    check_source(records, cases, length_oracle, bsnd_probe["directory"], name)
    native = evaluate(length_oracle, cases, automatic_fd=True)
    (bsnd_probe["directory"] / (name + "-dispatch.json")).write_text(
        json.dumps(
            [
                dict(
                    shape=shape,
                    scale=scale,
                    lengths=lengths,
                    native_fd=row[5] != 0,
                    native_schedule=row,
                )
                for (shape, scale, lengths), row in zip(specs, native)
            ],
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("spec", PUBLIC_CASES)
@pytest.mark.parametrize("scale", [None, 0.0, -0.23])
def test_public_actual_length_source_accuracy(bsnd_attention, bsnd_probe, spec, scale):
    case = make_bsnd_case(bsnd_attention, bsnd_probe, spec, scale=scale)
    empty = torch.tensor(case["logical_lengths"]) == 0
    launch_bsnd(bsnd_attention, case, "oracle")
    source = [t.cpu() for t in case["outputs"]["oracle"]]
    refs = references(case)
    record = generate(case["device"], case["lengths"])
    storage = torch.full((record.numel() + 2,), 123456, dtype=torch.int32, device="npu")
    storage[1:-1].copy_(record)
    before = storage.cpu().clone()
    input_before = {n: t.cpu().view(torch.uint8).clone() for n, t in case["device"].items()}
    length_before = None if case["lengths"] is None else case["lengths"].cpu().clone()
    errors = []
    for md in (None, record.clone(), storage[1:-1]):
        with forbid_host_readback():
            result = cached(case["device"], md, case["lengths"], scale)
        for i, name in enumerate(("out", "lse")):
            actual = result[i].cpu()
            torch.testing.assert_close(actual[~empty], source[i][~empty], rtol=0, atol=0)
            assert_fa_close(actual, refs[i], refs[i + 2], name=name)
            if i == 0:
                assert (actual[empty] == 0).all()
            else:
                assert actual[empty].isneginf().all() and source[i][empty].isposinf().all()
            finite = torch.isfinite(refs[i])
            errors.append(
                float((actual[finite] - refs[i][finite]).abs().max()) if finite.any() else 0
            )
    torch.testing.assert_close(storage.cpu(), before, rtol=0, atol=0)
    for name, tensor in case["device"].items():
        assert torch.equal(tensor.cpu().view(torch.uint8), input_before[name])
    if length_before is not None:
        torch.testing.assert_close(case["lengths"].cpu(), length_before, rtol=0, atol=0)
    with (bsnd_attention[1] / "public-accuracy.jsonl").open("a") as f:
        f.write(json.dumps(dict(shape=spec[0], lengths=spec[1], scale=scale, errors=errors)) + "\n")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("entry", ["public", "op", "cpp", "public_fake", "op_fake"])
@pytest.mark.parametrize("values", [[0, 16], [1, 17], [0, 0], [17, 17]])
def test_public_length_metadata_entries(dtype, entry, values):
    with FakeTensorMode() if entry.endswith("fake") else nullcontext():
        args = metadata_args(dtype)
        args["cache_seqlens"] = torch.tensor(values, dtype=torch.int32, device="npu")
        with forbid_host_readback():
            record = call_metadata(entry, args)
        assert record.shape == (403,) and record.dtype == torch.int32
        if not entry.endswith("fake"):
            row = record.cpu().tolist()
            assert row[:8] == [0x4D4C4131, 2, 2, 403, 1 if dtype == torch.float16 else 2, 0, 0, 0]
            assert row[401:] == values


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("length", [None, 0, 1, 16, 17])
def test_public_integer_and_direct_entries(dtype, fake, length):
    with FakeTensorMode() if fake else nullcontext():
        device = _inputs(dtype=dtype)
        lengths = torch.full(
            (2,), 17 if length is None else length, dtype=torch.int32, device="npu"
        )
        record = generate(device, lengths)
        for md in (None, record):
            with forbid_host_readback():
                results = [cached(device, md, length), cached(device, md, lengths)]
                results.append(
                    torch.ops.flash_attn_npu_3._mla_dense_forward(
                        **device, cache_seqlens=lengths, scheduler_metadata=md
                    )
                )
                results.append(
                    _forward(
                        "fake" if fake else "cpp",
                        dict(device, seqused_k=lengths, scheduler_metadata=md),
                    )[:2]
                )
            for result in results:
                assert result[0].shape == (2, 3, 4, 512) and result[1].shape == (2, 4, 3)
                if not fake:
                    assert (result[0].cpu() == 0).all()
                    want = torch.full((2, 4, 3), float(17 if length is None else length)).log()
                    torch.testing.assert_close(result[1].cpu(), want)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_public_length_updates_and_cross_stream_lifetime(dtype, compiled):
    def run(q, k, v, qv, lengths):
        device = dict(q=q, k=k, v=v, qv=qv)
        record = generate(device, lengths)
        return cached(device, record, lengths, -0.23)

    compute = torch.compile(run, backend="eager", fullgraph=True) if compiled else run
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    lengths = torch.full((2,), 17, dtype=torch.int32, device="npu")
    producer.wait_stream(torch.npu.current_stream())
    for values in ([17, 17], [0, 16], [1, 0], [0, 0], [17, 1]):
        # Reuse length storage only after the preceding consumption completed.
        producer.wait_stream(consumer)
        with torch.npu.stream(producer):
            lengths.copy_(torch.tensor(values, dtype=torch.int32, device="npu"))
            device = _inputs(dtype=dtype)
            record = generate(device, lengths).clone()
        consumer.wait_stream(producer)
        with torch.npu.stream(consumer):
            with forbid_host_readback():
                results.append((cached(device, record, lengths), values))
            # Dynamo inspects Tensor.cpu.__name__ during tracing; mocking it
            # breaks capture itself. Eager enqueue above forbids readback.
            with nullcontext() if compiled else forbid_host_readback():
                results.append(
                    (compute(*(device[n] for n in ("q", "k", "v", "qv")), lengths), values)
                )
        del device, record
        with torch.npu.stream(producer):
            pressure = torch.empty(32768, dtype=torch.int32, device="npu")
            pressure.fill_(-1)
    consumer.synchronize()
    for outputs, values in results:
        assert (outputs[0].cpu() == 0).all()
        want = torch.tensor(values, dtype=torch.float32).log().view(2, 1, 1).expand(2, 4, 3)
        torch.testing.assert_close(outputs[1].cpu(), want)


@pytest.mark.parametrize("entry", ["public_fake", "op_fake"])
def test_metadata_word_count_overflow(entry):
    with FakeTensorMode():
        args = metadata_args(torch.float16)
        args.update(batch_size=2**32 - 401, max_seqlen_q=0, max_seqlen_k=0)
        args["cache_seqlens"] = torch.empty(args["batch_size"], dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="word count overflow"):
            call_metadata(entry, args)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("entry", ["public", "op", "cpp"])
def test_public_old_record_shape_and_word_overflow(fake, entry):
    with FakeTensorMode() if fake else nullcontext():
        for batch, message in ((2, "shape"), (2**32 - 401, "word count overflow")):
            device = {
                name: torch.empty((batch, 0, 1, width), dtype=torch.float16, device="npu")
                for name, width in (("q", 64), ("qv", 512), ("k", 64), ("v", 512))
            }
            record = torch.empty(401, dtype=torch.int32, device="npu")
            with pytest.raises(RuntimeError, match=message):
                if entry == "public":
                    cached(device, record)
                elif entry == "op":
                    torch.ops.flash_attn_npu_3._mla_dense_forward(
                        **device, scheduler_metadata=record
                    )
                else:
                    _forward("fake" if fake else "cpp", dict(device, scheduler_metadata=record))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
def test_public_short_lengths_with_empty_query(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        device = {
            name: torch.full(
                (2, 0 if name in ("q", "qv") else 17, 1, width),
                torch.nan,
                dtype=dtype,
                device="npu",
            )
            for name, width in (("q", 64), ("qv", 512), ("k", 64), ("v", 512))
        }
        lengths = torch.tensor([0, 16], dtype=torch.int32, device="npu")
        record = generate(device, lengths)
        with forbid_host_readback():
            outputs = [cached(device, md, lengths) for md in (None, record)]
        for result in outputs:
            assert result[0].shape == (2, 0, 1, 512) and result[1].shape == (2, 1, 0)
            if not fake:
                assert result[0].cpu().numel() == result[1].cpu().numel() == 0
        if not fake:
            assert record.cpu()[401:].tolist() == [0, 16]
