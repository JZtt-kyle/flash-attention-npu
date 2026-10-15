"""TND PA causal: v11 prefixes/used, source math and complete public dispatch."""

from contextlib import nullcontext
import gc
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.public_cases import paged_tnd_production as production
from tests.mla.support.components.paged_attention import paged_probe as paged_probe
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault

from tests.mla.support.public_cases.causal_paged_public import compare, masked_rows as masked_rows

from tests.mla.support.public_cases.causal_paged_tnd_public import (
    ACCURACY_CASES as ACCURACY_CASES,
    PERFORMANCE_CASES as PERFORMANCE_CASES,
    REPRESENTATIVES as REPRESENTATIVES,
    TRAPS as TRAPS,
    call as call,
    causal_paged_tnd_attention as causal_paged_tnd_attention,
    check_constant as check_constant,
    check_spec as check_spec,
    component as component,
    dtype as dtype,
    forward_args as forward_args,
    make_case as make_case,
    metadata as metadata,
)


@pytest.mark.parametrize("spec", REPRESENTATIVES)
def test_representative(causal_paged_tnd_attention, paged_probe, spec):
    check_spec(causal_paged_tnd_attention, paged_probe, spec)


@pytest.mark.parametrize("page", range(16, 1025, 16))
def test_all_legal_pages(causal_paged_tnd_attention, paged_probe, page):
    check_spec(
        causal_paged_tnd_attention,
        paged_probe,
        production.tnd.config(page, q=(7, 7), lengths=(3, page + 1), uq=None, heads=3, repeat=True),
    )


@pytest.mark.parametrize("spec", ACCURACY_CASES)
def test_source_math(causal_paged_tnd_attention, paged_probe, spec):
    check_spec(causal_paged_tnd_attention, paged_probe, spec)


@pytest.mark.parametrize(
    "profile", ["ntd", "broadcast", "overlap", "row_65536", "combined_65537", "table_overlap"]
)
def test_strides_and_scale(causal_paged_tnd_attention, paged_probe, profile):
    check_spec(
        causal_paged_tnd_attention,
        paged_probe,
        production.tnd.config(16, profile, q=(7, 7), lengths=(33, 7), uq=None, heads=5),
        scale=-0.125,
    )


@pytest.mark.parametrize(
    "q,k,uq,capacity",
    [
        ([], [], None, 0),
        ([0, 0], [17, 0], None, 224),
        ([7, 7], [0, 0], None, 0),
        ([7, 7], [0, 0], [5, 0], 224),
    ],
)
def test_empty(causal_paged_tnd_attention, paged_probe, q, k, uq, capacity):
    check_spec(
        causal_paged_tnd_attention,
        paged_probe,
        production.tnd.config(
            112, q=q, lengths=k, uq=uq, capacity=capacity, pool=0 if capacity == 0 else 7
        ),
    )


@pytest.mark.parametrize("fake", [False, True])
def test_lengths_returns_offset_reuse(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs(dtype, sq=7)
        args["v"].fill_(3)
        for length in (None, 0, 3, 448, "tensor"):
            expected = (
                [3, 17, 113] if length == "tensor" else [448 if length is None else length] * 3
            )
            args["cache_seqlens"] = torch.tensor(expected, dtype=torch.int32, device="npu")
            record = metadata(args)
            storage = torch.empty(record.numel() + 2, dtype=torch.int32, device="npu")
            storage[1:-1].copy_(record)
            args["cache_seqlens"] = args["cache_seqlens"] if length == "tensor" else length
            for reused in (None, storage[1:-1]):
                result = call(dict(args, scheduler_metadata=reused))
                assert len(result) == 4 and result[2] is not result[3]
                assert all(
                    t.numel() == 0 and t.dtype == torch.get_default_dtype() for t in result[2:]
                )
                out = call(dict(args, scheduler_metadata=reused), lse=False)
                assert out.shape == result[0].shape
                if not fake:
                    check_constant(result, expected)
                    torch.testing.assert_close(out, result[0], rtol=0, atol=0)


def test_compile_updates_and_used(dtype):
    torch._dynamo.reset()
    args = production.small_inputs(dtype, sq=7)
    args["v"].fill_(3)
    for mode in ("implicit", "generated", "reuse"):

        def invoke(kw):
            if mode == "generated":
                kw = dict(kw, scheduler_metadata=metadata(kw))
            return call(kw)

        compiled = torch.compile(invoke, backend="eager", fullgraph=True)
        for lengths, allocation in zip(
            ([3, 17, 113], [0, 0, 0], [224, 1, 7]), ((7, 0, 2), (2, 7, 0), (3, 3, 3))
        ):
            a, b, c = allocation
            args["cu_seqlens_q"].copy_(torch.tensor([0, a, a + b, a + b + c], dtype=torch.int32))
            args["cache_seqlens"].copy_(torch.tensor(lengths, dtype=torch.int32))
            if mode == "reuse":
                args["scheduler_metadata"] = metadata(args)
            check_constant(compiled(args), lengths, allocations=allocation)
    args.pop("scheduler_metadata", None)
    args["cu_seqlens_q"].copy_(torch.tensor([0, 7, 7, 9], dtype=torch.int32))
    args["seqused_q"] = torch.tensor([5, 0, 1], dtype=torch.int32, device="npu")
    compiled = torch.compile(lambda kw: call(kw, "op"), backend="eager", fullgraph=True)
    check_constant(compiled(args), [224, 1, 7], used=[5, 0, 1])


@pytest.mark.parametrize("reuse", [False, True])
def test_stream_lifetime(dtype, reuse):
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    for step in range(3):
        with torch.npu.stream(creator):
            args = production.small_inputs(dtype, sq=7)
            args["v"].fill_(step + 1)
            args["cache_seqlens"].copy_(torch.tensor([3, 17, 0], dtype=torch.int32))
            if reuse:
                args["scheduler_metadata"] = metadata(args)
                args["cache_seqlens"] = args["cache_seqlens"].clone()
        consumer.wait_stream(creator)
        with torch.npu.stream(consumer), forbid_host_readback():
            results.append(call(args))
        del args
        gc.collect()
        with torch.npu.stream(creator):
            pressure = [torch.empty(8192, dtype=dtype, device="npu") for _ in range(16)]
        del pressure
    consumer.synchronize()
    for step, result in enumerate(results):
        check_constant(result, [3, 17, 0], step + 1)
    creator.synchronize()


def test_record_allows_new_pages_stride_and_scale(causal_paged_tnd_attention, paged_probe):
    spec = production.tnd.config(112, q=(7, 0, 3), lengths=(3, 17, 225), uq=None)
    original = make_case(causal_paged_tnd_attention, paged_probe, spec)
    record = metadata(forward_args(original))
    before = record.cpu().clone()
    for profile, repeat, scale in (("pa02_both", True, 0.0), ("ntd", False, -0.125)):
        case = make_case(
            causal_paged_tnd_attention,
            paged_probe,
            dict(spec, profile=profile, repeat=repeat, pool=11),
            scale=scale,
        )
        result = call(dict(forward_args(case), scheduler_metadata=record))
        compare(causal_paged_tnd_attention, case, [result])
    torch.testing.assert_close(before, record.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("fake", [False, True])
def test_static_and_unopened_boundaries(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs(dtype)
        for change in (dict(num_splits=0), dict(attention_chunk=1), dict(softcap=1.0)):
            with pytest.raises(RuntimeError, match="not implemented"):
                call(dict(args, **change))
        record = metadata(args)
        for entry in ("op", "eager") if fake else ("op", "eager", "cpp"):
            with pytest.raises(RuntimeError, match="Q-used"):
                call(
                    dict(
                        args,
                        seqused_q=torch.ones(3, dtype=torch.int32, device="npu"),
                        scheduler_metadata=record,
                    ),
                    entry,
                )


@pytest.mark.parametrize("fake", [False, True])
def test_storage_bounds(fake, storage_fault):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs()
        for name in ("q", "qv", "k", "v"):
            for entry in ("public", "op", "eager") if fake else ("public", "op", "eager", "cpp"):
                with (
                    storage_fault(args[name]),
                    pytest.raises(RuntimeError, match="(input view exceeds|storage.*bounds)"),
                ):
                    call(args, entry)


@pytest.mark.parametrize("mode", TRAPS)
def test_release_trap(dtype, mode):
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "traps"
    directory.mkdir(exist_ok=True)
    path = directory / f"{str(dtype).split('.')[-1]}-{mode}.log"
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_causal_paged_tnd_error_probe",
        str(dtype).split(".")[-1],
        mode,
    ]
    with path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    text = path.read_text()
    assert (
        result.returncode != 0
        and "MLA_PAGED_CAUSAL_LAUNCHED" in text
        and "MLA_PAGED_CAUSAL_COMPLETED" not in text
    ), text[-3000:]
    assert any(
        token in text.lower()
        for token in (
            "507035",
            "aicore",
            "device task abort",
            "vector core",
            "acl stream synchronize",
        )
    ), text[-3000:]


@pytest.mark.parametrize("fake", [False, True])
def test_prefix_updates_and_metadata_boundaries(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs(dtype, sq=7)
        args["v"].fill_(3)
        args["cache_seqlens"].copy_(torch.tensor([3, 7, 0], dtype=torch.int32))
        for allocation in ((7, 0, 2), (2, 7, 0), (3, 3, 3)):
            prefix = [0]
            for q in allocation:
                prefix.append(prefix[-1] + q)
            args["cu_seqlens_q"].copy_(torch.tensor(prefix, dtype=torch.int32))
            record = metadata(args)
            for entry in ("public", "op", "eager") if fake else ("public", "op", "eager", "cpp"):
                result = call(dict(args, scheduler_metadata=record), entry)
                if not fake:
                    assert record[1].item() == record[2].item() == 11
                    check_constant(result, [3, 7, 0], allocations=allocation)
        for prefix, error in (
            (args["cu_seqlens_q"].to(torch.int64), "cu_seqlens_q"),
            # Python derives B from the prefix; C++ derives it from the table.
            (args["cu_seqlens_q"][:-1], "cu_seqlens_q|page_table"),
            (args["cu_seqlens_q"][::2], "cu_seqlens_q"),
        ):
            for entry in ("public", "op", "eager") if fake else ("public", "op", "eager", "cpp"):
                with pytest.raises(RuntimeError, match=error):
                    call(dict(args, cu_seqlens_q=prefix), entry)


def test_empty_allocation_with_positive_maxq(dtype):
    args = production.small_inputs(dtype, sq=0)
    args["max_seqlen_q"] = 7
    record = metadata(args)
    for entry in ("public", "op", "eager", "cpp"):
        result = call(dict(args, scheduler_metadata=record), entry)
        assert result[0].shape == (0, 8, 512) and result[1].shape == (8, 0)
    torch.npu.synchronize()
