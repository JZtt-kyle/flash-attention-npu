"""BSND PA window: v15, independent page/mask math and public/internal dispatch."""

from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.public_cases import paged_bsnd_production as production
from tests.mla.support.components.paged_attention import paged_probe as paged_probe
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault

from tests.mla.support import mla_window_paged_helpers as component
from tests.mla.support.mla_window_paged_cases import REPRESENTATIVES, ACCURACY_CASES, COST_CASES
from tests.mla.support.mla_window_paged_cases import PERFORMANCE_CASES as PERFORMANCE_CASES

from tests.mla.support.public_cases.window_paged_public import (
    TRAPS as TRAPS,
    call as call,
    check_constant as check_constant,
    check_spec as check_spec,
    compare as compare,
    dtype as dtype,
    forward_args as forward_args,
    make_case as make_case,
    masked_rows as masked_rows,
    metadata as metadata,
    window_paged_attention as window_paged_attention,
)


@pytest.mark.parametrize("spec", REPRESENTATIVES)
def test_representative(window_paged_attention, paged_probe, spec):
    check_spec(window_paged_attention, paged_probe, spec)


@pytest.mark.parametrize("page", range(16, 1025, 16))
def test_all_legal_pages(window_paged_attention, paged_probe, page):
    check_spec(
        window_paged_attention,
        paged_probe,
        component.config(page, q=(7, 7), lengths=(3, page + 1), uq=None, heads=3, repeat=True),
    )


@pytest.mark.parametrize("spec", [s for s in ACCURACY_CASES if s not in COST_CASES])
def test_source_math(window_paged_attention, paged_probe, spec):
    check_spec(window_paged_attention, paged_probe, spec)


@pytest.mark.parametrize(
    "profile", ["bnsd", "broadcast", "overlap", "row_65536", "combined_65537", "table_overlap"]
)
def test_strides_and_scale(window_paged_attention, paged_probe, profile):
    check_spec(
        window_paged_attention,
        paged_probe,
        component.config(16, profile, q=(7, 7), lengths=(33, 7), uq=None, heads=5),
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
def test_empty(window_paged_attention, paged_probe, q, k, uq, capacity):
    check_spec(
        window_paged_attention,
        paged_probe,
        component.config(
            112, q=q, lengths=k, uq=uq, capacity=capacity, pool=0 if capacity == 0 else 7
        ),
    )


@pytest.mark.parametrize("fake", [False, True])
def test_lengths_returns_offset_reuse(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs(dtype, sq=7)
        args["v"].fill_(3)
        for length in (None, 0, 3, 448, "tensor"):
            expected = [3, 113] if length == "tensor" else [448 if length is None else length] * 2
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
        for lengths in ([3, 113], [0, 0], [224, 7]):
            args["cache_seqlens"].copy_(torch.tensor(lengths, dtype=torch.int32))
            if mode == "reuse":
                args["scheduler_metadata"] = metadata(args)
            check_constant(compiled(args), lengths)
    args.pop("scheduler_metadata", None)
    args["seqused_q"] = torch.tensor([5, 1], dtype=torch.int32, device="npu")
    compiled = torch.compile(lambda kw: call(kw, "op"), backend="eager", fullgraph=True)
    check_constant(compiled(args), [224, 7], used=[5, 1])


@pytest.mark.parametrize("reuse", [False, True])
def test_stream_lifetime(dtype, reuse):
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    for step in range(3):
        with torch.npu.stream(creator):
            args = production.small_inputs(dtype, sq=7)
            args["v"].fill_(step + 1)
            args["cache_seqlens"].copy_(torch.tensor([3, 0], dtype=torch.int32))
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
        check_constant(result, [3, 0], step + 1)
    creator.synchronize()


def test_record_allows_new_pages_stride_and_scale(window_paged_attention, paged_probe):
    spec = component.config(112, q=(7, 7), lengths=(3, 225), uq=None)
    original = make_case(window_paged_attention, paged_probe, spec)
    record = metadata(forward_args(original))
    before = record.cpu().clone()
    for profile, repeat, scale in (("pa02_both", True, 0.0), ("bnsd", False, -0.125)):
        case = make_case(
            window_paged_attention,
            paged_probe,
            dict(spec, profile=profile, repeat=repeat, pool=11),
            scale=scale,
        )
        result = call(dict(forward_args(case), scheduler_metadata=record))
        compare(window_paged_attention, case, [result])
    torch.testing.assert_close(before, record.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("fake", [False, True])
def test_static_and_unopened_boundaries(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs(dtype)
        for change in (dict(num_splits=0), dict(softcap=1.0), dict(attention_chunk=1)):
            with pytest.raises(RuntimeError, match="not implemented"):
                call(dict(args, **change))
        record = metadata(args)
        for entry in ("op", "eager") if fake else ("op", "eager", "cpp"):
            with pytest.raises(RuntimeError, match="Q-used"):
                call(
                    dict(
                        args,
                        seqused_q=torch.ones(2, dtype=torch.int32, device="npu"),
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
        "tests.mla.support.mla_window_paged_error_probe",
        str(dtype).split(".")[-1],
        mode,
    ]
    with path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    text = path.read_text()
    assert (
        result.returncode != 0
        and "MLA_PAGED_WINDOW_LAUNCHED" in text
        and "MLA_PAGED_WINDOW_COMPLETED" not in text
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
def test_metadata_entries_and_window_updates(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs(dtype, sq=7)
        args["v"].fill_(3)
        for window, causal in [
            ((0, 0), False),
            ((-1, 2), False),
            ((2, -1), False),
            ((2, 7), True),
            ((99999, 99999), False),
        ]:
            kw = dict(args, window_size=window, causal=causal)
            r = metadata(kw)
            assert r.shape == (421,)
            result = call(dict(kw, scheduler_metadata=r))
            if not fake:
                check_constant(result, [113, 0], window=window, causal=causal)
                for raw in [
                    api.flash_attn_npu_3.get_scheduler_metadata,
                    api._get_scheduler_metadata_op,
                ]:
                    actual = raw(
                        2,
                        7,
                        448,
                        8,
                        1,
                        64,
                        512,
                        dtype,
                        args["cache_seqlens"],
                        None,
                        None,
                        None,
                        None,
                        None,
                        112,
                        0,
                        causal,
                        *window,
                        0,
                        False,
                        1,
                        None,
                        0,
                        None,
                        None,
                    )
                    torch.testing.assert_close(r, actual, rtol=0, atol=0)


def test_cost_boundary(window_paged_attention, paged_probe):
    import time
    import resource

    start = time.monotonic()
    for spec in COST_CASES:
        check_spec(window_paged_attention, paged_probe, spec)
    report = dict(
        seconds=time.monotonic() - start,
        host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        npu_peak_bytes=torch.npu.max_memory_allocated(),
    )
    (window_paged_attention[1] / "cost.json").write_text(json.dumps(report, indent=2) + "\n")
