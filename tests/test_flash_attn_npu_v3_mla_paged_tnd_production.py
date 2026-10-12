"""Internal production TND PA: real storage, allocation prefixes, v6 reuse and lifetime."""

import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla_metadata_error_probe import forbid_host_readback
from tests import test_flash_attn_npu_v3_mla_paged_attention as component
from tests import test_flash_attn_npu_v3_mla_tnd_paged_attention as tnd
from tests.test_flash_attn_npu_v3_mla_paged_attention import (
    paged_attention as paged_attention,
    paged_probe as paged_probe,
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
)
from tests.test_flash_attn_npu_v3_mla_bsnd_stride_production import storage_fault as storage_fault


PERFORMANCE_CASES = [dict(c, aic=None) for c in tnd.PERFORMANCE_CASES]


def make_case(probes, producer, spec, **kwargs):
    spec = dict(spec, aic=None)
    if spec["capacity"] is None:
        p = spec["page"]
        spec["capacity"] = ((max(spec["lengths"], default=0) + p - 1) // p + 1) * p
    return component.make_case(probes, producer, spec, full_launch=True, **kwargs)


def forward_args(case):
    return dict(
        case["devices"]["migrated"],
        page_table=case["tables"]["migrated"],
        softmax_scale=case["scale"],
        cache_seqlens=case["inputs"][2],
        seqused_q=case["inputs"][1],
        cu_seqlens_q=case["inputs"][0],
        max_seqlen_q=case["context"].query.maximum,
    )


def forward(args, metadata=None):
    return torch.ops.flash_attn_npu_3._mla_paged_tnd_forward(**args, scheduler_metadata=metadata)


def generate(args):
    q, k, table = (args[n] for n in ("q", "k", "page_table"))
    return api.get_scheduler_metadata(
        table.shape[0],
        args["max_seqlen_q"],
        table.shape[1] * k.shape[1],
        q.shape[1],
        1,
        64,
        args["cache_seqlens"],
        qkv_dtype=q.dtype,
        headdim_v=512,
        page_size=k.shape[1],
        cu_seqlens_q=args["cu_seqlens_q"],
        num_splits=1,
    )


def source_available(case):
    return bool(
        case["context"].batch
        and case["context"].query.maximum
        and case["schedule"].tiling.sk
        and sum(case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"])
    )


def check_result(case, result, *, source=True):
    q = case["devices"]["migrated"]["q"]
    assert result[0].shape == (*q.shape[:-1], 512) and result[0].dtype == q.dtype
    assert result[1].shape == (q.shape[1], q.shape[0])
    assert result[1].dtype == torch.float32
    actual = component.valid_outputs(case, [t.cpu() for t in result])
    used = case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"]
    if source and source_available(case):
        expected = component.valid_outputs(
            case, [t.cpu() for t in case["outputs"]["oracle"]], source=True
        )
        empty = torch.tensor(
            [k == 0 for q, k in zip(used, case["config"]["lengths"]) for _ in range(q)],
            dtype=torch.bool,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][:, ~empty], expected[1][:, ~empty], rtol=0, atol=0)
        assert expected[1][:, empty].isposinf().all()
    refs = component.references(case)
    errors = component.check_math(actual, refs)
    return dict(errors=errors, source_compared=source and source_available(case))


def check_case(probes, case, *, metadata=None):
    args = forward_args(case)
    with (
        forbid_host_readback(),
        patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("input copy")),
    ):
        if source_available(case):
            assert component.launch(probes, case, "oracle") == 0
        record = metadata
        if args["seqused_q"] is None and record is None:
            record = generate(args)
        results = [forward(args), forward(args, record)]
    if record is not None:
        assert [v & 0xFFFFFFFF for v in record.cpu().tolist()] == case["expected"]
    report = check_result(case, results[-1])
    for a, b in zip(*[component.valid_outputs(case, [t.cpu() for t in r]) for r in results]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    report.update(config=case["config"], scale=case["scale"], expected_metadata=case["expected"])
    digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:20]
    target = probes[1] / "tnd-production-accuracy"
    target.mkdir(exist_ok=True)
    torch.save(
        dict(
            host=case["host"],
            pages=case["table_host"],
            actual=[t.cpu() for t in results[-1]],
            source=[t.cpu() for t in case["outputs"]["oracle"]] if source_available(case) else None,
            report=report,
        ),
        target / (digest + ".pt"),
    )
    with (target / "results.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    return report


def small_inputs(dtype=torch.float16, *, sq=5, pool=7, width=4, used=False):
    args = {}
    allocations = (sq, 0, min(2, sq))
    for name, dim in (("q", 64), ("qv", 512)):
        base = torch.zeros((2 * sum(allocations) + 1, 8, dim + 16), dtype=dtype, device="npu")
        args[name] = base[1::2, :, :dim]
    for name, dim in (("k", 64), ("v", 512)):
        base = torch.zeros((pool, 225, 1, dim + 16), dtype=dtype, device="npu")
        args[name] = base[:, 1::2, :, :dim]
    table = torch.full((3, width + 3), -1, dtype=torch.int32, device="npu")
    args["page_table"] = table[:, 1 : 1 + width]
    if width and pool:
        args["page_table"].copy_(
            torch.arange(width, dtype=torch.int32, device="npu").remainder(pool)
        )
    args["cache_seqlens"] = torch.tensor(
        [min(113, width * 112), min(17, width * 112), 0], dtype=torch.int32, device="npu"
    )
    args["cu_seqlens_q"] = torch.tensor(
        [0, sq, sq, sum(allocations)], dtype=torch.int32, device="npu"
    )
    args["max_seqlen_q"] = sq
    if used:
        args["seqused_q"] = torch.tensor(
            [min(3, sq), 0, min(1, sq)], dtype=torch.int32, device="npu"
        )
    return args


def check_constant(result, dtype, allocations, used, kv, value):
    out, lse = (t.cpu() for t in result)
    assert out.shape == (sum(allocations), 8, 512) and lse.shape == (8, sum(allocations))
    pos = 0
    for allocation, length, n in zip(allocations, used, kv):
        torch.testing.assert_close(
            out[pos : pos + length], torch.full((length, 8, 512), value if n else 0, dtype=dtype)
        )
        torch.testing.assert_close(
            lse[:, pos : pos + length], torch.full((8, length), float(n)).log()
        )
        pos += allocation


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "mode", ("implicit-full", "full-reused", "generated", "reused", "used", "used-full")
)
def test_smoke(dtype, mode):
    used = mode in ("used", "used-full")
    args = small_inputs(dtype, used=used)
    args["v"].fill_(3)
    lengths, record = [113, 17, 0], None
    if mode in ("implicit-full", "full-reused", "used-full"):
        if mode == "full-reused":
            args["cache_seqlens"].fill_(448)
            record = generate(args)
        del args["cache_seqlens"]
        lengths = [448] * 3
    with forbid_host_readback():
        result = forward(args, generate(args) if mode == "reused" else record)
    check_constant(result, dtype, (5, 0, 2), (3, 0, 1) if used else (5, 0, 2), lengths, 3)


@pytest.mark.parametrize("page", range(16, 1025, 16))
@pytest.mark.parametrize("profile", ("contiguous", "pa02_k", "pa02_v", "pa02_both"))
def test_all_page_sizes(paged_attention, paged_probe, page, profile):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            tnd.config(
                page,
                profile,
                q=(5, 0, 3),
                lengths=(page + 1, 0, 2 * page - 3),
                uq=None if profile == "contiguous" else (3, 0, 1),
                heads=16,
            ),
        ),
    )


@pytest.mark.parametrize(
    "profile",
    ["ntd", "independent", "broadcast", "overlap", "table_broadcast", "table_overlap"]
    + [
        f"{axis}_{s}"
        for axis, s in itertools.product(
            ("row", "matrix", "head", "combined"), (65535, 65536, 65537)
        )
    ],
)
def test_independent_strides(paged_attention, paged_probe, profile):
    lengths = (257, 257) if profile == "table_broadcast" else (257, 17)
    check_case(
        paged_attention,
        make_case(paged_attention, paged_probe, tnd.config(112, profile, lengths=lengths)),
    )


@pytest.mark.parametrize(
    "profile",
    (
        "wide_q_1",
        "wide_q_2",
        "wide_qv_1",
        "wide_qv_2",
        "wide_k_0",
        "wide_k_1",
        "wide_v_0",
        "wide_v_1",
        "table_wide",
        "singleton",
    ),
)
def test_wide_addresses(paged_attention, paged_probe, profile):
    torch.npu.empty_cache()
    spec = tnd.config(16, profile, q=(2, 2), lengths=(17, 17), uq=(2, 1), heads=2, pool=2)
    if profile == "singleton":
        spec = tnd.config(16, profile, q=(0, 1), lengths=(0, 1), uq=None, heads=1, pool=1)
    check_case(paged_attention, make_case(paged_attention, paged_probe, spec))


@pytest.mark.parametrize(
    "spec",
    PERFORMANCE_CASES
    + [
        tnd.config(
            112, "ntd", (7, 0, 3, 9, 1), (17, 65, 513, 0, 1), (3, 0, 1, 7, 0), 171, repeat=True
        ),
        tnd.config(528, "independent", (5, 0, 3), (512, 0, 513), (2, 0, 1), 256),
        tnd.config(16, "independent", (3, 2), (16384, 8193), (1, 2), 8, capacity=16384),
        *[
            tnd.config(112, "independent", (5, 2), (127, 129), (1, 2), h)
            for h in (1, 15, 16, 127, 128, 129, 255, 256)
        ],
    ],
)
def test_task_ring(paged_attention, paged_probe, spec):
    check_case(paged_attention, make_case(paged_attention, paged_probe, spec))


@pytest.mark.parametrize("pool,capacity", ((7, 0), (0, 0), (0, 224), (7, 224)))
@pytest.mark.parametrize("used", (None, (3, 1), (0, 0)))
def test_empty_cache(paged_attention, paged_probe, pool, capacity, used):
    case = make_case(
        paged_attention,
        paged_probe,
        tnd.config(lengths=(0, 0), uq=used, pool=pool, capacity=capacity),
    )
    check_case(paged_attention, case)
    # PA never uses the nonpaged global-empty +inf branch, even at C=0/Np=0.
    out, lse = component.valid_outputs(case, [t.cpu() for t in forward(forward_args(case))])
    assert (out == 0).all() and lse.isneginf().all()


@pytest.mark.parametrize(
    "q,kv,used,capacity",
    (
        ([], [], None, 0),
        ([], [], [], 224),
        ([0, 0], [257, 1], None, 448),
        ([5, 2], [257, 1], [0, 0], 448),
    ),
)
def test_empty_query(paged_attention, paged_probe, q, kv, used, capacity):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            tnd.config(q=q, lengths=kv, uq=used, capacity=capacity),
        ),
    )


@pytest.mark.parametrize("scale", (0.0, -0.125, 0.25, None))
@pytest.mark.parametrize("perturb", range(4))
def test_scale_halves(paged_attention, paged_probe, scale, perturb):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            tnd.config(lengths=(1025, 513)),
            scale=scale,
            perturb=perturb,
            amplitude=4,
        ),
    )


@pytest.mark.parametrize(
    "profile", ("contiguous", "pa02_both", "independent", "broadcast", "table_overlap")
)
def test_record_reuse_with_other_pool_table_and_strides(paged_attention, paged_probe, profile):
    old = make_case(paged_attention, paged_probe, tnd.config(uq=None, lengths=(257, 17)))
    new = make_case(
        paged_attention,
        paged_probe,
        tnd.config(profile=profile, uq=None, pool=11, lengths=(257, 17)),
        scale=-0.125,
    )
    record = generate(forward_args(old))
    before = record.cpu().clone()
    check_case(paged_attention, new, metadata=record)
    torch.testing.assert_close(record.cpu(), before, rtol=0, atol=0)


def test_source_schedules(paged_attention, paged_probe, paged_oracles):
    cases = [
        make_case(paged_attention, paged_probe, spec)
        for spec in PERFORMANCE_CASES + [tnd.config(p, uq=None) for p in range(16, 1025, 16)]
    ]
    items = [c["item"] for c in cases]
    source = component.execute(paged_oracles[0], items)
    native = component.execute(paged_oracles[0], items, native=True)
    kind = 1 if paged_attention[0]["dtype"] == torch.float16 else 2
    for case, wanted in zip(cases, source):
        component.decode_record(case["expected"], case["item"], kind, wanted)
        if case["config"]["uq"] is None:
            assert [x & 0xFFFFFFFF for x in generate(forward_args(case)).cpu().tolist()] == case[
                "expected"
            ]
    (paged_attention[1] / "tnd-production-schedules.json").write_text(
        json.dumps(
            dict(
                items=items, source=source, native=native, metadata=[c["expected"] for c in cases]
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("mode", ("generated", "reused", "used"))
def test_stream_lifetime(dtype, mode):
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    for step in range(4):
        with torch.npu.stream(creator):
            args = small_inputs(dtype, used=mode == "used")
            args["v"].fill_(step + 1)
            record = generate(args) if mode == "reused" else None
        consumer.wait_stream(creator)
        with torch.npu.stream(consumer), forbid_host_readback():
            results.append(forward(args, record))
        del args, record
        gc.collect()
        with torch.npu.stream(creator):
            churn = [torch.full((8192,), -77, dtype=dtype, device="npu") for _ in range(16)]
        del churn
    consumer.synchronize()
    for step, result in enumerate(results):
        check_constant(
            result,
            dtype,
            (5, 0, 2),
            (3, 0, 1) if mode == "used" else (5, 0, 2),
            (113, 17, 0),
            step + 1,
        )
    creator.synchronize()


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("used", (False, True))
def test_lengths_and_prefix_regenerated_on_current_stream(dtype, used):
    args = small_inputs(dtype, used=used)
    args["v"].fill_(2)
    for allocations, effective, kv in (
        ((5, 0, 2), (3, 0, 1), (113, 17, 0)),
        ((3, 0, 4), (1, 0, 0), (0, 0, 0)),
        ((1, 3, 3), (0, 2, 3), (224, 113, 17)),
    ):
        args["cu_seqlens_q"].copy_(
            torch.tensor(list(itertools.accumulate([0, *allocations])), dtype=torch.int32)
        )
        if used:
            args["seqused_q"].copy_(torch.tensor(effective, dtype=torch.int32))
        args["cache_seqlens"].copy_(torch.tensor(kv, dtype=torch.int32))
        with forbid_host_readback():
            result = forward(args)
        check_constant(result, dtype, allocations, effective if used else allocations, kv, 2)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("maximum", (8, 17, 262144))
@pytest.mark.parametrize("used", (False, True))
def test_loose_max_query(dtype, maximum, used):
    args = small_inputs(dtype, used=used)
    args["max_seqlen_q"] = maximum
    args["v"].fill_(2)
    with forbid_host_readback():
        record = None if used else generate(args)
        result = forward(args, record)
    check_constant(result, dtype, (5, 0, 2), (3, 0, 1) if used else (5, 0, 2), (113, 17, 0), 2)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("batch", (0, 3))
@pytest.mark.parametrize("width", (0, 4))
@pytest.mark.parametrize("used", (False, True))
def test_empty_query_with_loose_max(dtype, batch, width, used):
    args = small_inputs(dtype, sq=0, width=width, used=used)
    args["max_seqlen_q"] = 17
    args["page_table"] = args["page_table"][:batch]
    args["cache_seqlens"] = args["cache_seqlens"][:batch]
    args["cu_seqlens_q"] = args["cu_seqlens_q"][: batch + 1]
    if used:
        args["seqused_q"] = args["seqused_q"][:batch]
    with forbid_host_readback():
        results = [forward(args)]
        if not used:
            results.append(forward(args, generate(args)))
    for out, lse in results:
        assert out.shape == (0, 8, 512) and lse.shape == (8, 0)
    torch.npu.synchronize()


@pytest.mark.parametrize("name", ("q", "qv", "k", "v", "page_table"))
def test_complete_storage_before_launch(name, storage_fault):
    args = small_inputs(used=True)
    args["seqused_q"].zero_()
    args["cache_seqlens"].zero_()
    with (
        storage_fault(args[name]),
        forbid_host_readback(),
        pytest.raises(RuntimeError, match="storage"),
    ):
        forward(args)


@pytest.mark.parametrize(
    "name", ("cu_seqlens_q", "cache_seqlens", "seqused_q", "scheduler_metadata")
)
def test_descriptor_storage_before_launch(name, storage_fault):
    # Reuse the existing host-only storage metadata injector, without resizing
    # or freeing the real allocation or poisoning the NPU runtime.
    args = small_inputs(used=name == "seqused_q")
    record = generate(args) if name == "scheduler_metadata" else None
    t = record if record is not None else args[name]
    # The fixture truncates to numel bytes; alias a longer allocated view to
    # expose a positive stride span beyond it, then restore the original size.
    fault = torch.empty_strided((2, t.numel()), (2 * t.numel(), 1), dtype=torch.int32, device="npu")
    fault[1].copy_(t)
    if record is not None:
        record = fault[1]
    else:
        args[name] = fault[1]
    with storage_fault(fault), pytest.raises(RuntimeError, match="storage"):
        forward(args, record)


@pytest.mark.parametrize(
    "fault",
    (
        "prefix_rank",
        "prefix_shape",
        "prefix_dtype",
        "prefix_device",
        "prefix_stride",
        "max_negative",
        "max_small",
        "max_overflow",
        "rank",
        "dtype",
        "device",
        "shape",
        "table_dtype",
        "table_device",
        "table_inner",
        "q_inner",
        "k_shape",
        "page",
        "length_dtype",
        "length_shape",
        "length_stride",
        "length_device",
        "used_shape",
        "used_dtype",
        "used_device",
        "used_stride",
        "used_reuse",
        "metadata_shape",
        "metadata_dtype",
        "metadata_stride",
        "metadata_device",
        "grad",
        "nan",
        "inf",
        "overflow",
    ),
)
def test_host_rejections(fault):
    args = small_inputs()
    record = None
    if fault.startswith("prefix_"):
        t = args["cu_seqlens_q"]
        args["cu_seqlens_q"] = {
            "prefix_rank": lambda: t.view(2, 2),
            "prefix_shape": lambda: t[:-1],
            "prefix_dtype": lambda: t.long(),
            "prefix_device": lambda: t.cpu(),
            "prefix_stride": lambda: torch.zeros(8, dtype=torch.int32, device="npu")[::2],
        }[fault]()
    elif fault.startswith("max_"):
        args["max_seqlen_q"] = {"max_negative": -1, "max_small": 1, "max_overflow": 2**32}[fault]
    elif fault == "rank":
        args["q"] = args["q"][0]
    elif fault == "dtype":
        args["q"] = args["q"].float()
    elif fault == "device":
        args["qv"] = args["qv"].cpu()
    elif fault == "shape":
        args["qv"] = args["qv"][:1]
    elif fault == "table_dtype":
        args["page_table"] = args["page_table"].long()
    elif fault == "table_device":
        args["page_table"] = args["page_table"].cpu()
    elif fault == "table_inner":
        args["page_table"] = args["page_table"][:, ::2]
    elif fault == "q_inner":
        args["q"] = torch.zeros((7, 8, 128), dtype=torch.float16, device="npu")[..., ::2]
    elif fault == "k_shape":
        args["k"] = args["k"][:1]
    elif fault == "page":
        args["k"], args["v"] = args["k"][:, :17], args["v"][:, :17]
    elif fault == "length_dtype":
        args["cache_seqlens"] = args["cache_seqlens"].long()
    elif fault == "length_shape":
        args["cache_seqlens"] = args["cache_seqlens"][:1]
    elif fault == "length_stride":
        args["cache_seqlens"] = torch.zeros(6, dtype=torch.int32, device="npu")[::2]
    elif fault == "length_device":
        args["cache_seqlens"] = args["cache_seqlens"].cpu()
    elif fault == "used_shape":
        args["seqused_q"] = torch.zeros(1, dtype=torch.int32, device="npu")
    elif fault == "used_dtype":
        args["seqused_q"] = torch.zeros(3, dtype=torch.int64, device="npu")
    elif fault == "used_device":
        args["seqused_q"] = torch.zeros(3, dtype=torch.int32)
    elif fault == "used_stride":
        args["seqused_q"] = torch.zeros(6, dtype=torch.int32, device="npu")[::2]
    elif fault == "used_reuse":
        record = generate(args)
        args["seqused_q"] = torch.zeros(3, dtype=torch.int32, device="npu")
    elif fault.startswith("metadata"):
        record = generate(args)
        if fault == "metadata_shape":
            record = record[:-1]
        elif fault == "metadata_dtype":
            record = record.long()
        elif fault == "metadata_stride":
            record = torch.zeros(2 * record.numel(), dtype=torch.int32, device="npu")[::2]
        else:
            record = record.cpu()
    elif fault == "grad":
        args["v"].requires_grad_()
    else:
        args["softmax_scale"] = {"nan": float("nan"), "inf": float("inf"), "overflow": 1e100}[fault]
    with forbid_host_readback(), pytest.raises(RuntimeError, match="MLA"):
        forward(args, record)


@pytest.mark.parametrize("kind", ("fp16", "bf16"))
@pytest.mark.parametrize(
    "fault",
    (
        "page_negative",
        "page_over",
        "unused_q_page",
        "empty_q_page",
        "stale_k",
        "stale_prefix",
        "prefix_start",
        "prefix_end",
        "prefix_descending",
        "prefix_max",
        "zero_k_prefix",
        "bsnd_record",
        "bad_k",
        "bad_q",
        "zero_k_bad_q",
        "empty_pool",
        "magic",
        "dtype",
        "empty_payload",
        "v5",
    ),
)
def test_release_trap(kind, fault, tmp_path):
    directory = (
        Path(os.environ.get("MLA_PAGED_PRODUCTION_DIR", tmp_path)) / kind / ("trap-" + fault)
    )
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla_paged_bsnd_production_error_probe",
        kind,
        fault,
        str(directory),
        "tnd-internal",
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, (directory / "process.log").read_text()[-4000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert "trap error" in report["error"] and report["elapsed_seconds"] < 15
