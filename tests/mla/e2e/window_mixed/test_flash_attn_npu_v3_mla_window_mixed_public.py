"""v14 mixed window cache: physical KV capacity, prefix/used separation and reuse."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault
from tests.mla.unit.window_bsnd.test_flash_attn_npu_v3_mla_window_attention import (
    window_attention as window_attention,
)
from tests.mla.e2e.window_tnd.test_flash_attn_npu_v3_mla_window_tnd_public import (
    compare as compare_window,
)
from tests.mla.support.mla_window_mixed_cases import (
    spec,
    REPRESENTATIVES,
    COST_CASES,
    ACCURACY_CASES,
)
from tests.mla.support.components.used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    used_case,
    execute,
)
from tests.mla.support.components.window_attention import make_case

from tests.mla.support.public_cases.window_mixed_public import (
    call as call,
    metadata as metadata,
    prepare as prepare,
)


def compare(probes, case, results, tile):
    # The reference takes flattened physical allocations, as TND does. KV starts
    # are multiples of BSND capacity; used lengths never change those starts.
    mapped = dict(case, args=dict(case["args"], max_seqlen_k=case["spec"]["capacity"]))
    return compare_window(probes, mapped, results, tile)


def source_tile(probes, case, oracle):
    s = case["spec"]
    details = make_case(probes, dict(shape=(1, 1, 1, 1), window=(-1, -1)))["details"]
    item = used_case(
        2,
        s["q"],
        s["kv"],
        s["uq"],
        s["uk"],
        heads=s["heads"],
        aic=int(details[2]),
        ratio=2,
        sq=case["args"]["max_seqlen_q"],
        sk=s["capacity"],
    )
    row = execute(oracle, [item])[0]
    assert row[0] == 0
    return row[1:3]


def check_spec(probes, s, oracle):
    case = prepare(probes[0]["dtype"], s)
    args = case["args"]
    before = {name: t.cpu() for name, t in args.items() if isinstance(t, torch.Tensor)}
    with forbid_host_readback():
        entries = (
            ("op", "eager", "cpp") if s["uq"] is not None else ("public", "op", "eager", "cpp")
        )
        results = [call(args, entry)[:2] for entry in entries]
        if s["uq"] is None and s["uk"] is not None:
            record = metadata(args)
            results += [call(dict(args, scheduler_metadata=record), e)[:2] for e in entries]
    compare(probes, case, results, source_tile(probes, case, oracle))
    for name, original in before.items():
        torch.testing.assert_close(args[name].cpu(), original, rtol=0, atol=0, equal_nan=True)
    for expected, actual in case["guards"]:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    return case, results[0]


@pytest.mark.parametrize("s", REPRESENTATIVES)
def test_representative(window_attention, used_oracle, s):
    check_spec(window_attention, s, used_oracle)


@pytest.mark.parametrize("s", COST_CASES)
def test_cost_boundary(window_attention, used_oracle, s):
    import resource
    import time
    import hashlib

    start = time.monotonic()
    torch.npu.reset_peak_memory_stats()
    check_spec(window_attention, s, used_oracle)
    identity = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]
    (window_attention[1] / f"mixed-cost-{identity}.json").write_text(
        json.dumps(
            dict(
                spec=s,
                seconds=time.monotonic() - start,
                host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                npu_peak_bytes=torch.npu.max_memory_allocated(),
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("s", ACCURACY_CASES)
def test_source_math(window_attention, used_oracle, s):
    check_spec(window_attention, s, used_oracle)


@pytest.mark.parametrize("profile", ["ntd", "broadcast", "overlap", "row_65536", "combined_65537"])
def test_strides(window_attention, used_oracle, profile):
    check_spec(
        window_attention, spec(q=(3, 1), k=(17, 17), capacity=17, h=5, stride=profile), used_oracle
    )


@pytest.mark.parametrize("flags", range(4))
def test_optional_used(window_attention, used_oracle, flags):
    check_spec(
        window_attention,
        spec(uq=[5, 0, 3] if flags & 1 else None, implicit_k=not flags & 2),
        used_oracle,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "s",
    [
        spec(q=(), k=(), capacity=33, sq=7),
        spec(q=(0, 0), k=(17, 0), sq=7),
        spec(q=(3, 1), k=(0, 0), capacity=0),
        spec(q=(0, 0), k=(0, 0), capacity=0, sq=7),
    ],
)
def test_empty(dtype, s):
    args = prepare(dtype, s)["args"]
    with forbid_host_readback():
        record = metadata(args)
        results = [
            call(dict(args, scheduler_metadata=r), e)[:2]
            for r in (None, record)
            for e in ("public", "op", "eager", "cpp")
        ]
    assert record.cpu().tolist()[:6] == [
        0x4D4C4131,
        14,
        14,
        413 + 8 * len(s["q"]),
        1 if dtype == torch.float16 else 2,
        0,
    ]
    for out, lse in results:
        assert out.shape == (sum(s["q"]), s["heads"], 512) and lse.shape == (
            s["heads"],
            sum(s["q"]),
        )
        assert (out.cpu() == 0).all() and lse.cpu().isposinf().all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
def test_boundaries(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = prepare(dtype, spec())["args"]
        record = metadata(args)
        assert record.shape == (437,) and record.dtype == torch.int32
        for e in ("public", "op", "eager") if fake else ("public", "op", "eager", "cpp"):
            result = call(dict(args, scheduler_metadata=record), e)
            assert result[0].shape == (10, 16, 512) and result[1].shape == (16, 10)
            for name, value in (
                ("cu_seqlens_q", args["cu_seqlens_q"].to(torch.int64)),
                ("cu_seqlens_q", args["cu_seqlens_q"][:-1]),
                ("cache_seqlens", torch.empty(3, dtype=torch.int64, device="npu")),
                ("cache_seqlens", torch.empty(6, dtype=torch.int32, device="npu")[::2]),
                ("softmax_scale", float("inf")),
                ("window_size", (-(2**63), 1)),
                ("max_seqlen_q", -1),
                ("q", args["q"].detach().requires_grad_()),
                ("qv", args["qv"][..., :-1]),
                ("scheduler_metadata", record[:-1]),
                ("scheduler_metadata", record.to(torch.int64)),
            ):
                with pytest.raises(RuntimeError, match="MLA|FlashAttention"):
                    call(dict(args, **{name: value}), e)
        for e in ("op", "eager") if fake else ("op", "eager", "cpp"):
            with pytest.raises(RuntimeError, match="Q-used with explicit metadata"):
                call(
                    dict(
                        args,
                        seqused_q=torch.zeros(3, dtype=torch.int32, device="npu"),
                        scheduler_metadata=record,
                    ),
                    e,
                )
        for opts in (
            dict(num_splits=2),
            dict(pack_gqa=True),
        ):
            p = dict(args)
            p.pop("seqused_q")
            p["k_cache"], p["v_cache"] = p.pop("k"), p.pop("v")
            with pytest.raises(RuntimeError, match="MLA"):
                api.flash_attn_with_kvcache(**p, **opts)


def test_reuse_views_none_integer_and_scale(window_attention, used_oracle):
    dtype = window_attention[0]["dtype"]
    s = spec(q=(5, 3), k=(33, 33), h=3)
    base = prepare(dtype, s)
    record = metadata(base["args"])
    snapshot = record.cpu().clone()
    for profile, scale in (("independent", None), ("ntd", 0.0), ("row_65536", -0.1)):
        case = prepare(dtype, dict(s, stride=profile, scale=scale))
        for lengths in (None, 33, case["args"]["cache_seqlens"]):
            result = call(dict(case["args"], cache_seqlens=lengths, scheduler_metadata=record))
            assert len(result) == 4 and result[2] is not result[3]
            assert result[2].numel() == result[3].numel() == 0
            assert result[2].dtype == result[3].dtype == torch.get_default_dtype()
            compare(
                window_attention,
                case,
                [result[:2]],
                source_tile(window_attention, case, used_oracle),
            )
        out = call(dict(case["args"], scheduler_metadata=record), lse=False)
        torch.testing.assert_close(out, result[0], rtol=0, atol=0)
    torch.testing.assert_close(record.cpu(), snapshot, rtol=0, atol=0)


def test_normalized_identity_reuse(window_attention, used_oracle):
    dtype = window_attention[0]["dtype"]
    for first, second in (((32, 6), (2**63 - 1, 2**63 - 1)), ((-1, 0), (-2, 0))):
        initial = prepare(dtype, spec(window=first))
        case = prepare(dtype, spec(window=second))
        record = metadata(initial["args"])
        torch.testing.assert_close(record.cpu(), metadata(case["args"]).cpu(), rtol=0, atol=0)
        results = [
            call(dict(case["args"], scheduler_metadata=record), entry)[:2]
            for entry in ("public", "op", "eager", "cpp")
        ]
        compare(window_attention, case, results, source_tile(window_attention, case, used_oracle))
        assert record.cpu().tolist()[1:3] == [14, 14]


def test_compile_updates_and_lifetime(window_attention, used_oracle):
    dtype = window_attention[0]["dtype"]
    specs = [spec(), spec(q=(5, 2, 3), k=(17, 1, 0), sq=7, window=(1, 2), causal=True)]
    initial = prepare(dtype, specs[0], poison=False)
    args = initial["args"]

    def generate_and_call(**kw):
        return call(dict(kw, scheduler_metadata=metadata(kw)))

    compiled = torch.compile(generate_and_call, backend="eager", fullgraph=True)
    for s in specs:
        update = prepare(dtype, s, poison=False)["args"]
        for name in ("cu_seqlens_q", "cache_seqlens"):
            args[name].copy_(update[name])
        args.update(window_size=s["window"], causal=s.get("causal", False))
        result = compiled(**args)
        case = dict(initial, spec=s)
        compare(
            window_attention, case, [result[:2]], source_tile(window_attention, case, used_oracle)
        )
    # Records own snapshots; source length storage can die after generation.
    parent = torch.npu.current_stream()
    streams = [torch.npu.Stream(), torch.npu.Stream()]

    def worker(i):
        torch.npu.set_device(args["q"].device)
        with torch.npu.stream(streams[i]):
            streams[i].wait_stream(parent)
            local = prepare(dtype, specs[i], poison=False)
            for name in local["raw"]:
                local["args"][name] = args[name].clone()
            record = metadata(local["args"])
            # Consume using different descriptors with the same values.
            local["args"]["cu_seqlens_q"] = local["args"]["cu_seqlens_q"].clone()
            local["args"]["cache_seqlens"] = local["args"]["cache_seqlens"].clone()
            return call(dict(local["args"], scheduler_metadata=record))[:2]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, range(2)))
    pressure = [torch.empty((10000,), device="npu") for _ in range(32)]
    torch.npu.synchronize()
    assert len(pressure) == 32
    for s, result in zip(specs, results):
        case = dict(initial, spec=s)
        compare(window_attention, case, [result], source_tile(window_attention, case, used_oracle))
    used = dict(args, seqused_q=torch.tensor([3, 1, 2], dtype=torch.int32, device="npu"))
    internal = torch.compile(lambda **kw: call(kw, "op"), backend="eager", fullgraph=True)
    case = dict(initial, spec=dict(specs[-1], uq=[3, 1, 2]))
    compare(
        window_attention, case, [internal(**used)], source_tile(window_attention, case, used_oracle)
    )


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("used", [False, True])
def test_storage_boundary(fake, used, storage_fault):
    with FakeTensorMode() if fake else nullcontext():
        args = prepare(
            torch.float16,
            spec(q=(3, 1), k=(17, 7), h=5, stride="row_65536", uq=[0, 0] if used else None),
        )["args"]
        for name in ("q", "k", "v", "qv"):
            entries = ("op", "eager") if fake else ("op", "eager", "cpp")
            if not used:
                entries += ("public",)
            for entry in entries:
                with (
                    storage_fault(args[name]),
                    forbid_host_readback(),
                    pytest.raises(RuntimeError, match="input view exceeds"),
                ):
                    call(args, entry)


def test_fake_v14_words():
    with FakeTensorMode():
        batch = ((1 << 32) - 1 - 413) // 8 + 1
        args = prepare(torch.float16, spec(q=(), k=(), capacity=0, h=1))["args"]
        args["k"] = torch.empty((batch, 0, 1, 64), dtype=torch.float16, device="npu")
        args["v"] = torch.empty((batch, 0, 1, 512), dtype=torch.float16, device="npu")
        args["cu_seqlens_q"] = torch.empty(batch + 1, dtype=torch.int32, device="npu")
        args["cache_seqlens"] = torch.empty(batch, dtype=torch.int32, device="npu")
        for entry in ("public", "op", "eager"):
            with pytest.raises(RuntimeError, match="metadata word count overflow"):
                call(args, entry)
        with pytest.raises(RuntimeError, match="metadata word count overflow"):
            metadata(args)


TRAPS = [
    "q-start",
    "q-descending",
    "q-tail",
    "q-used",
    "k-used",
    "empty-q",
    "empty-k",
    "stale-prefix",
    "stale-length",
    "old-record",
    "v9-record",
    "v12-record",
    "v13-record",
    "window",
    "window-args",
    "capacity",
    "signature",
]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("fault", TRAPS)
def test_release_trap(dtype, fault):
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "traps" / f"{dtype}-{fault}"
    directory.mkdir(parents=True)
    with (directory / "output.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.mla.support.mla_window_mixed_error_probe",
                dtype,
                fault,
                str(directory),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
    assert result.returncode == 0, (directory / "output.log").read_text()[-4000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] == "event" and report["extension_sha256"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
def test_metadata_entries_and_descriptors(dtype, fake):
    from tests.test_flash_attn_npu_v3_metadata_signature import metadata as generate
    from tests.mla.e2e.mixed.test_flash_attn_npu_v3_mla_mixed_metadata_public import arguments

    with FakeTensorMode() if fake else nullcontext():
        args = dict(
            arguments(dtype, q=(7, 0, 3), k=(3, 0, 17), capacity=33, heads=16),
            causal=False,
            window_size=(2, 1),
        )
        records = []
        for entry in ("public", "op") if fake else ("public", "op", "cpp"):
            with forbid_host_readback():
                record = generate(entry, args)
            assert record.shape == (437,) and record.dtype == torch.int32
            records.append(record)
            for key, value in (
                ("cache_seqlens", args["cache_seqlens"].cpu()),
                ("cu_seqlens_q", args["cu_seqlens_q"].to(torch.int64)),
                ("cu_seqlens_q", args["cu_seqlens_q"][:-1]),
                ("cu_seqlens_q", torch.empty(8, dtype=torch.int32, device="npu")[::2]),
                ("qkv_dtype", torch.float32),
                ("max_seqlen_k", 16385),
            ):
                with pytest.raises(RuntimeError, match="MLA"):
                    generate(entry, dict(args, **{key: value}))
        if not fake:
            for record in records:
                torch.testing.assert_close(record.cpu(), records[0].cpu(), rtol=0, atol=0)
                assert record.cpu().tolist()[:6] == [
                    0x4D4C4131,
                    14,
                    14,
                    437,
                    1 if dtype == torch.float16 else 2,
                    0,
                ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metadata_failure_record_and_recovery(dtype):
    args = prepare(dtype, spec())["args"]
    valid = metadata(args).cpu()
    for name, index, value in (
        ("cu_seqlens_q", 0, 1),
        ("cu_seqlens_q", 2, 6),
        ("cu_seqlens_q", 3, -1),
        ("cu_seqlens_q", 1, 8),
        ("cache_seqlens", 0, -1),
        ("cache_seqlens", 1, 34),
    ):
        original = args[name].clone()
        args[name][index] = value
        with forbid_host_readback():
            record = metadata(args)
        words = record.cpu().tolist()
        assert words[1:3] == [14, 14] and words[5] != 0 and not any(words[8:])
        args[name].copy_(original)
        torch.testing.assert_close(metadata(args).cpu(), valid, rtol=0, atol=0)


def test_compiled_storage_boundary(storage_fault):
    args = prepare(torch.float16, spec(q=(3, 1), k=(17, 7), h=5, stride="row_65536"))["args"]
    compiled = torch.compile(lambda **kw: call(kw), backend="eager", fullgraph=True)
    compiled(**args)
    torch.npu.synchronize()
    for name in ("q", "k", "v", "qv"):
        with storage_fault(args[name]), pytest.raises(RuntimeError, match="input view exceeds"):
            compiled(**args)
