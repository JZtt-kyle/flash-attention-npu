"""TND window varlen: allocation gaps, source/math, public boundaries and lifetime."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import resource
import time
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
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.unit.window_bsnd.test_flash_attn_npu_v3_mla_window_attention import (
    window_attention as window_attention,
)
from tests.mla.support.components.window_attention import make_case, launch, normalized_window
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault
from tests.mla.support.components.used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    used_case,
    execute,
)


from tests.mla.support.mla_window_tnd_cases import spec, REPRESENTATIVES, COST_CASES, ACCURACY_CASES

from tests.mla.support.public_cases.window_tnd_public import (
    call as call,
    effective as effective,
    prepare as prepare,
    reference_pairs as reference_pairs,
)


def source_tile(probes, case, oracle):
    s = case["spec"]
    details = make_case(probes, dict(shape=(1, 1, 1, 1), window=(-1, -1)))["details"]
    item = used_case(
        1,
        s["q"],
        s["kv"],
        s["uq"],
        s["uk"],
        heads=s["heads"],
        aic=int(details[2]),
        ratio=2,
        sq=case["args"]["max_seqlen_q"],
        sk=case["args"]["max_seqlen_k"],
    )
    row = execute(oracle, [item])[0]
    assert row[0] == 0
    return row[1:3]


def compare(probes, case, results, tile):
    s = case["spec"]
    uq, uk = effective(s)
    start_q = start_k = 0
    window = normalized_window(
        dict(
            shape=(len(uq), case["args"]["max_seqlen_q"], s["heads"], case["args"]["max_seqlen_k"]),
            window=s["window"],
            causal=s.get("causal", False),
        )
    )
    references = reference_pairs(case, window)
    reports = []
    outputs = [[t.cpu() for t in result] for result in results]
    for b, (q, k) in enumerate(zip(uq, uk)):
        if q:
            # Independent per-sample source input, using original source-selected
            # tile geometry. Padding preserves capacity without exposing gaps.
            source = make_case(
                probes,
                dict(
                    shape=(1, q, s["heads"], max(s["kv"])),
                    lengths=[k],
                    max_q=case["args"]["max_seqlen_q"],
                    max_k=case["args"]["max_seqlen_k"],
                    window=s["window"],
                    causal=s.get("causal", False),
                    scale=s.get("scale"),
                    m=tile[0],
                    n=tile[1],
                ),
            )
            logical, refs = references[b]
            for name, tensor in logical.items():
                source["inputs"][name].fill_(torch.nan)
                source["inputs"][name][:, : tensor.shape[1]].copy_(tensor)
                source["device"][name].copy_(source["inputs"][name])
            assert launch(probes, source, "oracle") == 0
            ref_source = [t.cpu() for t in source["outputs"]["oracle"]]
            masked = (
                q
                if k == 0
                else (
                    0
                    if window.kind == 0
                    else max(0, min(q, q - k - (window.right if window.kind == 2 else 0)))
                )
            )
            assert (ref_source[0][:, :masked] == 0).all() and ref_source[1][
                :, :, :masked
            ].isposinf().all()
            ref_source[1][:, :, :masked] = -torch.inf
            if not k:
                refs[1].fill_(-torch.inf)
                refs[3].fill_(-torch.inf)
            actual = []
            try:
                for result in outputs:
                    got = [
                        result[0][start_q : start_q + q].unsqueeze(0),
                        result[1][:, start_q : start_q + q].unsqueeze(0),
                    ]
                    actual.append(got)
                    reports.append(
                        {
                            name: assert_mla_source_close(
                                got[i], ref_source[i], refs[i], refs[i + 2], name=name
                            )
                            for i, name in enumerate(("out", "lse"))
                        }
                    )
            except AssertionError:
                torch.save(
                    dict(
                        spec=s,
                        sample=b,
                        inputs=logical,
                        source=ref_source,
                        refs=refs,
                        actual=actual,
                    ),
                    probes[1] / "tnd-source-failure.pt",
                )
                raise
            identity = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]
            torch.save(
                dict(
                    spec=s,
                    sample=b,
                    inputs=logical,
                    source=ref_source,
                    refs=refs,
                    actual=actual,
                ),
                probes[1] / f"window-tnd-source-{identity}-{b}.pt",
            )
        start_q += s["q"][b]
        start_k += s["kv"][b]
    identity = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]
    (probes[1] / f"tnd-source-{identity}.json").write_text(
        json.dumps(dict(spec=s, tile=tile, comparisons=reports), indent=2) + "\n"
    )
    return reports


def check_spec(probes, s, oracle):
    case = prepare(probes[0]["dtype"], s)
    before = {name: t.cpu() for name, t in case["args"].items() if isinstance(t, torch.Tensor)}
    with forbid_host_readback():
        results = [call(case["args"], entry) for entry in ("public", "op", "eager", "cpp")]
    compare(probes, case, results, source_tile(probes, case, oracle))
    for name, original in before.items():
        torch.testing.assert_close(
            case["args"][name].cpu(), original, rtol=0, atol=0, equal_nan=True
        )
    for expected, actual in case["guards"]:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    return case, results[0]


@pytest.mark.parametrize("s", REPRESENTATIVES)
def test_representative(window_attention, used_oracle, s):
    check_spec(window_attention, s, used_oracle)


@pytest.mark.parametrize("s", ACCURACY_CASES)
def test_source_math(window_attention, used_oracle, s):
    check_spec(window_attention, s, used_oracle)


@pytest.mark.parametrize("profile", ["ntd", "broadcast", "overlap", "row_65536", "combined_65537"])
def test_strides(window_attention, used_oracle, profile):
    # Overlapping views define their own logical values; keep them fully used.
    check_spec(window_attention, spec(q=(3, 1), k=(17, 7), h=5, stride=profile), used_oracle)


@pytest.mark.parametrize("flags", range(4))
def test_optional_used(window_attention, used_oracle, flags):
    check_spec(
        window_attention,
        spec(uq=[5, 0, 3] if flags & 1 else None, uk=[3, 0, 17] if flags & 2 else None),
        used_oracle,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "s",
    [
        spec(q=(), k=(), uq=[], uk=[]),
        spec(q=(0, 0), k=(17, 0), sq=7),
        spec(q=(3, 1), k=(0, 0), uq=[2, 1], sk=33),
        spec(q=(0, 0), k=(0, 0), sq=7, sk=33),
    ],
)
def test_empty(dtype, s):
    case = prepare(dtype, s)
    with forbid_host_readback():
        outputs = [call(case["args"], entry) for entry in ("public", "op", "eager", "cpp")]
    for out, lse in outputs:
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
        for entry in ("public", "op", "eager") if fake else ("public", "op", "eager", "cpp"):
            out, lse = call(args, entry)
            assert out.shape == (10, 16, 512) and lse.shape == (16, 10)
            for name, value in (
                ("cu_seqlens_q", args["cu_seqlens_q"].to(torch.int64)),
                ("cu_seqlens_k", args["cu_seqlens_k"][:-1]),
                ("cu_seqlens_k", torch.empty(8, dtype=torch.int32, device="npu")[::2]),
                ("seqused_q", torch.empty(2, dtype=torch.int32, device="npu")),
                ("seqused_k", torch.empty(3, dtype=torch.int64, device="npu")),
                ("softmax_scale", float("inf")),
                ("window_size", (-(2**63), 1)),
                ("window_size", (2, -(2**63))),
                ("max_seqlen_q", -1),
                ("max_seqlen_k", 16385),
                ("q", args["q"].detach().requires_grad_()),
                ("qv", args["qv"][..., :-1]),
            ):
                with pytest.raises(RuntimeError, match="MLA|FlashAttention"):
                    call(dict(args, **{name: value}), entry)
        for option in (
            dict(num_splits=2),
            dict(pack_gqa=True),
        ):
            with pytest.raises(RuntimeError, match="MLA"):
                api.flash_attn_varlen_func(**args, **option)
        with pytest.raises(RuntimeError, match="MLA"):
            call(
                dict(args, scheduler_metadata=torch.empty(434, dtype=torch.int32, device="npu")),
                "eager",
            )


def test_compile_updates_and_lifetime(window_attention, used_oracle):
    dtype = window_attention[0]["dtype"]
    specs = [
        spec(uq=[5, 0, 3], uk=[3, 0, 17]),
        spec(q=(5, 2, 3), k=(31, 2, 17), uq=[2, 1, 3], uk=[17, 1, 0], window=(-1, 2), causal=True),
    ]
    initial = prepare(dtype, spec(), poison=False)
    expected = []
    for s in specs:
        case = prepare(dtype, s, poison=False)
        for name in initial["raw"]:
            case["raw"][name] = initial["raw"][name]
            case["args"][name] = initial["args"][name]
        expected.append(call(case["args"]))
        compare(
            window_attention, case, [expected[-1]], source_tile(window_attention, case, used_oracle)
        )
    compiled = torch.compile(lambda **args: call(args), backend="eager", fullgraph=True)
    args = prepare(dtype, specs[0], poison=False)["args"]
    for name in initial["raw"]:
        args[name] = initial["args"][name]
    for i, s in enumerate(specs):
        updates = prepare(dtype, s, poison=False)["args"]
        for name in ("cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k"):
            args[name].copy_(updates[name])
        args.update(
            {
                name: updates[name]
                for name in ("window_size", "causal", "max_seqlen_q", "max_seqlen_k")
            }
        )
        for got, ref in zip(compiled(**args), expected[i]):
            uq, _ = effective(s)
            start = 0
            for capacity, length in zip(s["q"], uq):
                axis = 0 if got.ndim == 3 else 1
                torch.testing.assert_close(
                    got.narrow(axis, start, length), ref.narrow(axis, start, length), rtol=0, atol=0
                )
                start += capacity
    parent = torch.npu.current_stream()
    streams = [torch.npu.Stream(), torch.npu.Stream()]

    def worker(i):
        torch.npu.set_device(initial["args"]["q"].device)
        with torch.npu.stream(streams[i]):
            streams[i].wait_stream(parent)
            temporary = prepare(dtype, specs[i], poison=False)["args"]
            for name in initial["raw"]:
                temporary[name] = initial["args"][name].clone()
            return call(temporary)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(worker, range(2)))
    pressure = [torch.empty((10000,), device="npu") for _ in range(32)]
    torch.npu.synchronize()
    assert len(pressure) == 32
    for i, result in enumerate(outputs):
        bounds = dict(args, max_seqlen_q=max(specs[i]["q"]), max_seqlen_k=max(specs[i]["kv"]))
        case = dict(initial, spec=specs[i], args=bounds)
        compare(window_attention, case, [result], source_tile(window_attention, case, used_oracle))


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("used", [False, True])
def test_storage_boundary(fake, used, storage_fault):
    with FakeTensorMode() if fake else nullcontext():
        s = spec(
            q=(3, 1),
            k=(17, 7),
            h=5,
            stride="row_65536",
            uq=[0, 0] if used else None,
            uk=[0, 0] if used else None,
        )
        args = prepare(torch.float16, s)["args"]
        for name in ("q", "k", "v", "qv"):
            for entry in ("public", "op", "eager") if fake else ("public", "op", "eager", "cpp"):
                with (
                    storage_fault(args[name]),
                    forbid_host_readback(),
                    pytest.raises(RuntimeError, match="TND input view exceeds"),
                ):
                    call(args, entry)


def test_compiled_storage_boundary(storage_fault):
    args = prepare(torch.float16, spec(q=(3, 1), k=(17, 7), h=5, stride="row_65536"))["args"]
    compiled = torch.compile(lambda **kwargs: call(kwargs), backend="eager", fullgraph=True)
    compiled(**args)
    torch.npu.synchronize()
    for name in ("q", "k", "v", "qv"):
        with storage_fault(args[name]), pytest.raises(RuntimeError, match="TND input view exceeds"):
            compiled(**args)


def test_fake_v13_words_without_used():
    with FakeTensorMode():
        args = prepare(torch.float16, spec(q=(), k=(), h=1))["args"]
        batch = ((1 << 32) - 1 - 413) // 8 + 1
        args["cu_seqlens_q"] = torch.empty(batch + 1, dtype=torch.int32, device="npu")
        args["cu_seqlens_k"] = torch.empty_like(args["cu_seqlens_q"])
        for entry in ("public", "op", "eager"):
            with pytest.raises(RuntimeError, match="metadata word count overflow"):
                call(args, entry)


TRAPS = [
    "q-start",
    "q-descending",
    "q-tail",
    "k-descending",
    "k-max",
    "k-start",
    "k-tail",
    "q-used",
    "k-used",
    "empty-q",
    "empty-k",
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
                "tests.mla.support.mla_window_tnd_error_probe",
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


@pytest.mark.parametrize("s", COST_CASES)
def test_cost_boundary(window_attention, used_oracle, s):
    torch.npu.reset_peak_memory_stats()
    started = time.monotonic()
    check_spec(window_attention, s, used_oracle)
    result = dict(
        spec=s,
        seconds=time.monotonic() - started,
        host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        device_peak_bytes=torch.npu.max_memory_allocated(),
    )
    identity = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]
    (window_attention[1] / f"tnd-cost-{identity}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_output_only(dtype):
    args = prepare(dtype, spec())["args"]
    out = api.flash_attn_varlen_func(**args)
    expected, _ = call(args)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
