"""TND causal varlen: allocation gaps, source/math, public boundaries and lifetime."""

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
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_attention import (
    causal_attention as causal_attention,
)
from tests.mla.support.public_cases.contract import _forward
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault
from tests.mla.support.components.used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
)

from tests.mla.support.public_cases.causal_tnd_public import (
    PERFORMANCE_CASES as PERFORMANCE_CASES,
    REPRESENTATIVES as REPRESENTATIVES,
    call as call,
    compare as compare,
    effective as effective,
    prepare as prepare,
    source_tile as source_tile,
    spec as spec,
)


ACCURACY_CASES = [c for c in PERFORMANCE_CASES if c not in REPRESENTATIVES] + [
    spec(q=(7, 7, 7), k=(3, 7, 33), h=257),
    spec(q=(7, 3), k=(513, 33), uq=[0, 0], uk=[513, 0]),
    spec(q=(3, 7), k=(33, 513), uq=[1, 2], uk=[17, 257], sq=64, sk=1025),
    *[spec(scale=s, amplitude=a) for s, a in ((0.0, 1), (-0.1, 1), (None, 4), (None, 8))],
]


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
def test_representative(causal_attention, used_oracle, s):
    check_spec(causal_attention, s, used_oracle)


@pytest.mark.parametrize("s", ACCURACY_CASES)
def test_source_math(causal_attention, used_oracle, s):
    check_spec(causal_attention, s, used_oracle)


@pytest.mark.parametrize("profile", ["ntd", "broadcast", "overlap", "row_65536", "combined_65537"])
def test_strides(causal_attention, used_oracle, profile):
    # Overlapping views define their own logical values; keep them fully used.
    check_spec(causal_attention, spec(q=(3, 1), k=(17, 7), h=5, stride=profile), used_oracle)


@pytest.mark.parametrize("flags", range(4))
def test_optional_used(causal_attention, used_oracle, flags):
    check_spec(
        causal_attention,
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
                ("max_seqlen_q", -1),
                ("max_seqlen_k", 16385),
                ("q", args["q"].detach().requires_grad_()),
                ("qv", args["qv"][..., :-1]),
            ):
                with pytest.raises(RuntimeError, match="MLA|FlashAttention"):
                    call(dict(args, **{name: value}), entry)
        for option in (
            dict(num_splits=-1),
            dict(sm_margin=-1),
        ):
            with pytest.raises(RuntimeError, match="MLA"):
                api.flash_attn_varlen_func(**args, causal=True, **option)
        with pytest.raises(RuntimeError, match="MLA"):
            _forward(
                "eager",
                dict(
                    args,
                    causal=True,
                    scheduler_metadata=torch.empty(434, dtype=torch.int32, device="npu"),
                ),
            )


def test_compile_updates_and_lifetime(causal_attention, used_oracle):
    dtype = causal_attention[0]["dtype"]
    specs = [
        spec(uq=[5, 0, 3], uk=[3, 0, 17]),
        spec(q=(5, 2, 3), k=(31, 2, 17), uq=[2, 1, 3], uk=[17, 1, 0]),
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
            causal_attention, case, [expected[-1]], source_tile(causal_attention, case, used_oracle)
        )
    compiled = torch.compile(lambda **args: call(args), backend="eager", fullgraph=True)
    args = prepare(dtype, specs[0], poison=False)["args"]
    for name in initial["raw"]:
        args[name] = initial["args"][name]
    for i, s in enumerate(specs):
        updates = prepare(dtype, s, poison=False)["args"]
        for name in ("cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k"):
            args[name].copy_(updates[name])
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
        case = dict(initial, spec=specs[i], args=args)
        compare(causal_attention, case, [result], source_tile(causal_attention, case, used_oracle))


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


def test_fake_v8_words_without_used():
    with FakeTensorMode():
        args = prepare(torch.float16, spec(q=(), k=(), h=1))["args"]
        batch = ((1 << 32) - 1 - 410) // 8 + 1
        args["cu_seqlens_q"] = torch.empty(batch + 1, dtype=torch.int32, device="npu")
        args["cu_seqlens_k"] = torch.empty_like(args["cu_seqlens_q"])
        for entry in ("public", "op", "eager"):
            with pytest.raises(RuntimeError, match="used metadata word count overflow"):
                call(args, entry)


TRAPS = [
    "q-start",
    "q-descending",
    "q-tail",
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
                "tests.mla.support.mla_causal_tnd_error_probe",
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
