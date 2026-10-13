"""TND causal varlen: allocation gaps, source/math, public boundaries and lifetime."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.mla_ref import mla_reference_pair
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.mla_source_compare import assert_mla_source_close
from tests.mla_validation_policy import public_cases
from tests.test_flash_attn_npu_v3_mla_bsnd_stride import make_view, strides_for
from tests.test_flash_attn_npu_v3_mla_causal_attention import (
    causal_attention as causal_attention,
    make_case,
    launch,
)
from tests.test_flash_attn_npu_v3_mla_contract import _forward
from tests.test_flash_attn_npu_v3_mla_bsnd_stride_production import storage_fault as storage_fault
from tests.test_flash_attn_npu_v3_mla_used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    used_case,
    execute,
)


def spec(q=(7, 0, 3), k=(33, 0, 17), uq=None, uk=None, h=16, **kwargs):
    return dict(q=list(q), kv=list(k), uq=uq, uk=uk, heads=h, **kwargs)


REPRESENTATIVES = [
    spec(uq=[5, 0, 3], uk=[3, 0, 0]),
    spec(q=(257, 3), k=(257, 513), h=1),
    spec(uq=[5, 0, 3], uk=[3, 0, 17], stride="independent"),
]
PERFORMANCE_CASES = public_cases(
    [
        spec(q=(3, 0, 1), k=(17, 0, 33), h=128),
        spec(q=(1, 7, 2), k=(1, 513, 31)),
        spec(q=(0, 3, 1, 0, 7), k=(33, 511, 0, 513, 1025), h=128),
        spec(q=(1, 2, 3), k=(16384, 8193, 17), h=8),
        spec(q=(257, 1, 0, 3), k=(33, 513, 17, 1), h=1),
        spec(q=(3, 0) * 21, k=(0, 513) * 21, h=128),
        *REPRESENTATIVES,
        spec(q=(129, 3), k=(513, 17), uq=[65, 1], uk=[257, 3], h=3),
        spec(q=(3, 2), k=(16384, 8193), uq=[1, 2], uk=[16384, 8193], h=8),
        spec(stride="ntd", h=128),
        spec(q=(5, 1), k=(17, 17), uq=[3, 1], uk=[17, 7], h=5, stride="combined_65536"),
        spec(q=(513, 257), k=(257, 129), h=3),
    ],
    group="tnd",
)
ACCURACY_CASES = [c for c in PERFORMANCE_CASES if c not in REPRESENTATIVES] + [
    spec(q=(7, 7, 7), k=(3, 7, 33), h=257),
    spec(q=(7, 3), k=(513, 33), uq=[0, 0], uk=[513, 0]),
    spec(q=(3, 7), k=(33, 513), uq=[1, 2], uk=[17, 257], sq=64, sk=1025),
    *[spec(scale=s, amplitude=a) for s, a in ((0.0, 1), (-0.1, 1), (None, 4), (None, 8))],
]


def effective(s):
    return s["q"] if s["uq"] is None else s["uq"], s["kv"] if s["uk"] is None else s["uk"]


def prepare(dtype, s, *, poison=True):
    gen = torch.Generator().manual_seed(20261013)
    raw, device, guards = {}, {}, []
    uq, uk = effective(s)
    for name, dim, offset in (("q", 64, 5), ("qv", 512, 11), ("k", 64, 19), ("v", 512, 29)):
        query = name in ("q", "qv")
        allocation, used = (s["q"], uq) if query else (s["kv"], uk)
        tensor = torch.randn((sum(allocation), s["heads"] if query else 1, dim), generator=gen).to(
            dtype
        )
        if query:
            tensor.mul_(s.get("amplitude", 1))
        start = 0
        if poison:
            for capacity, length in zip(allocation, used):
                tensor[start + length : start + capacity] = torch.nan
                start += capacity
        if s.get("stride"):
            shape = (1, *tensor.shape)
            profile = "bnsd" if s["stride"] == "ntd" else s["stride"]
            view, logical, checks = make_view(
                tensor.unsqueeze(0),
                strides_for(shape, profile, name),
                offset,
                overlap=profile in ("broadcast", "overlap"),
            )
            device[name], raw[name] = view[0], logical[0]
            guards += checks
        else:
            raw[name], device[name] = tensor, tensor.npu()
    args = dict(
        device,
        cu_seqlens_q=torch.tensor(
            list(itertools.accumulate(s["q"], initial=0)), dtype=torch.int32, device="npu"
        ),
        cu_seqlens_k=torch.tensor(
            list(itertools.accumulate(s["kv"], initial=0)), dtype=torch.int32, device="npu"
        ),
        max_seqlen_q=s.get("sq", max(s["q"], default=0)),
        max_seqlen_k=s.get("sk", max(s["kv"], default=0)),
        seqused_q=None
        if s["uq"] is None
        else torch.tensor(s["uq"], dtype=torch.int32, device="npu"),
        seqused_k=None
        if s["uk"] is None
        else torch.tensor(s["uk"], dtype=torch.int32, device="npu"),
        softmax_scale=s.get("scale"),
    )
    return dict(spec=s, raw=raw, args=args, guards=guards)


def call(args, entry="public"):
    if entry == "public":
        return api.flash_attn_varlen_func(**args, causal=True, num_splits=1, return_attn_probs=True)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_causal_tnd_forward(**args)
    return _forward(entry, dict(args, causal=True, num_splits=1))[:2]


def source_tile(probes, case, oracle):
    s = case["spec"]
    details = make_case(probes, dict(shape=(1, 1, 1, 1)))["details"]
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
    s, raw = case["spec"], case["raw"]
    uq, uk = effective(s)
    start_q = start_k = 0
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
                    scale=s.get("scale"),
                    m=tile[0],
                    n=tile[1],
                ),
            )
            logical = {}
            for name, data in raw.items():
                query = name in ("q", "qv")
                start, length = (start_q, q) if query else (start_k, k)
                logical[name] = data[start : start + length].unsqueeze(0).contiguous()
                source["inputs"][name].fill_(torch.nan)
                source["inputs"][name][:, :length].copy_(logical[name])
                source["device"][name].copy_(source["inputs"][name])
            assert launch(probes, source, "oracle") == 0
            ref_source = [t.cpu() for t in source["outputs"]["oracle"]]
            masked = max(0, q - k)
            assert (ref_source[0][:, :masked] == 0).all() and ref_source[1][
                :, :, :masked
            ].isposinf().all()
            ref_source[1][:, :, :masked] = -torch.inf
            refs = mla_reference_pair(**logical, softmax_scale=s.get("scale"), causal=True)
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
            if any(v["math"] != "passed" for row in reports for v in row.values()):
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
                    probes[1] / f"tnd-source-limit-{identity}-{b}.pt",
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
            dict(window_size=(0, 0)),
            dict(num_splits=2),
            dict(softcap=1.0),
            dict(pack_gqa=True),
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
                "tests.mla_causal_tnd_error_probe",
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
