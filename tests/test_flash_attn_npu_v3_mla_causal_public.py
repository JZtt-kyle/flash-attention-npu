"""BSND causal production/public paths, fixed source, independent math and contracts."""

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
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
from tests.mla_profile_cases import (
    CASES,
    S1G_CASES,
    KV_CASES,
    LONG_KV_CASES,
    BSND_CASES,
    STRIDE_CASES,
)
from tests.test_flash_attn_npu_v3_mla_causal_attention import (
    causal_attention as causal_attention,
    make_case,
    launch,
    references,
)
from tests.test_flash_attn_npu_v3_mla_contract import _forward, _inputs
from tests.test_flash_attn_npu_v3_metadata_signature import metadata as metadata_entry
from tests.test_flash_attn_npu_v3_mla_kvcache import metadata_args
from tests.test_flash_attn_npu_v3_mla_bsnd_stride import make_view, strides_for
from tests.test_flash_attn_npu_v3_mla_used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    used_case,
    execute,
)
from tests.mla_validation_policy import public_cases


PERFORMANCE_CASES = [
    dict(shape=s, scale=scale) for s, scale in CASES + S1G_CASES + KV_CASES + LONG_KV_CASES
]
PERFORMANCE_CASES += [dict(shape=s, scale=scale, lengths=k) for s, scale, k in BSND_CASES]
PERFORMANCE_CASES += [
    dict(shape=s, scale=scale, lengths=k, stride=p) for s, scale, k, p in STRIDE_CASES
]
PERFORMANCE_CASES += [dict(shape=(2, 513, 3, 257), lengths=[257, 129])]
FULL_PERFORMANCE_CASES = public_cases(PERFORMANCE_CASES)
PERFORMANCE_CASES = public_cases(FULL_PERFORMANCE_CASES, group="bsnd")
NATIVE_PREFLIGHT_CASES = [
    dict(shape=(2, 7, 16, 33), lengths=[3, 0]),
    dict(shape=(1, 7, 16, 8193)),
    FULL_PERFORMANCE_CASES[4],
    FULL_PERFORMANCE_CASES[24],
]
USED_CASES = [
    dict(shape=(2, 7, 16, 33), used=[5, 3], lengths=[3, 0]),
    dict(shape=(2, 129, 3, 513), used=[65, 129], lengths=[17, 513]),
    dict(shape=(2, 7, 129, 16384), used=[7, 1], lengths=[8193, 16384]),
    dict(shape=(2, 7, 16, 33), used=[0, 0], lengths=[0, 33]),
    dict(shape=(2, 7, 16, 33), used=[5, 3], lengths=[3, 0], stride="independent"),
    dict(shape=(2, 7, 16, 33), used=[5, 3], implicit_kv=True),
]
REPRESENTATIVES = [
    dict(shape=(2, 7, 16, 33), lengths=[3, 0]),
    dict(shape=(1, 257, 1, 257)),
    dict(shape=(10, 3, 86, 8193)),
    USED_CASES[0],
    USED_CASES[4],
    USED_CASES[5],
]
ACCURACY_CASES = (
    PERFORMANCE_CASES
    + USED_CASES
    + [dict(shape=(2, 7, h, 513), lengths=[512, 1]) for h in (1, 15, 17, 127, 129, 255, 257)]
    + [dict(shape=(2, 7, 3, 33), lengths=[3, 17], scale=s) for s in (0.0, -0.1, 0.23)]
)


def generate(device, lengths, causal=True):
    q, k = device["q"], device["k"]
    return api.get_scheduler_metadata(
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        1,
        64,
        lengths,
        qkv_dtype=q.dtype,
        headdim_v=512,
        causal=causal,
        num_splits=1,
    )


def cached(device, lengths=None, record=None, scale=None, causal=True):
    return api.flash_attn_with_kvcache(
        device["q"],
        device["k"],
        device["v"],
        qv=device["qv"],
        cache_seqlens=lengths,
        scheduler_metadata=record,
        softmax_scale=scale,
        causal=causal,
        num_splits=1,
        return_softmax_lse=True,
    )


def prepare(probes, spec):
    b, sq, h, sk = spec["shape"]
    case = make_case(probes, dict(spec, lengths=spec.get("lengths") or [sk] * b))
    raw = case["inputs"]
    used = spec.get("used", [sq] * b)
    for sample, length in enumerate(used):
        for name in ("q", "qv"):
            raw[name][sample, length:] = torch.nan
    device = {name: t.npu() for name, t in raw.items()}
    if spec.get("stride"):
        for name, offset in zip(raw, (5, 11, 19, 29)):
            device[name], raw[name], _ = make_view(
                raw[name],
                strides_for(raw[name].shape, spec["stride"], name),
                offset,
                overlap=spec["stride"] in ("broadcast", "overlap"),
            )
    lengths = torch.tensor(case["lengths"], dtype=torch.int32, device="npu")
    uq = torch.tensor(used, dtype=torch.int32, device="npu") if "used" in spec else None
    return raw, device, lengths, uq, case["details"]


def source_tiles(probes, spec, oracle, details):
    """Preserve the original allocation's independently selected source tile."""
    b, sq, heads, sk = spec["shape"]
    item = used_case(
        q=[sq] * b,
        kv=[sk] * b,
        uq=spec.get("used"),
        uk=None if spec.get("implicit_kv") else spec.get("lengths") or [sk] * b,
        heads=heads,
        aic=int(details[2]),
        ratio=2,
    )
    result = execute(oracle, [item])[0]
    assert result[0] == 0
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    (probes[1] / f"public-source-schedule-{identity}.json").write_text(
        json.dumps(dict(input=item, result=result), indent=2) + "\n"
    )
    return result[1:3]


def compare(probes, spec, raw, results, tile):
    b, sq, h, sk = spec["shape"]
    used = spec.get("used", [sq] * b)
    lengths = spec.get("lengths") or [sk] * b
    # Reuse b1's bounded sample parallelism and the same content cache. Internal
    # Q-used cases have different Q lengths, so keep their individual slices.
    full_refs = (
        references(dict(shape=spec["shape"], inputs=raw, lengths=lengths, scale=spec.get("scale")))
        if "used" not in spec
        else None
    )
    reports = []
    for sample, (q_length, k_length) in enumerate(zip(used, lengths)):
        if q_length == 0:
            continue
        source_case = make_case(
            probes,
            dict(
                shape=(1, q_length, h, sk),
                lengths=[k_length],
                scale=spec.get("scale"),
                m=tile[0],
                n=tile[1],
            ),
        )
        logical = {}
        for name, tensor in raw.items():
            host = tensor[
                sample : sample + 1, : q_length if name in ("q", "qv") else sk
            ].contiguous()
            source_case["inputs"][name].copy_(host)
            source_case["device"][name].copy_(host)
            logical[name] = host if name in ("q", "qv") else host[:, :k_length]
        assert launch(probes, source_case, "oracle") == 0
        source = [t.cpu() for t in source_case["outputs"]["oracle"]]
        refs = (
            [ref[sample : sample + 1] for ref in full_refs]
            if full_refs is not None
            else mla_reference_pair(**logical, softmax_scale=spec.get("scale"), causal=True)
        )
        empty = max(0, q_length - k_length)
        assert (source[0][:, :empty] == 0).all() and source[1][:, :, :empty].isposinf().all()
        if sk:
            source[1][:, :, :empty] = -torch.inf
            if k_length == 0:
                refs[1].fill_(-torch.inf)
                refs[3].fill_(-torch.inf)
        actual = []
        for result in results:
            got = [
                result[0].cpu()[sample : sample + 1, :q_length],
                result[1].cpu()[sample : sample + 1, :, :q_length],
            ]
            actual.append(got)
            try:
                reports.append(
                    {
                        name: assert_mla_source_close(
                            got[i], source[i], refs[i], refs[i + 2], name=name
                        )
                        for i, name in enumerate(("out", "lse"))
                    }
                )
            except AssertionError:
                identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[
                    :16
                ]
                torch.save(
                    dict(
                        spec=spec,
                        tile=tile,
                        inputs=logical,
                        source=source,
                        refs=refs,
                        actual=actual,
                    ),
                    probes[1] / f"public-source-failure-{identity}-{sample}.pt",
                )
                raise
        if any(v["math"] != "passed" for report in reports for v in report.values()):
            identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
            torch.save(
                dict(
                    spec=spec,
                    inputs=logical,
                    source=source,
                    refs=refs,
                    actual=actual,
                    comparisons=reports,
                ),
                probes[1] / f"public-source-limit-{identity}-{sample}.pt",
            )
    return reports


def check_spec(probes, spec, oracle):
    raw, device, lengths, uq, details = prepare(probes, spec)
    tile = source_tiles(probes, spec, oracle, details)
    scale = spec.get("scale")
    before = {name: tensor.cpu().clone() for name, tensor in device.items()}
    with forbid_host_readback():
        if uq is not None:
            args = dict(
                **device,
                seqused_q=uq,
                seqused_k=None if spec.get("implicit_kv") else lengths,
                softmax_scale=scale,
                causal=True,
            )
            results = [_forward(entry, args) for entry in ("eager", "cpp")]
        else:
            record = generate(device, lengths)
            offset_record = torch.empty(
                record.numel() + 2, dtype=torch.int32, device=record.device
            )[1:-1]
            offset_record.copy_(record)
            results = [
                cached(device, lengths, scale=scale),
                cached(device, lengths, offset_record, scale),
            ]
            if spec.get("lengths") is None:
                results += [
                    api.flash_attn_func(
                        **device, softmax_scale=scale, causal=True, return_attn_probs=True
                    ),
                    cached(device, None, record, scale),
                    cached(device, device["k"].shape[1], record, scale),
                ]
    if uq is None:
        words = record.cpu()
        assert words.shape == (410 + 8 * device["q"].shape[0],) and words[1:3].tolist() == [7, 7]
        assert words[409] == 2
        assert words[15:17].tolist() == tile
        assert words[417::8].tolist() == lengths.cpu().tolist()
        for result in results[:2]:
            assert len(result) == 4 and result[2].numel() == result[3].numel() == 0
    reports = compare(probes, spec, raw, results, tile)
    for name, tensor in device.items():
        torch.testing.assert_close(tensor.cpu(), before[name], rtol=0, atol=0, equal_nan=True)
    return reports


def test_representative(causal_attention, used_oracle):
    for spec in REPRESENTATIVES:
        check_spec(causal_attention, spec, used_oracle)


def test_representative_same_stream_ordering(causal_attention):
    """Two host threads update lengths and submit complete dependent chains."""
    shape = (2, 7, 16, 33)
    length_values = ([33, 33], [3, 0], [17, 33])
    expected = []
    for values in length_values:
        case = make_case(causal_attention, dict(shape=shape, lengths=values))
        assert launch(causal_attention, case, "oracle") == 0
        result = [tensor.cpu() for tensor in case["outputs"]["oracle"]]
        for b, length in enumerate(values):
            result[1][b, :, : max(0, shape[1] - length)] = -torch.inf
        expected.append(result)
    _, device, lengths, _, _ = prepare(causal_attention, dict(shape=shape))
    values = [torch.tensor(v, dtype=torch.int32, device="npu") for v in length_values]
    parent = torch.npu.current_stream()
    streams = [torch.npu.Stream(), torch.npu.Stream()]

    def worker(index):
        torch.npu.set_device(device["q"].device)
        with torch.npu.stream(streams[index]):
            streams[index].wait_stream(parent)
            inputs = {name: tensor.clone() for name, tensor in device.items()}
            current = torch.empty_like(lengths)
            results = []
            for step in range(6):
                variant = (index + step) % len(length_values)
                if step == 2:
                    # Evict the bounded host candidate while prior submissions
                    # remain outstanding. No device tensor or length is cached.
                    for heads in range(1, 34):
                        api.get_scheduler_metadata(
                            2,
                            7,
                            33,
                            heads,
                            1,
                            64,
                            current,
                            qkv_dtype=device["q"].dtype,
                            headdim_v=512,
                            causal=True,
                            num_splits=1,
                        )
                current.copy_(values[variant])
                record = generate(inputs, current)
                results.append((variant, cached(inputs, current, record)))
            return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, i) for i in range(2)]
        results = [future.result() for future in futures]
    pressure = [torch.full((426,), 123, dtype=torch.int32, device="npu") for _ in range(64)]
    torch.npu.synchronize()
    for group in results:
        for variant, tensors in group:
            for actual, source in zip(tensors[:2], expected[variant]):
                torch.testing.assert_close(actual.cpu(), source, rtol=0, atol=0)
    assert len(pressure) == 64


@pytest.mark.parametrize("spec", ACCURACY_CASES)
def test_public_source_math(causal_attention, used_oracle, spec):
    check_spec(causal_attention, spec, used_oracle)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("shape", [(2, 3, 4, 17), (0, 3, 4, 17), (2, 0, 4, 17), (2, 3, 4, 0)])
def test_metadata_entries(dtype, fake, shape):
    with FakeTensorMode() if fake else nullcontext():
        args = dict(metadata_args(dtype, shape), causal=True)
        records = [
            metadata_entry(entry, args)
            for entry in (("public", "op") if fake else ("public", "op", "cpp"))
        ]
        for record in records:
            assert record.dtype == torch.int32 and record.shape == (410 + 8 * shape[0],)
        if not fake:
            for record in records:
                torch.testing.assert_close(record.cpu(), records[0].cpu(), rtol=0, atol=0)
                assert record.cpu()[1:3].tolist() == [7, 7]
            if shape[0]:
                snapshot = records[0].cpu().clone()
                args["cache_seqlens"].fill_(-1)
                failed = metadata_entry("public", args).cpu()
                assert failed[5] != 0 and (failed[8:] == 0).all()
                torch.testing.assert_close(records[0].cpu(), snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,sq,sk", [(0, 7, 33), (2, 0, 33), (2, 7, 0)])
@pytest.mark.parametrize("used", [False, True])
def test_empty(dtype, batch, sq, sk, used):
    device = {
        name: torch.empty(
            (batch, sq if name in ("q", "qv") else sk, 3 if name in ("q", "qv") else 1, d),
            dtype=dtype,
            device="npu",
        )
        for name, d in (("q", 64), ("k", 64), ("v", 512), ("qv", 512))
    }
    lengths = torch.full((batch,), sk, dtype=torch.int32, device="npu")
    with forbid_host_readback():
        if used:
            out, lse = torch.ops.flash_attn_npu_3._mla_causal_bsnd_forward(
                **device,
                cache_seqlens=lengths,
                seqused_q=torch.full((batch,), sq, dtype=torch.int32, device="npu"),
            )
        else:
            record = generate(device, lengths)
            out, lse, _, _ = cached(device, lengths, record)
    assert out.shape == (batch, sq, 3, 512) and lse.shape == (batch, 3, sq)
    if batch and sq:
        assert (out.cpu() == 0).all() and lse.cpu().isposinf().all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "old-record",
        "old-causal",
        "used-reuse",
        "length-dtype",
        "length-shape",
        "length-stride",
        "scale",
        "split",
    ],
)
def test_static_rejections(dtype, fake, fault):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(dtype=dtype)
        device = {name: args[name] for name in ("q", "k", "v", "qv")}
        lengths = torch.full((2,), 17, dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="MLA"):
            if fault == "old-record":
                cached(device, lengths, generate(device, lengths, causal=False))
            elif fault == "old-causal":
                cached(device, lengths, generate(device, lengths), causal=False)
            elif fault == "used-reuse":
                torch.ops.flash_attn_npu_3._mla_causal_bsnd_forward(
                    **device,
                    cache_seqlens=lengths,
                    seqused_q=lengths,
                    scheduler_metadata=generate(device, lengths),
                )
            elif fault == "split":
                api.flash_attn_func(**device, causal=True, num_splits=2)
            elif fault == "scale":
                cached(device, lengths, scale=float("inf"))
            else:
                lengths = {
                    "length-dtype": lambda: lengths.long(),
                    "length-shape": lambda: lengths[:1],
                    "length-stride": lambda: torch.ones(4, dtype=torch.int32, device="npu")[::2],
                }[fault]()
                cached(device, lengths)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_representative_cache_returns(dtype, fake, default_dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        with FakeTensorMode() if fake else nullcontext():
            args = _inputs(dtype=dtype)
            device = {name: args[name] for name in ("q", "k", "v", "qv")}
            result = cached(device)
            assert len(result) == 4
            assert result[0].shape == device["qv"].shape and result[0].dtype == dtype
            assert result[1].shape == (2, 4, 3) and result[1].dtype == torch.float32
            assert result[2] is not result[3]
            for accumulator in result[2:]:
                assert accumulator.shape == (0,) and accumulator.dtype == default_dtype
                assert accumulator.device == device["q"].device
            if not fake:
                result[2].resize_(1)
                assert result[3].numel() == 0
                output = api.flash_attn_with_kvcache(
                    device["q"],
                    device["k"],
                    device["v"],
                    qv=device["qv"],
                    causal=True,
                    num_splits=1,
                )
                torch.testing.assert_close(output, result[0], rtol=0, atol=0)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "rank",
        "dtype",
        "qv-shape",
        "stride",
        "grad",
        "grad-k",
        "grad-v",
        "grad-qv",
        "heads",
        "batch",
        "scale",
    ],
)
def test_representative_func_boundary(dtype, fake, fault):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(dtype=dtype)
        if fault == "rank":
            args["q"] = args["q"][0]
        elif fault == "dtype":
            args["q"] = args["q"].float()
        elif fault == "qv-shape":
            args["qv"] = args["qv"][..., :-1]
        elif fault == "stride":
            args["q"] = torch.zeros(2, 3, 4, 128, dtype=dtype, device="npu")[..., ::2]
        elif fault == "grad":
            args["q"].requires_grad_(True)
        elif fault.startswith("grad-"):
            args[fault[5:]].requires_grad_(True)
        elif fault == "heads":
            args["q"] = args["q"][:, :, :0]
            args["qv"] = args["qv"][:, :, :0]
        elif fault == "batch":
            args["k"], args["v"] = args["k"][:1], args["v"][:1]
        else:
            args["softmax_scale"] = float("inf")
        with pytest.raises(RuntimeError, match="MLA|FlashAttention"):
            api.flash_attn_func(**args, causal=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_compile_and_lifecycle(dtype):
    args = _inputs(dtype=dtype)
    device = {name: args[name] for name in ("q", "k", "v", "qv")}
    lengths = torch.tensor([2, 17], dtype=torch.int32, device="npu")

    def run(q, k, v, qv, lengths):
        d = dict(q=q, k=k, v=v, qv=qv)
        return cached(d, lengths, generate(d, lengths))[:2]

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    expected = run(**device, lengths=lengths)
    for actual, ref in zip(compiled(**device, lengths=lengths), expected):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    func = torch.compile(
        lambda q, k, v, qv: api.flash_attn_func(
            q, k, v, qv=qv, causal=True, return_attn_probs=True
        ),
        backend="eager",
        fullgraph=True,
    )
    for actual, ref in zip(
        func(**device), api.flash_attn_func(**device, causal=True, return_attn_probs=True)
    ):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    record = generate(device, lengths)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream), forbid_host_readback():
        outputs = [
            cached(
                {name: t.clone() for name, t in device.items()},
                lengths.clone(),
                record.clone(),
                scale,
            )
            for scale in (None, 0.0, -0.1)
        ]
        scratch = [torch.empty_like(device["qv"]) for _ in range(32)]
        del scratch
    torch.npu.current_stream().wait_stream(stream)
    for output, scale in zip(outputs, (None, 0.0, -0.1)):
        expected = cached(device, lengths, scale=scale)
        for actual, ref in zip(output[:2], expected[:2]):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    lengths.fill_(1)
    updated = generate(device, lengths)
    assert updated.cpu()[417::8].tolist() == [1, 1]
    for a, b in zip(cached(device, lengths, updated)[:2], cached(device, lengths)[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "kind",
        "snapshot",
        "endpoints",
        "stale",
        "negative",
        "over",
        "q-used",
        "empty",
        "fd-first",
        "fd-middle",
        "fd-last",
    ],
)
def test_release_trap(tmp_path, dtype, fault):
    directory = tmp_path / f"{dtype}-{fault}"
    directory.mkdir()
    command = [
        sys.executable,
        "-m",
        "tests.mla_causal_public_error_probe",
        dtype,
        fault,
        str(directory),
    ]
    with (directory / "output.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, (directory / "output.log").read_text()
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] == "event" and report["fault"] == fault
    assert (
        report["extension_sha256"]
        == hashlib.sha256(Path(api.flash_attn_npu_3.__file__).read_bytes()).hexdigest()
    )
