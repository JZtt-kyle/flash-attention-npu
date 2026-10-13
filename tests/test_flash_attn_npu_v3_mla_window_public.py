"""Nonpaged BSND window public/production paths and v12 lifecycle."""

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.mla_ref import mla_reference_pair
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.mla_source_compare import assert_mla_source_close
from tests.test_flash_attn_npu_v3_mla_window_attention import (
    window_attention as window_attention,
    make_case,
    launch,
    normalized_window,
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

from tests.mla_window_public_cases import REPRESENTATIVES, COST_CASES, ACCURACY_CASES


def generate(device, lengths, window=(2, 1), causal=False):
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
        num_splits=1,
        window_size=window,
        causal=causal,
    )


def cached(device, lengths=None, record=None, scale=None, window=(2, 1), causal=False):
    return api.flash_attn_with_kvcache(
        device["q"],
        device["k"],
        device["v"],
        qv=device["qv"],
        cache_seqlens=lengths,
        scheduler_metadata=record,
        softmax_scale=scale,
        num_splits=1,
        window_size=window,
        causal=causal,
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
    batch, sq, heads, sk = spec["shape"]
    window = normalized_window(spec)
    reports = []
    sequences = list(zip(spec.get("used", [sq] * batch), spec.get("lengths", [sk] * batch)))

    def reference(sample):
        q_length, k_length = sequences[sample]
        if q_length == 0:
            return None
        logical = {}
        for name, tensor in raw.items():
            host = tensor[
                sample : sample + 1, : q_length if name in ("q", "qv") else sk
            ].contiguous()
            logical[name] = host if name in ("q", "qv") else host[:, :k_length]
        return logical, mla_reference_pair(
            **logical,
            softmax_scale=spec.get("scale"),
            causal=window.kind == 1,
            window_size=(window.left, window.right) if window.kind == 2 else (-1, -1),
        )

    # Same bounded sample pool and intra-op threads as the standalone probe.
    # Device/source execution stays on the caller's stream; only CPU math runs
    # concurrently, with identical logical tensors and content-cache identities.
    with ThreadPoolExecutor(max_workers=min(4, max(1, batch))) as pool:
        references = list(pool.map(reference, range(batch)))
    for sample, (q_length, k_length) in enumerate(sequences):
        if q_length == 0:
            continue
        source_spec = dict(
            spec,
            shape=(1, q_length, heads, sk),
            lengths=[k_length],
            max_q=sq,
            max_k=sk,
            m=tile[0],
            n=tile[1],
        )
        source_case = make_case(probes, source_spec)
        for name, tensor in raw.items():
            host = tensor[
                sample : sample + 1, : q_length if name in ("q", "qv") else sk
            ].contiguous()
            source_case["inputs"][name].copy_(host)
            source_case["device"][name].copy_(host)
        assert launch(probes, source_case, "oracle") == 0
        source = [t.cpu() for t in source_case["outputs"]["oracle"]]
        logical, refs = references[sample]
        empty = (
            q_length
            if k_length == 0
            else (
                0
                if window.kind == 0
                else max(
                    0,
                    min(q_length, q_length - k_length - (window.right if window.kind == 2 else 0)),
                )
            )
        )
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
            reports.append(
                {
                    name: assert_mla_source_close(
                        got[i], source[i], refs[i], refs[i + 2], name=name
                    )
                    for i, name in enumerate(("out", "lse"))
                }
            )
        identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
        # Retain numerical records for every new entry, including source limits.
        torch.save(
            dict(
                spec=spec,
                tile=tile,
                inputs=logical,
                source=source,
                refs=refs,
                actual=actual,
                comparisons=reports[-len(results) :],
            ),
            probes[1] / f"public-window-{identity}-{sample}.pt",
        )
    return reports


def check_spec(probes, spec, oracle):
    raw, device, lengths, uq, details = prepare(probes, spec)
    tile = source_tiles(probes, spec, oracle, details)
    scale = spec.get("scale")
    window, causal = spec["window"], spec.get("causal", False)
    before = {name: tensor.cpu().clone() for name, tensor in device.items()}
    with forbid_host_readback():
        if uq is not None:
            args = dict(
                **device,
                seqused_q=uq,
                seqused_k=None if spec.get("implicit_kv") else lengths,
                softmax_scale=scale,
                causal=causal,
                window_size_left=window[0],
                window_size_right=window[1],
            )
            results = [_forward(entry, args) for entry in ("eager", "cpp")]
        else:
            record = generate(device, lengths, window, causal)
            offset_record = torch.empty(
                record.numel() + 2, dtype=torch.int32, device=record.device
            )[1:-1]
            offset_record.copy_(record)
            results = [
                cached(device, lengths, scale=scale, window=window, causal=causal),
                cached(device, lengths, offset_record, scale, window, causal),
            ]
            if spec.get("lengths") is None:
                results += [
                    api.flash_attn_func(
                        **device,
                        softmax_scale=scale,
                        causal=causal,
                        window_size=window,
                        return_attn_probs=True,
                    ),
                    cached(device, None, record, scale, window, causal),
                    cached(device, device["k"].shape[1], record, scale, window, causal),
                ]
    if uq is None:
        words = record.cpu()
        assert words.shape == (413 + 8 * device["q"].shape[0],) and words[1:3].tolist() == [12, 12]
        assert words[409] == 2
        assert words[15:17].tolist() == tile
        assert words[417:-3:8].tolist() == lengths.cpu().tolist()
        for result in results[:2]:
            assert len(result) == 4 and result[2].numel() == result[3].numel() == 0
    reports = compare(probes, spec, raw, results, tile)
    for name, tensor in device.items():
        torch.testing.assert_close(tensor.cpu(), before[name], rtol=0, atol=0, equal_nan=True)
    return reports


@pytest.mark.parametrize("spec", REPRESENTATIVES)
def test_representative(window_attention, used_oracle, spec):
    check_spec(window_attention, spec, used_oracle)


@pytest.mark.parametrize("spec", COST_CASES)
def test_cost_boundary(window_attention, used_oracle, spec):
    started = time.monotonic()
    check_spec(window_attention, spec, used_oracle)
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    (window_attention[1] / f"public-cost-{identity}.json").write_text(
        json.dumps(
            dict(
                spec=spec,
                seconds=time.monotonic() - started,
                host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                npu_peak_allocated_bytes=torch.npu.max_memory_allocated(),
                reference_workers=min(4, spec["shape"][0]),
                reference_intraop_threads=torch.get_num_threads(),
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("spec", ACCURACY_CASES)
def test_public_source_math(window_attention, used_oracle, spec):
    check_spec(window_attention, spec, used_oracle)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
def test_metadata_entries_and_normalized_reuse(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        for shape in [(2, 3, 4, 17), (0, 3, 4, 17), (2, 0, 4, 17), (2, 3, 4, 0)]:
            args = dict(metadata_args(dtype, shape), window_size=(2, 1))
            records = [
                metadata_entry(e, args)
                for e in (("public", "op") if fake else ("public", "op", "cpp"))
            ]
            assert all(r.shape == (413 + 8 * shape[0],) and r.dtype == torch.int32 for r in records)
            if not fake:
                for r in records:
                    torch.testing.assert_close(r, records[0], rtol=0, atol=0)
                    assert r.cpu()[1:3].tolist() == [12, 12]
                if shape[0]:
                    args["cache_seqlens"].fill_(-1)
                    failed = metadata_entry("public", args).cpu()
                    assert failed[5] != 0 and (failed[8:] == 0).all()
        d = {n: t for n, t in _inputs(dtype=dtype).items() if n in ("q", "k", "v", "qv")}
        lengths = torch.tensor([2, 17], dtype=torch.int32, device="npu")
        # Different raw windows with the same normalized identity may reuse.
        r = generate(d, lengths, (16, 2))
        a = cached(d, lengths, r, window=(2**63 - 1, 2**63 - 1))
        if not fake:
            b = cached(d, lengths, window=(16, 2))
            for x, y in zip(a[:2], b[:2]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_empty_and_returns(dtype):
    for batch, sq, sk in [(0, 7, 33), (2, 0, 33), (2, 7, 0)]:
        d = {
            n: torch.empty(
                (batch, sq if n in ("q", "qv") else sk, 3 if n in ("q", "qv") else 1, dim),
                dtype=dtype,
                device="npu",
            )
            for n, dim in (("q", 64), ("k", 64), ("v", 512), ("qv", 512))
        }
        lengths = torch.full((batch,), sk, dtype=torch.int32, device="npu")
        r = generate(d, lengths)
        for used in (False, True):
            if used:
                o, lse = torch.ops.flash_attn_npu_3._mla_window_bsnd_forward(
                    **d,
                    window_left=2,
                    window_right=1,
                    causal=False,
                    cache_seqlens=lengths,
                    seqused_q=torch.full((batch,), sq, dtype=torch.int32, device="npu"),
                )
            else:
                o, lse, *_ = cached(d, lengths, r)
            assert o.shape == (batch, sq, 3, 512) and lse.shape == (batch, 3, sq)
            if batch and sq:
                assert (o.cpu() == 0).all() and lse.cpu().isposinf().all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
def test_static_boundaries_and_returns(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        d = {n: t for n, t in _inputs(dtype=dtype).items() if n in ("q", "k", "v", "qv")}
        lengths = torch.tensor([2, 17], dtype=torch.int32, device="npu")
        for options in [
            dict(num_splits=2),
            dict(softcap=1),
            dict(pack_gqa=True),
            dict(window_size=(-(2**63), 1)),
        ]:
            with pytest.raises(RuntimeError, match="MLA"):
                api.flash_attn_func(**d, **{"window_size": (2, 1), **options})
        for bad in [
            lengths.long(),
            lengths[:1],
            torch.empty(4, dtype=torch.int32, device="npu")[::2],
        ]:
            with pytest.raises(RuntimeError, match="MLA"):
                cached(d, bad)
        for words in [410 + 16, 401 + 2]:
            with pytest.raises(RuntimeError, match="MLA"):
                cached(d, lengths, torch.empty(words, dtype=torch.int32, device="npu"))
        with pytest.raises(RuntimeError, match="MLA"):
            torch.ops.flash_attn_npu_3._mla_window_bsnd_forward(
                **d,
                window_left=2,
                window_right=1,
                causal=False,
                seqused_q=lengths,
                scheduler_metadata=generate(d, lengths),
            )
        # Unopened TND/mixed/PA windows reject before device execution.
        prefix = torch.tensor([0, 3, 6], dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="not implemented"):
            api.flash_attn_with_kvcache(
                d["q"].flatten(0, 1),
                d["k"],
                d["v"],
                qv=d["qv"].flatten(0, 1),
                cu_seqlens_q=prefix,
                max_seqlen_q=3,
                window_size=(2, 1),
                num_splits=1,
            )
        with pytest.raises(RuntimeError, match="not implemented"):
            api.flash_attn_with_kvcache(
                d["q"],
                d["k"],
                d["v"],
                qv=d["qv"],
                page_table=torch.zeros((2, 1), dtype=torch.int32, device="npu"),
                window_size=(2, 1),
                num_splits=1,
            )
        previous = torch.get_default_dtype()
        try:
            for dt in [torch.float32, torch.float64]:
                torch.set_default_dtype(dt)
                r = cached(d, lengths)
                assert r[2] is not r[3] and all(t.shape == (0,) and t.dtype == dt for t in r[2:])
        finally:
            torch.set_default_dtype(previous)
        record = generate(d, lengths)
        cache_out = api.flash_attn_with_kvcache(
            d["q"],
            d["k"],
            d["v"],
            qv=d["qv"],
            cache_seqlens=lengths,
            scheduler_metadata=record,
            window_size=(2, 1),
            num_splits=1,
        )
        func_out = api.flash_attn_func(**d, window_size=(2, 1))
        assert cache_out.shape == func_out.shape == (*d["q"].shape[:-1], 512)
        if not fake:
            torch.testing.assert_close(cache_out, cached(d, lengths, record)[0], rtol=0, atol=0)
            torch.testing.assert_close(func_out, cached(d)[0], rtol=0, atol=0)


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
            q, k, v, qv=qv, window_size=(2, 1), return_attn_probs=True
        ),
        backend="eager",
        fullgraph=True,
    )
    for actual, ref in zip(
        func(**device), api.flash_attn_func(**device, window_size=(2, 1), return_attn_probs=True)
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
    assert updated.cpu()[417:-3:8].tolist() == [1, 1]
    for a, b in zip(cached(device, lengths, updated)[:2], cached(device, lengths)[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_stream_length_window_lifecycle(dtype):
    d = {n: t for n, t in _inputs(dtype=dtype).items() if n in ("q", "k", "v", "qv")}
    variants = [([17, 17], (2, 1)), ([2, 0], (-1, 1)), ([7, 17], (0, 0))]
    expected = [
        cached(d, torch.tensor(v, dtype=torch.int32, device="npu"), window=w) for v, w in variants
    ]
    parent = torch.npu.current_stream()
    streams = [torch.npu.Stream(), torch.npu.Stream()]

    def worker(i):
        torch.npu.set_device(d["q"].device)
        with torch.npu.stream(streams[i]):
            streams[i].wait_stream(parent)
            tensors = {n: t.clone() for n, t in d.items()}
            lengths = torch.empty(2, dtype=torch.int32, device="npu")
            outputs = []
            for step in range(6):
                v, w = variants[step % 3]
                lengths.copy_(torch.tensor(v, dtype=torch.int32, device="npu"))
                r = generate(tensors, lengths, w)
                outputs.append(cached(tensors, lengths, r, window=w))
            return outputs

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(worker, range(2)))
    pressure = [torch.empty((429,), dtype=torch.int32, device="npu") for _ in range(64)]
    torch.npu.synchronize()
    for group in outputs:
        for i, result in enumerate(group):
            for a, b in zip(result[:2], expected[i % 3][:2]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(pressure) == 64


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
        "window-kind",
        "window-left",
        "window-right",
        "window-stale",
    ],
)
def test_release_trap(tmp_path, dtype, fault):
    directory = tmp_path / f"{dtype}-{fault}"
    directory.mkdir()
    command = [
        sys.executable,
        "-m",
        "tests.mla_window_public_error_probe",
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
