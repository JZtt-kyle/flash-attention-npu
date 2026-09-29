"""Public dense MLA metadata and read-only cache: contracts, accuracy and lifetime."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.common.mla_ref import mla_reference_pair
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_metadata_signature import metadata as call_metadata
from tests.test_flash_attn_npu_v3_mla_contract import _forward, _inputs
from tests.test_flash_attn_npu_v3_mla_dense import launch, make_case, make_rounding_case
from tests.test_flash_attn_npu_v3_mla_dense import probes as probes
from tests.test_flash_attn_npu_v3_mla_metadata import Args, Shape, WORDS
from tests.test_flash_attn_npu_v3_mla_metadata import probe as probe


@pytest.fixture(scope="module", autouse=True)
def require_910b():
    assert "Ascend910B" in torch.npu.get_device_name(0)


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture(params=["public", "op", "cpp", "public_fake", "op_fake"])
def metadata_entry(request):
    with FakeTensorMode() if request.param.endswith("fake") else nullcontext():
        yield request.param


@pytest.fixture(params=["eager", "cpp", "fake"])
def forward_entry(request):
    with FakeTensorMode() if request.param == "fake" else nullcontext():
        yield request.param


def metadata_args(dtype, shape=(2, 3, 4, 17)):
    b, sq, h, sk = shape
    return dict(
        batch_size=b,
        max_seqlen_q=sq,
        max_seqlen_k=sk,
        num_heads_q=h,
        num_heads_kv=1,
        headdim=64,
        headdim_v=512,
        qkv_dtype=dtype,
        cache_seqlens=torch.full((b + 2,), sk, dtype=torch.int32, device="npu")[1:-1],
        num_splits=1,
    )


def generate(device, lengths=None):
    q, k = device["q"], device["k"]
    if lengths is None:
        lengths = torch.full((q.shape[0],), k.shape[1], dtype=torch.int32, device=q.device)
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
    )


def cached(device, metadata=None, lengths=None, scale=None, return_lse=True):
    return api.flash_attn_with_kvcache(
        device["q"],
        device["k"],
        device["v"],
        qv=device["qv"],
        cache_seqlens=lengths,
        scheduler_metadata=metadata,
        softmax_scale=scale,
        num_splits=1,
        return_softmax_lse=return_lse,
    )


def test_metadata_official_entries(metadata_entry, dtype):
    result = call_metadata(metadata_entry, metadata_args(dtype))
    assert result.shape == (WORDS,) and result.dtype == torch.int32
    assert result.device.type == "npu" and result.is_contiguous()
    assert not hasattr(result, "_fa_scheduler_params")
    if not metadata_entry.endswith("fake"):
        assert result.cpu()[:8].tolist() == [
            0x4D4C4131,
            1,
            1,
            WORDS,
            1 if dtype == torch.float16 else 2,
            0,
            0,
            0,
        ]


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 7, 16, 65), (21, 2, 128, 511)])
def test_public_record_matches_host(probe, dtype, shape):
    args = metadata_args(dtype, shape)
    record = api.get_scheduler_metadata(**args)
    host = (ctypes.c_uint32 * WORDS)()
    launch_args = Args(
        0, WORDS, 1 if dtype == torch.float16 else 2, Shape(*shape), probe["platform"]
    )
    probe["lib"].metadata_host(ctypes.byref(launch_args), host)
    assert [x & 0xFFFFFFFF for x in record.cpu().tolist()] == list(host)


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("batch_size", 2**32),
        ("max_seqlen_q", 0),
        ("max_seqlen_q", 257),
        ("max_seqlen_q", 65),
        ("max_seqlen_k", 0),
        ("max_seqlen_k", 513),
        ("num_heads_q", 0),
        ("num_heads_q", 257),
        ("num_heads_kv", 2),
        ("qkv_dtype", torch.float32),
        ("headdim_v", 128),
        ("causal", True),
        ("window_size", (0, -1)),
        ("attention_chunk", 1),
        ("has_softcap", True),
        ("num_splits", 0),
        ("num_splits", 2),
        ("pack_gqa", False),
        ("pack_gqa", True),
        ("sm_margin", 1),
        ("page_size", 16),
        ("max_seqlen_k_new", 1),
        ("softcap", 0.0),
        ("softcap", 15.0),
        ("softmax_scale", 0.23),
        ("cu_seqlens_q", "tensor"),
        ("cu_seqlens_k_new", "tensor"),
        ("cache_leftpad", "tensor"),
    ],
)
def test_metadata_rejects_unsupported(metadata_entry, dtype, field, value):
    args = metadata_args(dtype)
    if value == "tensor":
        value = torch.zeros(3, dtype=torch.int32, device="npu")
    args[field] = value
    with pytest.raises(RuntimeError, match="MLA"):
        call_metadata(metadata_entry, args)


def malformed(tensor, fault):
    if fault == "dtype":
        return tensor.to(torch.int64)
    if fault == "device":
        return tensor.cpu()
    if fault == "rank":
        return tensor.view(1, -1)
    if fault == "size":
        return tensor[:-1]
    return tensor.repeat_interleave(2)[::2]


@pytest.mark.parametrize("fault", ["dtype", "device", "rank", "size", "stride"])
def test_metadata_length_tensor_boundary(metadata_entry, dtype, fault):
    args = metadata_args(dtype)
    args["cache_seqlens"] = malformed(args["cache_seqlens"], fault)
    with pytest.raises(RuntimeError, match="MLA: cache_seqlens"):
        call_metadata(metadata_entry, args)


@pytest.mark.parametrize("name", ["seqused_k", "scheduler_metadata"])
@pytest.mark.parametrize("fault", ["dtype", "device", "rank", "size", "stride"])
def test_forward_metadata_tensor_boundary(forward_entry, dtype, name, fault):
    args = _inputs(dtype=dtype)
    tensor = torch.zeros(2 if name == "seqused_k" else WORDS, dtype=torch.int32, device="npu")
    args[name] = malformed(tensor, fault)
    with pytest.raises(RuntimeError, match="MLA: (cache_seqlens|scheduler_metadata)"):
        _forward(forward_entry, args)


def test_forward_metadata_entries(forward_entry, dtype):
    args = _inputs(dtype=dtype)
    args["seqused_k"] = torch.full((2,), 17, dtype=torch.int32, device="npu")
    args["scheduler_metadata"] = generate(args, args["seqused_k"]).clone()
    out, lse, *_ = _forward(forward_entry, args)
    assert out.shape == (2, 3, 4, 512) and lse.shape == (2, 4, 3)
    if forward_entry != "fake":
        torch.testing.assert_close(out.cpu(), torch.zeros(out.shape, dtype=dtype))
        torch.testing.assert_close(lse.cpu(), torch.full(lse.shape, 17.0).log())


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("length_kind", ["none", "integer", "tensor"])
@pytest.mark.parametrize("reuse", [False, True])
def test_public_return_contract(dtype, fake, length_kind, reuse):
    with FakeTensorMode() if fake else nullcontext():
        device = _inputs(dtype=dtype)
        lengths = {
            "none": None,
            "integer": 17,
            "tensor": torch.full((2,), 17, dtype=torch.int32, device="npu"),
        }[length_kind]
        record = generate(device) if reuse else None
        result = cached(device, record, lengths)
        assert len(result) == 4
        out, lse, out_accum, lse_accum = result
        assert out.shape == (2, 3, 4, 512) and out.dtype == dtype
        assert lse.shape == (2, 4, 3) and lse.dtype == torch.float32
        assert out_accum.shape == lse_accum.shape == (0,)
        assert out_accum.dtype == lse_accum.dtype == torch.float32
        assert cached(device, record, lengths, return_lse=False).shape == out.shape


@pytest.mark.parametrize("length", [-1, 0, 16, 18])
def test_public_integer_length_rejection(dtype, length):
    with pytest.raises(RuntimeError, match="cache_seqlens equals KV capacity"):
        cached(_inputs(dtype=dtype), lengths=length)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1),
        (2, 1, 8, 257),
        (1, 7, 16, 65),
        (2, 2, 128, 511),
        (21, 2, 128, 512),
        (2, 256, 1, 17),
    ],
)
@pytest.mark.parametrize("scale", [None, 0.0, -0.23, 0.23])
def test_public_source_accuracy(probes, shape, scale):
    case = make_case(probes, *shape, scale=scale, offset=1)
    refs = mla_reference_pair(**case["inputs"], softmax_scale=scale)
    launch(probes[0]["oracle"], case, "oracle")
    lengths = torch.full((shape[0] + 2,), shape[3], dtype=torch.int32, device="npu")[1:-1]
    record = generate(case["device"], lengths)
    storage = torch.zeros(WORDS + 2, dtype=torch.int32, device="npu")
    storage[1:-1].copy_(record)
    before = storage.clone()
    results = [cached(case["device"], md, lengths, scale) for md in (None, record, storage[1:-1])]
    torch.npu.synchronize()
    errors = []
    for result in results:
        for i, name in enumerate(("out", "lse")):
            actual = result[i].cpu()
            torch.testing.assert_close(actual, case["outputs"]["oracle"][i].cpu(), rtol=0, atol=0)
            assert_fa_close(actual, refs[i], refs[i + 2], name=name)
            errors.append(float((actual - refs[i]).abs().max()))
    torch.testing.assert_close(storage, before, rtol=0, atol=0)
    for expected, device, _ in case["storage"][:4]:
        torch.testing.assert_close(device.cpu(), expected, rtol=0, atol=0)
    with (probes[1] / "public-metadata-accuracy.jsonl").open("a") as f:
        f.write(
            json.dumps(
                dict(
                    shape=shape,
                    scale=scale,
                    errors=errors,
                    modes=["implicit", "explicit", "offset_clone"],
                )
            )
            + "\n"
        )


def test_public_rounding(probes):
    case, expected = make_rounding_case(probes)
    actual = cached(case["device"], generate(case["device"]))
    launch(probes[0]["oracle"], case, "oracle")
    for i in range(2):
        torch.testing.assert_close(actual[i], case["outputs"]["oracle"][i], rtol=0, atol=0)
    torch.testing.assert_close(actual[0].cpu(), expected, rtol=0, atol=0)


def test_reuse_scales_streams_and_allocator(probes):
    case = make_case(probes, 21, 2, 128, 511)
    cached(case["device"], generate(case["device"]))
    torch.npu.synchronize()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    outputs = []
    with forbid_host_readback():
        for iteration in range(16):
            device = {n: t.clone() for n, t in case["device"].items()}
            lengths = torch.full((21,), 511, dtype=torch.int32, device="npu")
            record = generate(device, lengths)
            # Both consumer streams depend on the public producer stream.
            for index, stream in enumerate(streams):
                stream.wait_stream(torch.npu.current_stream())
                scale = (None, 0.0, -0.23, 0.23)[(iteration + index) % 4]
                with torch.npu.stream(stream):
                    outputs.append((scale, cached(device, record, lengths, scale)))
            del record, lengths, device
            # Release all producer-owned inputs and overwrite matching sizes
            # while the consuming streams may still be reading them.
            pressure = [
                torch.full((WORDS,), -1, dtype=torch.int32, device="npu"),
                torch.zeros(21, dtype=torch.int32, device="npu"),
            ]
            pressure += [torch.full_like(t, 123) for t in case["device"].values()]
            del pressure
    torch.npu.synchronize()
    references = {
        scale: mla_reference_pair(**case["inputs"], softmax_scale=scale)
        for scale in (None, 0.0, -0.23, 0.23)
    }
    for scale, result in outputs:
        refs = references[scale]
        for i, name in enumerate(("out", "lse")):
            assert_fa_close(result[i].cpu(), refs[i], refs[i + 2], name=name)


def test_shared_metadata_host_threads(probes):
    case = make_case(probes, 2, 7, 16, 65)
    record = generate(case["device"])
    torch.npu.synchronize()

    def worker(scale):
        torch.npu.set_device(0)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            lengths = torch.full((2,), 65, dtype=torch.int32, device="npu")
            local = generate(case["device"], lengths)
            results = [
                cached(case["device"], md, lengths, scale)
                for md in (record, local)
                for _ in range(4)
            ]
        stream.synchronize()
        return scale, results

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, (0.0, -0.23)))
    for scale, items in results:
        refs = mla_reference_pair(**case["inputs"], softmax_scale=scale)
        for result in items:
            for i, name in enumerate(("out", "lse")):
                assert_fa_close(result[i].cpu(), refs[i], refs[i + 2], name=name)


def test_producer_input_lifetime_and_snapshot(dtype):
    args = metadata_args(dtype)
    reference = api.get_scheduler_metadata(**args).cpu()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    records = []
    with forbid_host_readback():
        for iteration in range(32):
            lengths = torch.full((2,), 17, dtype=torch.int32, device="npu")
            stream = streams[iteration % 2]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                records.append(api.get_scheduler_metadata(**dict(args, cache_seqlens=lengths)))
            del lengths
            pressure = torch.zeros(2, dtype=torch.int32, device="npu")
            del pressure
    torch.npu.synchronize()
    for record in records:
        torch.testing.assert_close(record.cpu(), reference, rtol=0, atol=0)
    # A snapshot owns no live length pointer. Updating the original input
    # leaves the record unchanged; a new generation reads the changed values.
    args["cache_seqlens"].zero_()
    failed = api.get_scheduler_metadata(**args).cpu()
    assert failed[5].item() == 4 and failed[8:].count_nonzero().item() == 0
    torch.testing.assert_close(records[-1].cpu(), reference, rtol=0, atol=0)


def test_compile_metadata_reuse(dtype):
    device = _inputs(dtype=dtype)
    lengths = torch.full((2,), 17, dtype=torch.int32, device="npu")

    def run(q, k, v, qv, lengths, scale):
        inputs = dict(q=q, k=k, v=v, qv=qv)
        return cached(inputs, generate(inputs, lengths), lengths, scale)

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    for scale in (0.0, -0.23):
        result = compiled(*(device[n] for n in ("q", "k", "v", "qv")), lengths, scale)
        expected = cached(device, lengths=lengths, scale=scale)
        for actual, ref in zip(result, expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "fault",
    [
        f"{mode}-{value}"
        for mode in ("generated", "implicit", "reused")
        for value in ("negative", "zero", "short", "over")
    ]
    + ["magic", "dtype", "shape", "workspace"],
)
def test_public_runtime_error(dtype, fault, tmp_path):
    dtype_name = "fp16" if dtype == torch.float16 else "bf16"
    base = Path(os.environ.get("MLA_PUBLIC_ERROR_ARTIFACT_DIR", tmp_path))
    directory = base / dtype_name / fault
    directory.mkdir(parents=True)
    torch.npu.synchronize()
    command = [
        sys.executable,
        "-m",
        "tests.mla_public_error_probe",
        dtype_name,
        fault,
        str(directory),
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        with subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        ) as child:
            (directory / "pid.json").write_text(json.dumps(dict(pid=child.pid)) + "\n")
            try:
                code = child.wait(timeout=90)
            finally:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    assert code == 0, (directory / "process.log").read_text()
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] in ("launch", "event")
    assert "trap error" in report["error"]
    assert any(c in report["error"] for c in ("507015", "507016", "507035", "507036"))
    assert report["elapsed_seconds"] < 15
