"""Production BSND strides: public/raw dispatch, source/math and view lifetime."""

from contextlib import contextmanager, nullcontext
import ctypes as ct
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.mla_dense_build import ROOT
from tests.mla.support.mla_profile_cases import STRIDE_CASES
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_attention import values_for
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_metadata import check_source
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    evaluate,
    length_artifacts as length_artifacts,
    length_oracle as length_oracle,
)
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_stride import (
    BOUNDARY_CASES,
    LAYOUT_CASES,
    NAMES,
    PERFORMANCE_CASES as COMPONENT_CASES,
    WIDE_CASES,
    launch,
    make_case as make_probe_case,
    spec,
    stride_attention as stride_attention,
    stride_producer as stride_producer,
    strides_for,
)
from tests.mla.e2e.bsnd.test_flash_attn_npu_v3_mla_bsnd_used import check_result
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import _forward


# Fix the production component matrix before collection. Production uses every
# platform core for length-dependent launches; the source uses the same schedule.
PERFORMANCE_CASES = [dict(c, aic=None) for c in COMPONENT_CASES]
ACCURACY_CASES = (
    LAYOUT_CASES
    + BOUNDARY_CASES
    + PERFORMANCE_CASES
    + [spec(5, f"combined_{n}", (2, 5, 5, 17), [5, 3], [17, 7]) for n in (65536, 65537)]
    + [spec(5, "independent", uq=[3, 1])]
)


def make_case(probes, producer, config, **kwargs):
    return make_probe_case(probes, producer, config, full_launch=True, **kwargs)


def forward_args(case):
    args = dict(case["devices"]["migrated"], softmax_scale=case["scale"])
    if case["config"]["version"] == 5:
        args["seqused_q"] = case["inputs"][2]
    if case["config"]["version"] != 1:
        args["seqused_k"] = case["inputs"][3]
    return args


def generate(args):
    q, k = args["q"], args["k"]
    return api.get_scheduler_metadata(
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        1,
        64,
        args["seqused_k"],
        qkv_dtype=q.dtype,
        headdim_v=512,
        num_splits=1,
    )


def cached(args, metadata=None):
    return api.flash_attn_with_kvcache(
        args["q"],
        args["k"],
        args["v"],
        qv=args["qv"],
        cache_seqlens=args.get("seqused_k"),
        scheduler_metadata=metadata,
        softmax_scale=args.get("softmax_scale"),
        num_splits=1,
        return_softmax_lse=True,
    )


def forward(case, entry="public", metadata=None):
    args = forward_args(case)
    if entry == "cpp" or case["config"]["version"] == 5:
        return _forward("cpp" if entry == "cpp" else "eager", args)
    if case["config"]["version"] == 2:
        return cached(args, metadata)
    return api.flash_attn_func(**args, num_splits=1, return_attn_probs=True)


def check_production(probes, case, entry="public"):
    with forbid_host_readback():
        if sum(case["item"]["q"]) and sum(case["item"]["kv"]):
            assert launch(probes, case, "oracle") == 0
        record = generate(forward_args(case)) if case["config"]["version"] == 2 else None
        # A production forward must consume the views directly. Both Python and
        # the NPU operator trace are checked; the C++ source has no copy path.
        with patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("input copy")):
            results = [forward(case, entry), forward(case, entry, record)]
    for result in results:
        errors = check_result(case, result, source=True)
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    if record is not None:
        assert record.cpu().tolist() == generate(forward_args(case)).cpu().tolist()
    return errors


@pytest.mark.parametrize("config", ACCURACY_CASES)
def test_public_source_and_math(stride_attention, stride_producer, config, tmp_path):
    case = make_case(stride_attention, stride_producer, config)
    errors = check_production(stride_attention, case)
    directory = Path(os.environ.get("MLA_PRODUCTION_STRIDE_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "accuracy.jsonl").open("a") as log:
        log.write(
            json.dumps(dict(config=config, dtype=str(stride_attention[0]["dtype"]), errors=errors))
            + "\n"
        )


@pytest.mark.parametrize("config", LAYOUT_CASES)
def test_raw_cpp_source_and_math(stride_attention, stride_producer, config):
    check_production(stride_attention, make_case(stride_attention, stride_producer, config), "cpp")


@pytest.mark.parametrize("config", WIDE_CASES)
def test_production_wide_address(stride_attention, stride_producer, config):
    torch.npu.empty_cache()
    check_production(stride_attention, make_case(stride_attention, stride_producer, config))


def small_inputs(dtype, profile="independent", batch=2, sq=7, sk=33):
    args = {}
    for index, name in enumerate(NAMES):
        shape = (
            batch,
            sq if index < 2 else sk,
            8 if index < 2 else 1,
            64 if index % 2 == 0 else 512,
        )
        strides = strides_for(shape, profile, name)
        span = (
            0
            if batch == 0 or shape[1] == 0
            else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
        )
        base = torch.zeros(span + 32, dtype=dtype, device="npu")
        args[name] = base.as_strided(shape, strides, index + 3)
    return args


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("fake", (False, True))
@pytest.mark.parametrize("shape", ((2, 7, 33), (0, 7, 33), (2, 0, 33), (2, 7, 0)))
@pytest.mark.parametrize("path", ("func", "cache", "used", "cpp", "dispatch"))
def test_real_fake_empty_and_shapes(dtype, fake, shape, path):
    batch, sq, sk = shape
    with FakeTensorMode() if fake else nullcontext():
        args = small_inputs(dtype, batch=batch, sq=sq, sk=sk)
        if path == "func":
            out, lse = api.flash_attn_func(**args, return_attn_probs=True)
        elif path == "dispatch":
            out, lse = torch.ops.flash_attn_npu_3._mla_dense_forward(**args)
        else:
            args["seqused_k"] = torch.full((batch,), sk, dtype=torch.int32, device="npu")
            if path == "used":
                args["seqused_q"] = torch.full((batch,), sq, dtype=torch.int32, device="npu")
            out, lse, *_ = (
                cached(args, generate(args))
                if path == "cache"
                else _forward("eager" if fake or path == "used" else "cpp", args)
            )
        assert out.shape == (batch, sq, 8, 512) and out.dtype == dtype
        assert lse.shape == (batch, 8, sq) and lse.dtype == torch.float32
        if not fake:
            assert (out.cpu() == 0).all()
            expected = float("inf") if sk == 0 else torch.tensor(float(sk)).log().item()
            torch.testing.assert_close(lse.cpu(), torch.full(lse.shape, expected))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("profile", ("bnsd", "broadcast", "overlap", "matrix_65536", "head_65536"))
@pytest.mark.parametrize("path", ("func", "cache", "used"))
def test_compile_updates_and_reuse(dtype, profile, path):
    torch._dynamo.reset()
    args = small_inputs(dtype, profile)
    args["seqused_k"] = torch.tensor([17, 0], dtype=torch.int32, device="npu")
    record = generate(args)

    def call(q, k, v, qv, uq, uk, metadata, scale):
        if path == "func":
            return api.flash_attn_func(q, k, v, qv=qv, softmax_scale=scale, return_attn_probs=True)
        if path == "cache":
            return api.flash_attn_with_kvcache(
                q,
                k,
                v,
                qv=qv,
                cache_seqlens=uk,
                scheduler_metadata=metadata,
                softmax_scale=scale,
                num_splits=1,
                return_softmax_lse=True,
            )
        return api._flash_attn_forward(
            q, k, v, qv=qv, seqused_q=uq, seqused_k=uk, softmax_scale=scale
        )

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    for profile_now, uq_values, uk_values, scale in (
        (profile, [3, 1], [17, 0], -0.125),
        ("independent", [1, 0], [17, 0], 0.0),
        (profile, [7, 7], [33, 1], 0.23),
    ):
        args.update(small_inputs(dtype, profile_now))
        args["seqused_k"].copy_(torch.tensor(uk_values, dtype=torch.int32))
        if uk_values != [17, 0]:
            record = generate(args)
        uq = torch.tensor(uq_values, dtype=torch.int32, device="npu")
        values = [args[n] for n in ("q", "k", "v", "qv")] + [uq, args["seqused_k"], record, scale]
        out, lse, *_ = compiled(*values)
        for b in range(2):
            used = uq_values[b] if path == "used" else 7
            length = uk_values[b] if path != "func" else 33
            assert (out[b, :used].cpu() == 0).all()
            torch.testing.assert_close(
                lse[b, :, :used].cpu(), torch.full((8, used), float(length)).log()
            )


@pytest.mark.parametrize("scale", (None, 0.0, -0.125, 0.23))
def test_record_reused_across_strides_and_scales(stride_attention, stride_producer, scale):
    cases = [
        make_case(stride_attention, stride_producer, spec(2, p, uk=[17, 0]), scale=scale)
        for p in ("bnsd", "independent", "matrix_65536")
    ]
    record = generate(forward_args(cases[0]))
    before = record.cpu().clone()
    for case in cases:
        assert launch(stride_attention, case, "oracle") == 0
        with forbid_host_readback():
            result = forward(case, metadata=record)
        check_result(case, result, source=True)
    assert torch.equal(record.cpu(), before)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("used", (False, True))
def test_stream_and_storage_lifetime(dtype, used):
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    for step in range(4):
        with torch.npu.stream(creator):
            args = small_inputs(dtype)
            args["v"].fill_(step + 1)
            args["seqused_k"] = torch.tensor([17, 0], dtype=torch.int32, device="npu")
            if used:
                args["seqused_q"] = torch.tensor([3, 1], dtype=torch.int32, device="npu")
            record = None if used else generate(args)
        consumer.wait_stream(creator)
        with torch.npu.stream(consumer), forbid_host_readback():
            results.append(api._flash_attn_forward(**args) if used else cached(args, record))
        del args, record
        gc.collect()
        with torch.npu.stream(creator):
            churn = [torch.full((8192,), -77, dtype=dtype, device="npu") for _ in range(16)]
        del churn
    consumer.synchronize()
    for step, (out, lse, *_) in enumerate(results):
        q0, q1 = (3, 1) if used else (7, 7)
        torch.testing.assert_close(
            out[0, :q0].cpu(), torch.full((q0, 8, 512), step + 1, dtype=dtype)
        )
        assert (out[1, :q1].cpu() == 0).all() and lse[1, :, :q1].cpu().isneginf().all()
        torch.testing.assert_close(lse[0, :, :q0].cpu(), torch.full((8, q0), 17.0).log())
    creator.synchronize()


@pytest.fixture(scope="module")
def storage_fault(tmp_path_factory):
    directory = tmp_path_factory.mktemp("storage-fault")
    library = directory / "storage.so"
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-I" + str(torch_root / "include"),
            str(ROOT / "tests/mla/support/mla_tensor_storage_probe.cpp"),
            "-L" + str(torch_root / "lib"),
            "-Wl,-rpath," + str(torch_root / "lib"),
            "-lc10",
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ct.CDLL(str(library))
    lib.mla_test_storage_bytes.argtypes = [ct.c_uint64, ct.c_uint64]
    lib.mla_test_storage_bytes.restype = ct.c_uint64

    @contextmanager
    def inject(tensor):
        # Change only host byte-size metadata; preserve allocation and streams.
        # NPU Storage.resize_ queues copies and is unsuitable for this fixture.
        # Compact logical bytes still fit: rejection must use the actual outer
        # strides and the full allocation, even when all used lengths are zero.
        truncated = (tensor.storage_offset() + tensor.numel()) * tensor.element_size()
        assert truncated < tensor.untyped_storage().size()
        previous = lib.mla_test_storage_bytes(tensor._cdata, truncated)
        try:
            assert tensor.untyped_storage().size() == truncated
            yield
        finally:
            assert lib.mla_test_storage_bytes(tensor._cdata, previous) == truncated

    return inject


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("path", ("public", "cpp", "used", "fake"))
def test_complete_storage_checked_before_launch(name, path, storage_fault):
    with FakeTensorMode() if path == "fake" else nullcontext():
        args = small_inputs(torch.float16)
        if path == "used":
            args["seqused_q"] = torch.zeros(2, dtype=torch.int32, device="npu")
            args["seqused_k"] = torch.zeros(2, dtype=torch.int32, device="npu")
        with (
            storage_fault(args[name]),
            forbid_host_readback(),
            pytest.raises(RuntimeError, match="BSND input view exceeds"),
        ):
            if path in ("public", "fake"):
                cached(args)
            else:
                _forward("cpp" if path == "cpp" else "eager", args)


@pytest.mark.parametrize("path", ("func", "cache", "used"))
def test_compiled_storage_rechecked(path, storage_fault):
    torch._dynamo.reset()
    args = small_inputs(torch.float16)

    def call(q, k, v, qv):
        if path == "func":
            return api.flash_attn_func(q, k, v, qv=qv)
        if path == "cache":
            return api.flash_attn_with_kvcache(q, k, v, qv=qv, num_splits=1)
        return api._flash_attn_forward(
            q, k, v, qv=qv, seqused_q=torch.zeros(2, dtype=torch.int32, device=q.device)
        )

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    compiled(**args)
    torch.npu.synchronize()
    with storage_fault(args["v"]), pytest.raises(RuntimeError, match="BSND input view exceeds"):
        compiled(**args)


@pytest.mark.parametrize("kind", ("fp16", "bf16"))
@pytest.mark.parametrize(
    "fault",
    (
        "reused-short",
        "snapshot",
        "emptyq:magic",
        "q_over",
        "zero_q_bad_k",
        "zero_k_bad_q",
        "empty_k",
    ),
)
def test_strided_device_rejections(kind, fault, tmp_path):
    script = (
        "mla_public_error_probe"
        if fault in ("reused-short", "snapshot", "emptyq:magic")
        else "mla_bsnd_used_error_probe"
    )
    directory = (
        Path(os.environ.get("MLA_PRODUCTION_STRIDE_DIR", tmp_path)) / kind / ("trap-" + fault)
    )
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "tests." + script, kind, fault, str(directory), "strided"]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0, (directory / "process.log").read_text()[-4000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["input_layout"] == "strided" and "trap error" in report["error"]
    assert report["elapsed_seconds"] < 15


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_public_stride_source_schedules(stride_producer, length_oracle, dtype, tmp_path):
    values, records = [], []
    for shape, _, lengths, _ in STRIDE_CASES:
        b, sq, heads, sk = shape
        item = values_for(stride_producer, shape, lengths)
        p = stride_producer["platform"]
        item[12] = p.block_dims[p.aic - 1]
        used = torch.tensor(
            [sk] * b if lengths is None else lengths, dtype=torch.int32, device="npu"
        )
        record = api.get_scheduler_metadata(
            b, sq, sk, heads, 1, 64, used, qkv_dtype=dtype, headdim_v=512, num_splits=1
        )
        values.append(item)
        records.append([x & 0xFFFFFFFF for x in record.cpu().tolist()])
    directory = Path(os.environ.get("MLA_PRODUCTION_STRIDE_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    name = "public-schedules-" + str(dtype).removeprefix("torch.")
    check_source(records, values, length_oracle, directory, name)
    native = evaluate(length_oracle, values, automatic_fd=True)
    report = [
        dict(case=i, config=config, native_fd=row[5] != 0, native_schedule=row)
        for i, (config, row) in enumerate(zip(STRIDE_CASES, native))
    ]
    (directory / (name + "-dispatch.json")).write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize("path", ("func", "cache", "used"))
def test_compiled_nonfinite_scale_rejected(path):
    torch._dynamo.reset()
    args = small_inputs(torch.float16)
    used = torch.ones(2, dtype=torch.int32, device="npu")

    def call(q, k, v, qv, scale):
        if path == "func":
            return api.flash_attn_func(q, k, v, qv=qv, softmax_scale=scale)
        if path == "cache":
            return api.flash_attn_with_kvcache(q, k, v, qv=qv, softmax_scale=scale, num_splits=1)
        return api._flash_attn_forward(q, k, v, qv=qv, seqused_q=used, softmax_scale=scale)

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    compiled(**args, scale=0.125)
    for scale in (float("nan"), float("inf"), 1e100):
        with pytest.raises(RuntimeError, match="finite in FP32"):
            compiled(**args, scale=scale)
