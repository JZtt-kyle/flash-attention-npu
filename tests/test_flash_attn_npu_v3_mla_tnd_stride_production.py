"""Production TND Q/KV strides: public v3/v5, storage, updates and lifetime."""

from contextlib import nullcontext
import gc
import itertools
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
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_bsnd_stride import NAMES, strides_for
from tests.test_flash_attn_npu_v3_mla_bsnd_stride_production import (
    storage_fault as storage_fault,
)
from tests.test_flash_attn_npu_v3_mla_contract import _forward
from tests.test_flash_attn_npu_v3_mla_tnd_stride import (
    BOUNDARY_CASES,
    EMPTY_CASES,
    LAYOUT_CASES,
    PERFORMANCE_CASES as COMPONENT_CASES,
    WIDE_CASES,
    launch,
    make_case,
    spec,
    stride_attention as stride_attention,
    stride_producer as stride_producer,
)
from tests.test_flash_attn_npu_v3_mla_used_attention import effective
from tests.test_flash_attn_npu_v3_mla_varlen_used import check_result, forward_args


# Fix both matrices before sampling. Public native FIA accepts Hq<=128;
# larger heads and original geometry remain in the component/accuracy matrix.
PERFORMANCE_CASES = [dict(c, aic=None) for c in COMPONENT_CASES if c["item"]["layout"] == 1]
PUBLIC_CASES = [
    dict(c, item=dict(c["item"], heads=min(c["item"]["heads"], 128))) for c in PERFORMANCE_CASES
] + [
    spec(3, profile="broadcast", heads=8),
    spec(5, profile="ntd", uq=(2, 0, 1)),
    spec(5, uk=(7, 0, 17)),
    spec(5, uq=(3, 0, 1), uk=(17, 0, 33)),
    spec(5, uq=(2, 0, 1), uk=(0, 0, 0), heads=128),
]
ACCURACY_CASES = (
    [c for c in LAYOUT_CASES + BOUNDARY_CASES + EMPTY_CASES if c["item"]["layout"] == 1]
    + PERFORMANCE_CASES
    + PUBLIC_CASES
)


def production_case(probes, producer, config, **kwargs):
    assert config["item"]["layout"] == 1
    return make_case(probes, producer, config, full_launch=True, **kwargs)


def invoke(args, entry="public"):
    if entry == "public":
        return api.flash_attn_varlen_func(**args, num_splits=1, return_attn_probs=True)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_tnd_forward(**args)
    return _forward(entry, args)


def check_production(probes, case, entry="public"):
    with forbid_host_readback():
        if sum(effective(case["item"])[0]) and sum(case["item"]["kv"]):
            assert launch(probes, case, "oracle") == 0
        with patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("input copy")):
            results = [invoke(forward_args(case), entry) for _ in range(2)]
    for result in results:
        errors = check_result(case, result, source=True)
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    return errors


@pytest.mark.parametrize("config", ACCURACY_CASES)
def test_public_source_and_math(stride_attention, stride_producer, config, tmp_path):
    errors = check_production(
        stride_attention, production_case(stride_attention, stride_producer, config)
    )
    directory = Path(os.environ.get("MLA_TND_PRODUCTION_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "accuracy.jsonl").open("a") as log:
        log.write(
            json.dumps(dict(config=config, dtype=str(stride_attention[0]["dtype"]), errors=errors))
            + "\n"
        )


@pytest.mark.parametrize("config", [c for c in LAYOUT_CASES if c["item"]["layout"] == 1])
def test_raw_cpp_source_and_math(stride_attention, stride_producer, config):
    check_production(
        stride_attention, production_case(stride_attention, stride_producer, config), "cpp"
    )


@pytest.mark.parametrize("config", [c for c in WIDE_CASES if c["item"]["layout"] == 1])
def test_wide_active_address(stride_attention, stride_producer, config):
    torch.npu.empty_cache()
    check_production(stride_attention, production_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("version", (3, 5))
@pytest.mark.parametrize("scale", (0.0, -0.125, 0.23))
def test_scale_source_and_math(stride_attention, stride_producer, version, scale):
    config = spec(version, profile="ntd", uq=(2, 0, 1) if version == 5 else None)
    check_production(
        stride_attention, production_case(stride_attention, stride_producer, config, scale=scale)
    )


def inputs(dtype, profile="independent", qs=(7, 3), ks=(33, 17), used=False):
    args = {}
    for i, name in enumerate(NAMES):
        shape = (1, sum(qs if i < 2 else ks), 8 if i < 2 else 1, 64 if i % 2 == 0 else 512)
        strides = strides_for(shape, "bnsd" if profile == "ntd" else profile, name)
        span = 0 if shape[1] == 0 else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
        base = torch.zeros(span + 32, dtype=dtype, device="npu")
        args[name] = base.as_strided(shape, strides, i + 3)[0]
    args.update(
        cu_seqlens_q=torch.tensor(
            list(itertools.accumulate(qs, initial=0)), dtype=torch.int32, device="npu"
        ),
        cu_seqlens_k=torch.tensor(
            list(itertools.accumulate(ks, initial=0)), dtype=torch.int32, device="npu"
        ),
        max_seqlen_q=max(qs, default=0),
        max_seqlen_k=max(ks, default=0),
    )
    if used:
        args.update(
            seqused_q=torch.tensor(qs, dtype=torch.int32, device="npu"),
            seqused_k=torch.tensor(ks, dtype=torch.int32, device="npu"),
        )
    return args


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("fake", (False, True))
@pytest.mark.parametrize(
    "qs,ks", (((7, 3), (33, 17)), ((), ()), ((0, 0), (33, 17)), ((7, 3), (0, 0)))
)
@pytest.mark.parametrize("entry", ("public", "op", "eager", "cpp"))
@pytest.mark.parametrize("used", (False, True))
def test_shapes_and_empty(dtype, fake, qs, ks, entry, used):
    with FakeTensorMode() if fake else nullcontext():
        args = inputs(dtype, qs=qs, ks=ks, used=used)
        out, lse, *_ = invoke(args, "fake" if fake and entry == "cpp" else entry)
        assert out.shape == (sum(qs), 8, 512) and out.dtype == dtype
        assert lse.shape == (8, sum(qs)) and lse.dtype == torch.float32
        if not fake:
            assert (out.cpu() == 0).all()
            expected = [float(k) for q, k in zip(qs, ks) for _ in range(q)]
            expected = torch.tensor(expected).log().expand(8, -1)
            if not sum(ks):
                expected.fill_(float("inf"))
            torch.testing.assert_close(lse.cpu(), expected)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("profile", ("ntd", "broadcast", "overlap", "matrix_65536", "head_65536"))
@pytest.mark.parametrize("used", (False, True))
def test_compile_prefix_used_and_view_updates(dtype, profile, used):
    torch._dynamo.reset()

    def call(q, k, v, qv, cq, ck, uq, uk, scale):
        return api.flash_attn_varlen_func(
            q,
            k,
            v,
            cq,
            ck,
            7,
            33,
            qv=qv,
            seqused_q=uq,
            seqused_k=uk,
            softmax_scale=scale,
            return_attn_probs=True,
        )

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    args = inputs(dtype, profile, used=used)
    for current, qs, ks, uq, uk, scale in (
        (profile, (7, 3), (33, 17), (3, 1), (17, 0), -0.125),
        ("independent", (3, 7), (17, 33), (1, 0), (0, 17), 0.0),
        (profile, (7, 3), (33, 17), (7, 3), (33, 17), 0.23),
    ):
        views = inputs(dtype, current)
        args.update({n: views[n] for n in NAMES})
        for name, values in (
            ("cu_seqlens_q", (0, qs[0], sum(qs))),
            ("cu_seqlens_k", (0, ks[0], sum(ks))),
        ):
            args[name].copy_(torch.tensor(values, dtype=torch.int32))
        if used:
            args["seqused_q"].copy_(torch.tensor(uq, dtype=torch.int32))
            args["seqused_k"].copy_(torch.tensor(uk, dtype=torch.int32))
        values = [args[n] for n in ("q", "k", "v", "qv", "cu_seqlens_q", "cu_seqlens_k")]
        # Dynamo inspects Tensor methods while tracing; replacing them with
        # mocks prevents tracing itself. Eager and lifecycle tests forbid host
        # readback; this path checks fullgraph capture and device length updates.
        out, lse = compiled(*values, args.get("seqused_q"), args.get("seqused_k"), scale)
        start = 0
        for b in range(2):
            count, length = (uq[b], uk[b]) if used else (qs[b], ks[b])
            assert (out[start : start + count].cpu() == 0).all()
            torch.testing.assert_close(
                lse[:, start : start + count].cpu(), torch.full((8, count), float(length)).log()
            )
            start += qs[b]


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("used", (False, True))
def test_stream_and_storage_lifetime(dtype, used):
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    for step in range(4):
        with torch.npu.stream(creator):
            args = inputs(dtype, "ntd", ks=(33, 0), used=used)
            args["v"].fill_(step + 1)
            if used:
                args["seqused_q"].copy_(torch.tensor([3, 1], dtype=torch.int32))
        consumer.wait_stream(creator)
        with torch.npu.stream(consumer), forbid_host_readback():
            results.append(invoke(args))
        del args
        gc.collect()
        with torch.npu.stream(creator):
            churn = [torch.full((8192,), -77, dtype=dtype, device="npu") for _ in range(16)]
        del churn
    consumer.synchronize()
    for step, (out, lse) in enumerate(results):
        first, second = (3, 1) if used else (7, 3)
        torch.testing.assert_close(
            out[:first].cpu(), torch.full((first, 8, 512), step + 1, dtype=dtype)
        )
        assert (out[7 : 7 + second].cpu() == 0).all() and lse[
            :, 7 : 7 + second
        ].cpu().isneginf().all()
        torch.testing.assert_close(lse[:, :first].cpu(), torch.full((8, first), 33.0).log())
    creator.synchronize()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("entry", ("public", "cpp", "op", "fake"))
@pytest.mark.parametrize("used", (False, True))
def test_full_storage_rejected(name, entry, used, storage_fault):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        # A pure NTD transpose can occupy exactly numel elements. Give every
        # input real holes so a compact-byte truncation removes reachable data.
        args = inputs(torch.float16, "row_65536", used=used)
        if used:
            args["seqused_q"].zero_()
            args["seqused_k"].zero_()
        with (
            storage_fault(args[name]),
            forbid_host_readback(),
            pytest.raises(RuntimeError, match="TND input view exceeds"),
        ):
            invoke(args, entry)


@pytest.mark.parametrize("used", (False, True))
@pytest.mark.parametrize("name", NAMES)
def test_compiled_storage_rechecked(used, name, storage_fault):
    torch._dynamo.reset()
    args = inputs(torch.float16, "row_65536", used=used)
    compiled = torch.compile(lambda **kwargs: invoke(kwargs), backend="eager", fullgraph=True)
    compiled(**args)
    torch.npu.synchronize()
    with storage_fault(args[name]), pytest.raises(RuntimeError, match="TND input view exceeds"):
        compiled(**args)


@pytest.mark.parametrize("kind", ("fp16", "bf16"))
@pytest.mark.parametrize(
    "version,fault",
    (
        (3, "q_start"),
        (3, "k_tail"),
        (3, "empty_q"),
        (5, "q_over"),
        (5, "zero_q_bad_k"),
        (5, "zero_k_bad_q"),
        (5, "empty_k"),
    ),
)
def test_strided_device_rejections(kind, version, fault, tmp_path):
    script = "mla_varlen_public_error_probe" if version == 3 else "mla_varlen_used_error_probe"
    directory = (
        Path(os.environ.get("MLA_TND_PRODUCTION_DIR", tmp_path)) / kind / f"trap-v{version}-{fault}"
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
