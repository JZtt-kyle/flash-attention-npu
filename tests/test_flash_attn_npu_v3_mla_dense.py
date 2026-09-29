"""Independent 910B FP16/BF16 dense Kernel acceptance; no public FA3 dispatch."""

import ctypes
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.compare import assert_fa_close
from tests.common.mla_ref import mla_reference_pair
from tests.mla_cann_baseline import Case, make_inputs
from tests.mla_dense_build import ROOT, build_probes


class Schedule(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32)
        for name in (
            "batch",
            "sq",
            "heads",
            "sk",
            "m_base",
            "s2_base",
            "cores",
            "block_dim",
            "ratio",
            "r1",
            "r2",
        )
    ] + [(name, ctypes.c_uint32 * 26) for name in ("b_end", "m_end", "s_end")]


def load_probes(libraries, artifacts, dtype=torch.float16):
    torch.npu.set_device(0)
    assert "910B" in torch.npu.get_device_name(0)
    loaded = {name: ctypes.CDLL(path) for name, path in libraries.items()}
    for name in ("oracle", "migrated"):
        loaded[name].launch_dense.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_float]
        loaded[name].launch_dense.restype = ctypes.c_int
    host = loaded["schedule"]
    host.dense_schedule_size.restype = ctypes.c_uint32
    assert host.dense_schedule_size() == ctypes.sizeof(Schedule)
    host.prepare_dense.argtypes = [ctypes.c_int64] * 4 + [ctypes.c_void_p] * 2
    host.prepare_dense.restype = ctypes.c_int
    loaded["dtype"] = dtype
    return loaded, Path(artifacts)


@pytest.fixture(scope="module", params=["fp16", "bf16"])
def probes(tmp_path_factory, request):
    source = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    if not source.exists():
        pytest.skip("set MLA_CANN_SOURCE_ROOT to the pinned CANN source checkout")
    artifacts = (
        Path(os.environ.get("MLA_DENSE_ARTIFACT_DIR", tmp_path_factory.mktemp("mla-dense")))
        / request.param
    )
    dtype = torch.float16 if request.param == "fp16" else torch.bfloat16
    return load_probes(build_probes(artifacts, request.param), artifacts, dtype)


def make_case(probes, batch, sq, heads, sk, *, scale=None, offset=16, perturb=None, amplitude=1.0):
    case = Case("dense-probe", "compressed", batch, sq, sk, heads, 1, scale=scale)
    dtype = probes[0]["dtype"]
    inputs = make_inputs(case, dtype, seed=20260929)
    if perturb is not None:
        inputs["q"].zero_()
        inputs["qv"].zero_()
        name, lo, hi = [("qv", 0, 256), ("qv", 256, 512), ("q", 0, 32), ("q", 32, 64)][perturb]
        gen = torch.Generator().manual_seed(20260929 + perturb)
        inputs[name][..., lo:hi] = torch.randn(batch, sq, heads, hi - lo, generator=gen).to(dtype)
    inputs["q"].mul_(amplitude)
    inputs["qv"].mul_(amplitude)
    schedule, details = Schedule(), (ctypes.c_uint64 * 16)()
    assert (
        probes[0]["schedule"].prepare_dense(batch, sq, heads, sk, ctypes.byref(schedule), details)
        == 0
    )
    assert details[14] == 1 and details[15] == 0  # source BATCH_MODE, no FD
    assert schedule.ratio == 2
    storage = []

    def guarded(t, prefix):
        before = torch.full((t.numel() + prefix + 16,), 123.0, dtype=t.dtype)
        before[prefix : prefix + t.numel()] = t.flatten()
        device = before.npu()
        storage.append((before, device, prefix))
        return device[prefix : prefix + t.numel()].view(t.shape)

    device = {name: guarded(t, offset) for name, t in inputs.items()}
    desc = torch.tensor([24, 1 << 32, 0xFFFFFFFF, device["v"].data_ptr()], dtype=torch.int64).npu()
    outputs = {
        name: [
            guarded(torch.full((batch, sq, heads, 512), 12345.0, dtype=dtype), 16),
            guarded(torch.full((batch, heads, sq), 12345.0), 16),
        ]
        for name in ("oracle", "migrated")
    }
    ws_storage = {
        name: torch.full((details[0] + 1024,), 205, dtype=torch.uint8, device="npu")
        for name in outputs
    }
    workspace = {name: value[512:-512] for name, value in ws_storage.items()}
    return dict(
        dtype=str(dtype),
        shape=(batch, sq, heads, sk),
        scale=576**-0.5 if scale is None else scale,
        inputs=inputs,
        device=device,
        desc=desc,
        outputs=outputs,
        storage=storage,
        workspace=workspace,
        ws_storage=ws_storage,
        schedule=schedule,
        details=list(details),
        offset=offset,
        perturb=perturb,
        amplitude=amplitude,
    )


def launch(lib, case, name, *, scale=None):
    d = case["device"]
    pointers = [
        d["qv"],
        d["q"],
        d["v"],
        d["k"],
        case["desc"],
        *case["outputs"][name],
        case["workspace"][name],
    ]
    result = lib.launch_dense(
        torch.npu.current_stream().npu_stream,
        *(x.data_ptr() for x in pointers),
        ctypes.byref(case["schedule"]),
        case["scale"] if scale is None else scale,
    )
    assert result == 0, f"dense probe launch failed: {result}"


def check_case(probes, case, *, repeats=3, launch_fn=launch):
    libs, artifacts = probes
    first = {}
    for repeat in range(repeats):
        for name in ("oracle", "migrated"):
            launch_fn(libs[name], case, name)
        torch.npu.synchronize()
        results = {
            name: [x.cpu() for x in case["outputs"][name]] for name in ("oracle", "migrated")
        }
        for i in range(2):
            torch.testing.assert_close(results["migrated"][i], results["oracle"][i], rtol=0, atol=0)
            if repeat:
                torch.testing.assert_close(results["migrated"][i], first[i], rtol=0, atol=0)
            else:
                first[i] = results["migrated"][i]
    refs = mla_reference_pair(**case["inputs"], softmax_scale=case["scale"])
    errors = []
    digest = hashlib.sha256()
    for i, name in enumerate(("out", "lse")):
        assert_fa_close(first[i], refs[i], refs[i + 2], name=name)
        errors.append(float((first[i] - refs[i]).abs().max()))
        digest.update(first[i].contiguous().view(torch.uint8).numpy().tobytes())
    for i, (before, device, prefix) in enumerate(case["storage"]):
        after = device.cpu()
        if i < 4:
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        else:
            assert torch.equal(after[:prefix], before[:prefix])
            assert torch.equal(after[-16:], before[-16:])
    for ws in case["ws_storage"].values():
        assert (ws[:512].cpu() == 205).all() and (ws[-512:].cpu() == 205).all()
        # The source leaves vec2 and all host-only workspace reserve unused here.
        assert bool((ws[512 + case["details"][13] :] == 205).all().cpu())
    s = case["schedule"]
    record = {
        key: case[key]
        for key in ("dtype", "shape", "scale", "offset", "perturb", "amplitude", "details")
    }
    record.update(
        repeats=repeats,
        errors=errors,
        result_sha256=digest.hexdigest(),
        m_base=s.m_base,
        used_cores=s.cores,
        block_dim=s.block_dim,
        endpoints=list(zip(s.b_end[: s.cores], s.m_end[: s.cores], s.s_end[: s.cores])),
    )
    with (artifacts / "accuracy.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


@pytest.mark.parametrize("shape", [(2, 1, 8, 257), (1, 7, 16, 65)])
@pytest.mark.parametrize("scale", [None, 0.0, -0.23, 0.23])
def test_d01_d02_scale(probes, shape, scale):
    check_case(probes, make_case(probes, *shape, scale=scale))


@pytest.mark.parametrize("m", [1, 15, 16, 17, 127, 128, 129, 255, 256])
@pytest.mark.parametrize("sk", [1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512])
def test_dense_boundaries(probes, m, sk):
    check_case(probes, make_case(probes, 1, 1, m, sk))


@pytest.mark.parametrize("batch", [2, 9, 10, 11, 19, 20, 21, 39, 40, 41, 61])
def test_scheduler_and_pipeline(probes, batch):
    check_case(probes, make_case(probes, batch, 2, 128, 65), repeats=5)


@pytest.mark.parametrize(
    "sq,heads", [(7, 16), (256, 1), (16, 16), (4, 63), (2, 65), (2, 128), (128, 2)]
)
def test_bsnh_layout_and_offset(probes, sq, heads):
    check_case(probes, make_case(probes, 2, sq, heads, 511, offset=1))


@pytest.mark.parametrize("perturb", range(4))
def test_content_rope_halves(probes, perturb):
    check_case(probes, make_case(probes, 2, 2, 65, 129, perturb=perturb))


@pytest.mark.parametrize("amplitude", [0.125, 4.0, 8.0])
def test_amla_exponents(probes, amplitude):
    check_case(probes, make_case(probes, 1, 1, 128, 512, amplitude=amplitude))


@pytest.mark.parametrize("amplitude", [0.125, 4.0, 8.0])
def test_single_task_intermediates(probes, amplitude):
    case = make_case(probes, 1, 1, 17, 257, amplitude=amplitude)
    check_case(probes, case)
    s = case["schedule"]
    assert s.cores == s.block_dim == 1
    # One task leaves its QK, rounded aMLA P and FP32 PV in the first slot.
    # Compare only logical rows/columns; padding is not a numerical result.
    columns = (s.sk + 31) // 32 * 32
    qk_bytes, p_bytes = 2 * s.r1 * 4, 2 * s.r1 * 2
    for offset, size, dtype, width, used in (
        (0, s.r1 * 4, torch.float32, columns, s.sk),
        (qk_bytes, s.r1 * 2, probes[0]["dtype"], columns, s.sk),
        (qk_bytes + p_bytes, s.r2 * 4, torch.float32, 512, 512),
    ):
        tensors = [
            case["workspace"][name][offset : offset + size]
            .cpu()
            .view(dtype)
            .reshape(-1, width)[:17, :used]
            for name in ("migrated", "oracle")
        ]
        torch.testing.assert_close(*tensors, rtol=0, atol=0)


def make_rounding_case(probes):
    case = make_case(probes, 1, 1, 1, 2)
    dtype = probes[0]["dtype"]
    ulp = torch.finfo(dtype).eps
    for tensor in case["inputs"].values():
        tensor.zero_()
    # Uniform weights put O exactly on positive and negative halfway points,
    # with both even and odd low mantissas. BF16 output must round to even.
    v = case["inputs"]["v"]
    v[0, 0, 0] = torch.tensor([1, 1 + ulp, -1, -1 - ulp], dtype=dtype).repeat(128)
    v[0, 1, 0] = torch.tensor([1 + ulp, 1 + 2 * ulp, -1 - ulp, -1 - 2 * ulp], dtype=dtype).repeat(
        128
    )
    for tensor, (before, device, prefix) in zip(case["inputs"].values(), case["storage"][:4]):
        before[prefix : prefix + tensor.numel()] = tensor.flatten()
        device.copy_(before)
    if dtype == torch.bfloat16:
        expected = v.double().mean(dim=1, keepdim=True).to(dtype)
    else:
        # The source's FP16 CAST_ROUND rounds these ties away from zero.
        # PyTorch's ties-to-even cast is not the oracle for this instruction.
        expected = v[:, 1:2].clone()
    return case, expected


def test_output_rounding(probes):
    case, expected = make_rounding_case(probes)
    for _ in range(3):
        for name in ("oracle", "migrated"):
            launch(probes[0][name], case, name)
        torch.npu.synchronize()
        for name in ("oracle", "migrated"):
            torch.testing.assert_close(case["outputs"][name][0].cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(
                case["outputs"][name][1].cpu(), torch.full((1, 1, 1), 2.0).log(), rtol=0, atol=1e-7
            )


def test_interleaved_shapes_and_runtime_scale(probes):
    cases = [
        make_case(probes, *shape) for shape in ((2, 1, 8, 257), (1, 7, 16, 65), (21, 2, 128, 31))
    ]
    for case in cases + cases[::-1]:
        for name in ("oracle", "migrated"):
            launch(probes[0][name], case, name, scale=-0.23)
    torch.npu.synchronize()
    for case in cases:
        case["scale"] = -0.23
        # Check the queued, alternating-shape outputs before any new launch can
        # overwrite them; the repeated checks below also exercise scale reuse.
        refs = mla_reference_pair(**case["inputs"], softmax_scale=case["scale"])
        for i, name in enumerate(("out", "lse")):
            oracle = case["outputs"]["oracle"][i].cpu()
            migrated = case["outputs"]["migrated"][i].cpu()
            torch.testing.assert_close(migrated, oracle, rtol=0, atol=0)
            assert_fa_close(migrated, refs[i], refs[i + 2], name=name)
        check_case(probes, case)
        case["scale"] = 0.0
        check_case(probes, case)


@pytest.mark.parametrize(
    "shape",
    [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 257, 1), (1, 1, 1, 513)],
)
def test_unsupported_schedule(probes, shape):
    schedule, details = Schedule(), (ctypes.c_uint64 * 16)()
    assert probes[0]["schedule"].prepare_dense(*shape, ctypes.byref(schedule), details) != 0
