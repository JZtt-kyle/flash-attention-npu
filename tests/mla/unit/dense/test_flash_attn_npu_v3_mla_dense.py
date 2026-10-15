"""Independent 910B FP16/BF16 dense Kernel acceptance; no public FA3 dispatch."""

import ctypes

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.compare import assert_fa_close
from tests.mla.support.mla_ref import mla_reference_pair

from tests.mla.support.components.dense import (
    Schedule as Schedule,
    check_case as check_case,
    launch as launch,
    load_probes as load_probes,
    make_case as make_case,
    probes as probes,
)


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
    [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 2**23, 1), (1, 1, 1, 16385)],
)
def test_schedule_boundaries(probes, shape):
    schedule, details = Schedule(), (ctypes.c_uint64 * 16)()
    status = probes[0]["schedule"].prepare_dense(*shape, ctypes.byref(schedule), details)
    batch, sq, heads, sk = shape
    if heads and (batch == 0 or sq == 0 or sk == 0):
        # M04c4 accepts global empty capacities as validation-only schedules.
        # They reserve one launch core, with no attention tiles or scratch.
        assert status == 0
        assert (schedule.batch, schedule.sq, schedule.heads, schedule.sk) == shape
        assert schedule.cores == 1 and schedule.block_dim == 1
        assert schedule.m_base == schedule.s2_base == schedule.r1 == schedule.r2 == 0
        assert not any((*schedule.b_end, *schedule.m_end, *schedule.s_end))
        assert not any((details[0], details[1], *details[9:14], details[15]))
    else:
        assert status != 0
