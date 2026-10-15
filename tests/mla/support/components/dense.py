"""Shared helpers extracted from tests/mla/unit/dense/test_flash_attn_npu_v3_mla_dense.py.

Independent 910B FP16/BF16 dense Kernel acceptance; no public FA3 dispatch.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import pytest
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.mla_cann_baseline import Case, make_inputs
from tests.mla.support.mla_dense_build import ROOT, build_probes


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


def make_case(
    probes,
    batch,
    sq,
    heads,
    sk,
    *,
    scale=None,
    offset=16,
    perturb=None,
    amplitude=1.0,
    schedule_details=None,
):
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
    if schedule_details is None:
        schedule, details = Schedule(), (ctypes.c_uint64 * 16)()
        assert (
            probes[0]["schedule"].prepare_dense(
                batch, sq, heads, sk, ctypes.byref(schedule), details
            )
            == 0
        )
    else:
        schedule, details = schedule_details
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
