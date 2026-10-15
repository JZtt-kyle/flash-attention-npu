"""910B intermediate QK/PV checks, independent of production MLA dispatch."""

import ctypes
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_cube_build import ROOT, build_probes

from tests.mla.support.components.cube import (
    launch as launch,
    make_case as make_case,
)


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    source = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    if not source.exists():
        pytest.skip("set MLA_CANN_SOURCE_ROOT to a checkout containing the pinned CANN commit")
    artifacts = Path(os.environ.get("MLA_CUBE_ARTIFACT_DIR", tmp_path_factory.mktemp("mla-cube")))
    libraries = build_probes(artifacts)
    torch.npu.set_device(0)
    name = torch.npu.get_device_name(0)
    assert "910B" in name, f"910B acceptance required, found {name}"
    loaded = {}
    for name, path in libraries.items():
        lib = ctypes.CDLL(path)
        lib.launch_cube.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 6
        lib.launch_cube.restype = ctypes.c_int
        loaded[name] = lib
    return loaded, artifacts


def check_case(probes, case, cores, rounds=3):
    libraries, artifacts = probes
    for name in ("oracle", "migrated"):
        launch(libraries[name], case, name, cores, rounds)
    torch.npu.synchronize()
    batch, sq, heads, sk = case["shape"]
    m = sq * heads
    qc, qr, kv, kr, p = [t.double() for t in case["inputs"]]
    qk = qc @ kv.transpose(-1, -2) + qr @ kr.transpose(-1, -2)
    qk_abs = qc.abs() @ kv.abs().transpose(-1, -2) + qr.abs() @ kr.abs().transpose(-1, -2)
    pv = p[:, :, :m, :sk] @ kv[:, None]
    pv_abs = p[:, :, :m, :sk].abs() @ kv[:, None].abs()
    active = set()
    for core in range(cores):
        batches = list(range(core, batch, cores))
        for task in range(len(batches) * rounds):
            active.add((batches[task % len(batches)], task % 2))
    results = {name: [t.cpu() for t in case["outputs"][name]] for name in libraries}
    errors = [0.0, 0.0]
    digest = hashlib.sha256()
    for b in range(batch):
        for slot in range(2):
            for i, (ref, abs_sum, k) in enumerate(
                ((qk[b], qk_abs[b], 576), (pv[b, slot], pv_abs[b, slot], sk))
            ):
                n = sk if i == 0 else 512
                actual = results["migrated"][i][b, slot, :m, :n]
                source = results["oracle"][i][b, slot, :m, :n]
                if (b, slot) not in active:
                    assert (results["migrated"][i][b, slot] == 12345).all()
                    assert (results["oracle"][i][b, slot] == 12345).all()
                    continue
                torch.testing.assert_close(actual, source, rtol=0, atol=0)
                assert torch.isfinite(actual).all()
                # FP16 products are exact in FP32 for these bounded inputs.
                # Standard gamma_k forward-error bound for FP32 accumulation;
                # this mathematical check is independent of both Cube kernels.
                u = 2.0**-24
                bound = (k * u / (1 - k * u)) * abs_sum + 1e-6
                error = (actual.double() - ref).abs()
                assert (error <= bound).all(), (
                    case["shape"],
                    i,
                    float(error.max()),
                    float(bound.max()),
                )
                errors[i] = max(errors[i], float(error.max()))
                digest.update(actual.contiguous().numpy().tobytes())
    offset = case["offset"]
    for i, (before, device, prefix) in enumerate(case["storage"]):
        after = device.cpu()
        if i < 5:
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        else:
            torch.testing.assert_close(after[:prefix], before[:prefix], rtol=0, atol=0)
            torch.testing.assert_close(after[-16:], before[-16:], rtol=0, atol=0)
    record = dict(
        shape=case["shape"],
        cores=cores,
        rounds=rounds,
        offset=offset,
        active_slots=len(active),
        max_absolute_error=errors,
        result_sha256=digest.hexdigest(),
    )
    with (artifacts / "accuracy.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


@pytest.mark.parametrize("m", [1, 15, 16, 17, 127, 128, 129, 255, 256])
@pytest.mark.parametrize("sk", [1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512])
def test_cube_boundaries(probes, m, sk):
    check_case(probes, make_case(1, 1, m, sk), 1)


@pytest.mark.parametrize("batch", [1, 2, 19, 20, 21, 39, 40, 41])
def test_batch_and_buffer_reuse(probes, batch):
    check_case(probes, make_case(batch, 3, 17, 257), min(batch, 20), rounds=5)


@pytest.mark.parametrize(
    "sq,heads", [(7, 16), (256, 1), (16, 16), (4, 63), (2, 65), (2, 128), (128, 2)]
)
def test_query_head_copy_segments(probes, sq, heads):
    check_case(probes, make_case(2, sq, heads, 511, offset=1), 1)


@pytest.mark.parametrize("perturb", range(4))
def test_content_and_rope_halves(probes, perturb):
    check_case(probes, make_case(2, 2, 65, 129, perturb=perturb), 1)


def test_interleaved_launches(probes):
    # Enqueue unlike shapes without a host sync, then reuse each allocation.
    cases = [make_case(2, 1, 8, 257), make_case(1, 7, 16, 65), make_case(2, 2, 128, 511)]
    for case in cases + cases[::-1]:
        for name, lib in probes[0].items():
            launch(lib, case, name, 1, 3)
    for case in cases:
        check_case(probes, case, 1)
