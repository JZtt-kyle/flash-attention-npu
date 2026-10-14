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


def make_case(batch, sq, heads, sk, *, offset=16, perturb=None, seed=20260928):
    gen = torch.Generator().manual_seed(seed)
    m = sq * heads
    ma, na = (m + 15) // 16 * 16, (sk + 31) // 32 * 32
    qc = torch.randn(batch, m, 512, generator=gen).half()
    qr = torch.randn(batch, m, 64, generator=gen).half()
    kv = torch.randn(batch, sk, 512, generator=gen).half()
    kr = torch.randn(batch, sk, 64, generator=gen).half()
    if perturb is not None:
        qc.zero_()
        qr.zero_()
        target, lo, hi = ((qc, 0, 256), (qc, 256, 512), (qr, 0, 32), (qr, 32, 64))[perturb]
        target[..., lo:hi] = torch.randn(batch, m, hi - lo, generator=gen).half()
    p = torch.full((batch, 2, ma, na), 19.0, dtype=torch.float16)
    p[:, :, :m, :sk] = torch.randn(batch, 2, m, sk, generator=gen).half()
    inputs = [qc, qr, kv, kr, p]
    storage = []

    def guarded(t, prefix):
        # Guard allocations preserve nonzero storage offsets and input contents.
        host = torch.full((t.numel() + prefix + 16,), 123.0, dtype=t.dtype)
        host[prefix : prefix + t.numel()] = t.flatten()
        dev = host.npu()
        storage.append((host, dev, prefix))
        return dev[prefix : prefix + t.numel()].view(t.shape)

    device = [guarded(t, offset if i < 4 else 16) for i, t in enumerate(inputs)]
    # AscendC ListTensorDesc's documented pointer-only layout, for the oracle.
    desc = torch.tensor([24, 1 << 32, 0xFFFFFFFF, device[2].data_ptr()], dtype=torch.int64).npu()
    outputs = {
        name: [guarded(torch.full((batch, 2, ma, d), 12345.0), 16) for d in (na, 512)]
        for name in ("oracle", "migrated")
    }
    return dict(
        shape=(batch, sq, heads, sk),
        inputs=inputs,
        device=device,
        desc=desc,
        storage=storage,
        outputs=outputs,
        offset=offset,
    )


def launch(lib, case, which, cores, rounds):
    qc, qr, kv, kr, p = case["device"]
    qk, pv = case["outputs"][which]
    pointers = [qc, qr, kv, kr, case["desc"], p, qk, pv]
    error = lib.launch_cube(
        torch.npu.current_stream().npu_stream,
        *(t.data_ptr() for t in pointers),
        *case["shape"],
        cores,
        rounds,
    )
    assert error == 0, f"Cube probe launch failed: {error}"


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
