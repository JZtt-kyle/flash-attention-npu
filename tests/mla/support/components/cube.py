"""Shared helpers extracted from tests/mla/unit/dense/test_flash_attn_npu_v3_mla_cube.py.

910B intermediate QK/PV checks, independent of production MLA dispatch.
"""

import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers


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
