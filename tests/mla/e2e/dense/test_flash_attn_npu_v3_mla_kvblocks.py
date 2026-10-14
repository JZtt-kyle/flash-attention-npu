"""Two KV blocks: nUpdate, PV accumulation and non-FD public metadata reuse."""

import json

import pytest
import torch

from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_dense import check_case, make_case
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_dense import probes as probes
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_forward import public
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_kvcache import cached, generate
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_multiblock import check_stream_reuse
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_multiblock import (
    test_multiblock_compile_and_fake as compile_case,
)


SHAPES = [
    (1, 1, 1, 513),
    (1, 1, 15, 527),
    (1, 1, 16, 528),
    (1, 1, 17, 529),
    (1, 1, 127, 543),
    (1, 1, 128, 544),
    (1, 1, 129, 545),
    (2, 1, 8, 513),
    (1, 7, 16, 769),
    (2, 2, 128, 1023),
    (1, 257, 1, 1024),
    (2, 171, 3, 513),
    (10, 3, 86, 769),
    (10, 2, 255, 1024),
    (19, 3, 171, 513),
    (20, 3, 171, 1023),
    (21, 3, 171, 1024),
    (1, 129, 128, 513),
]


def check_public(probes, case, *, max_change=None):
    check_case(probes, case)
    b, _, _, sk = case["shape"]
    lengths = torch.full((b + 2,), sk, dtype=torch.int32, device="npu")[1:-1]
    record = generate(case["device"], lengths)
    snapshot = record.cpu().clone()
    assert snapshot[16] == 512 and snapshot[103:106].tolist() == [0, 0, 0]
    results = []
    with forbid_host_readback():
        for _ in range(3):
            results += [
                public(case, scale=case["scale"]),
                cached(case["device"], None, lengths, case["scale"]),
                cached(case["device"], record, lengths, case["scale"]),
            ]
    torch.npu.synchronize()
    for result in results:
        for got, want in zip(result[:2], case["outputs"]["oracle"]):
            torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(record.cpu(), snapshot, rtol=0, atol=0)
    for before, device, _ in case["storage"][:4]:
        torch.testing.assert_close(device.cpu(), before, rtol=0, atol=0)
    with (probes[1] / "public-kvblocks.jsonl").open("a") as log:
        log.write(
            json.dumps(
                dict(
                    shape=case["shape"],
                    dtype=case["dtype"],
                    scale=case["scale"],
                    perturb=case["perturb"],
                    max_change=max_change,
                    metadata=snapshot.tolist(),
                    calls_per_path=3,
                    source_equal=True,
                )
            )
            + "\n"
        )


@pytest.mark.parametrize("shape", SHAPES)
def test_two_kv_block_accuracy(probes, shape):
    check_public(probes, make_case(probes, *shape, offset=1))


@pytest.mark.parametrize("scale", [0.0, -0.23, 0.23])
def test_two_kv_block_scale(probes, scale):
    check_public(probes, make_case(probes, 2, 171, 3, 1023, scale=scale, offset=1))


@pytest.mark.parametrize("perturb", range(4))
def test_two_kv_block_content_rope(probes, perturb):
    check_public(probes, make_case(probes, 10, 3, 86, 769, perturb=perturb, offset=1))


@pytest.mark.parametrize("change", [-8.0, 0.0, 0.25, 1.0, 20.0, 64.0])
def test_running_max_changes_and_nupdate_clamp(probes, change):
    case = make_case(probes, 2, 3, 171, 769, scale=1.0, offset=1)
    # Exact logits in both blocks, alternating query signs. Positive increases
    # exercise aMLA's exponent/cof update; 64 crosses the source -30 clamp.
    inputs = case["inputs"]
    inputs["q"].zero_()
    inputs["qv"].zero_()
    inputs["k"].zero_()
    inputs["q"][..., 0] = 1
    inputs["q"][:, :, 1::2, 0] = -1
    inputs["k"][:, :512, :, 0] = 2
    inputs["k"][:, 512:, :, 0] = 2 + change
    for tensor, (before, device, prefix) in zip(inputs.values(), case["storage"][:4]):
        before[prefix : prefix + tensor.numel()] = tensor.flatten()
        device.copy_(before)
    check_public(probes, case, max_change=change)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 1, 8, 513), (2, 171, 3, 1024)])
def test_two_kv_blocks_compile_and_fake(dtype, shape):
    compile_case(dtype, shape)


def test_two_kv_blocks_stream_reuse(probes):
    check_stream_reuse(probes, ((10, 3, 86, 513), (21, 171, 3, 1024)))
