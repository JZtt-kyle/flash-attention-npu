"""Long dense KV: source task ring, repeated aMLA updates and public reuse."""

import pytest
import torch

from tests.test_flash_attn_npu_v3_mla_dense import make_case
from tests.test_flash_attn_npu_v3_mla_dense import probes as probes
from tests.test_flash_attn_npu_v3_mla_kvblocks import check_public
from tests.test_flash_attn_npu_v3_mla_multiblock import check_stream_reuse
from tests.test_flash_attn_npu_v3_mla_multiblock import (
    test_multiblock_compile_and_fake as compile_case,
)


SHAPES = [
    (1, 1, 1, 1025),
    (1, 1, 15, 1535),
    (1, 1, 16, 1536),
    (1, 1, 17, 1537),
    (2, 1, 8, 2049),
    (1, 1, 127, 4095),
    (1, 1, 128, 4096),
    (1, 1, 129, 4097),
    (1, 7, 16, 8191),
    (1, 7, 16, 8192),
    (2, 1, 8, 8193),
    (1, 257, 1, 8193),
    (2, 171, 3, 1537),
    (10, 3, 86, 4097),
    (1, 1, 17, 16383),
    (1, 1, 128, 16384),
    (21, 1, 8, 16384),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_long_kv_accuracy(probes, shape):
    check_public(probes, make_case(probes, *shape, offset=1))


@pytest.mark.parametrize("scale", [0.0, -0.23, 0.23])
def test_long_kv_scale(probes, scale):
    check_public(probes, make_case(probes, 1, 7, 16, 16383, scale=scale, offset=1))


@pytest.mark.parametrize("perturb", range(4))
def test_long_kv_content_rope(probes, perturb):
    check_public(probes, make_case(probes, 1, 7, 16, 8193, perturb=perturb, offset=1))


@pytest.mark.parametrize("pattern", ["flat", "rising", "falling", "alternating", "middle", "tail"])
def test_long_kv_running_max(probes, pattern):
    case = make_case(probes, 1, 3, 43, 8193, scale=1.0, offset=1)
    blocks = 17
    levels = {
        "flat": [2.0] * blocks,
        "rising": [i * 0.25 for i in range(blocks)],
        "falling": [-i for i in range(blocks)],
        "alternating": [64.0 if i % 2 else -8.0 for i in range(blocks)],
        "middle": [64.0 if i == 8 else 0.0 for i in range(blocks)],
        "tail": [64.0 if i == blocks - 1 else 0.0 for i in range(blocks)],
    }[pattern]
    # Exact logits expose maxima across task-ring rotations, both buffer slots
    # and the final one-element block. Opposite query signs also exercise
    # decreasing logits; large changes cross the source's -30 update clamp.
    inputs = case["inputs"]
    inputs["q"].zero_()
    inputs["qv"].zero_()
    inputs["k"].zero_()
    inputs["q"][..., 0] = 1
    inputs["q"][:, :, 1::2, 0] = -1
    for i, level in enumerate(levels):
        inputs["k"][:, i * 512 : (i + 1) * 512, :, 0] = level
    for tensor, (before, device, prefix) in zip(inputs.values(), case["storage"][:4]):
        before[prefix : prefix + tensor.numel()] = tensor.flatten()
        device.copy_(before)
    check_public(probes, case, max_change=levels)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 1, 8, 8193), (1, 257, 1, 16384)])
def test_long_kv_compile_and_fake(dtype, shape):
    compile_case(dtype, shape)


def test_long_kv_stream_reuse(probes):
    check_stream_reuse(probes, ((2, 7, 16, 8193), (1, 257, 1, 16384)))
