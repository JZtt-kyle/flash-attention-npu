"""Production FP16/BF16 MLA entry: source equality, math and launch lifetime."""

import json
from pathlib import Path

import pytest
import torch

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.common.mla_ref import mla_reference_pair
from tests.test_flash_attn_npu_v3_mla_contract import _forward
from tests.test_flash_attn_npu_v3_mla_dense import launch, make_case, make_rounding_case
from tests.test_flash_attn_npu_v3_mla_dense import probes as probes


def public(case, *, scale=None):
    return api.flash_attn_func(
        **case["device"], softmax_scale=scale, num_splits=1, return_attn_probs=True
    )


def check_forward(probes, shape, *, scale=None, perturb=None, offset=1, amplitude=1.0):
    case = make_case(
        probes, *shape, scale=scale, perturb=perturb, offset=offset, amplitude=amplitude
    )
    refs = mla_reference_pair(**case["inputs"], softmax_scale=case["scale"])
    launch(probes[0]["oracle"], case, "oracle")
    # Queue independent output/workspace allocations before synchronizing, then
    # check every result to detect accidental sharing between calls.
    results = [public(case, scale=scale) for _ in range(3)]
    torch.npu.synchronize()
    errors = []
    for result in results:
        for i, name in enumerate(("out", "lse")):
            actual = result[i].cpu()
            torch.testing.assert_close(actual, case["outputs"]["oracle"][i].cpu(), rtol=0, atol=0)
            assert_fa_close(actual, refs[i], refs[i + 2], name=name)
            errors.append(float((actual - refs[i]).abs().max()))
    for before, device, _ in case["storage"][:4]:
        torch.testing.assert_close(device.cpu(), before, rtol=0, atol=0)
    record = dict(
        dtype=case["dtype"],
        shape=shape,
        scale=scale,
        offset=offset,
        perturb=perturb,
        amplitude=amplitude,
        repeats=3,
        errors=errors,
    )
    with (Path(probes[1]) / "production-accuracy.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


@pytest.mark.parametrize("shape", [(2, 1, 8, 257), (1, 7, 16, 65)])
@pytest.mark.parametrize("scale", [None, 0.0, -0.23, 0.23])
def test_public_d01_d02(probes, shape, scale):
    check_forward(probes, shape, scale=scale)


@pytest.mark.parametrize("m", [1, 15, 16, 17, 127, 128, 129, 255, 256])
@pytest.mark.parametrize("sk", [1, 17, 33, 129, 257, 511, 512])
def test_public_boundaries(probes, m, sk):
    check_forward(probes, (1, 1, m, sk))


@pytest.mark.parametrize("batch", [9, 10, 11, 19, 20, 21, 39, 40, 41, 61])
def test_public_batches(probes, batch):
    check_forward(probes, (batch, 2, 128, 65))


@pytest.mark.parametrize("sq,heads", [(256, 1), (16, 16), (4, 63), (2, 65), (2, 128), (128, 2)])
def test_public_query_head_order(probes, sq, heads):
    check_forward(probes, (2, sq, heads, 511))


@pytest.mark.parametrize("perturb", range(4))
def test_public_content_rope(probes, perturb):
    check_forward(probes, (2, 2, 65, 129), perturb=perturb)


@pytest.mark.parametrize("amplitude", [0.125, 4.0, 8.0])
def test_public_amla(probes, amplitude):
    check_forward(probes, (1, 1, 128, 512), amplitude=amplitude)


def test_public_output_rounding(probes):
    case, expected = make_rounding_case(probes)
    launch(probes[0]["oracle"], case, "oracle")
    actual = public(case)
    for i in range(2):
        torch.testing.assert_close(actual[i], case["outputs"]["oracle"][i], rtol=0, atol=0)
    torch.testing.assert_close(actual[0].cpu(), expected, rtol=0, atol=0)


def test_interleaved_scales_and_streams(probes):
    cases = [
        make_case(probes, *shape) for shape in ((2, 1, 8, 257), (1, 7, 16, 65), (21, 2, 128, 31))
    ]
    torch.npu.synchronize()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    results = []
    for scale in (None, -0.23, 0.0, 0.23):
        for i, case in enumerate(cases):
            with torch.npu.stream(streams[i % 2]):
                results.append((case, scale, public(case, scale=scale)))
    torch.npu.synchronize()
    for case, scale, result in results:
        refs = mla_reference_pair(**case["inputs"], softmax_scale=scale)
        for i, name in enumerate(("out", "lse")):
            assert_fa_close(result[i].cpu(), refs[i], refs[i + 2], name=name)


def test_public_returns_out_only(probes):
    case = make_case(probes, 1, 3, 4, 17)
    out = api.flash_attn_func(**case["device"], num_splits=1)
    expected, _ = public(case)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("entry", ["eager", "cpp"])
@pytest.mark.parametrize("scale", [None, -0.23])
def test_internal_default_and_runtime_scale(probes, entry, scale):
    case = make_case(probes, 1, 7, 16, 65, scale=scale)
    actual = _forward(entry, dict(**case["device"], softmax_scale=scale))
    expected = public(case, scale=scale)
    for out, ref in zip(actual[:2], expected):
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_input_lifetime_on_consuming_stream(probes):
    case = make_case(probes, 21, 2, 128, 511)
    refs = mla_reference_pair(**case["inputs"])
    stream = torch.npu.Stream()
    outputs = []
    for _ in range(8):
        device = {name: tensor.clone() for name, tensor in case["device"].items()}
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            outputs.append(api.flash_attn_func(**device, num_splits=1, return_attn_probs=True))
        del device
        # Reallocate on the producer stream while the consuming stream may
        # still be running. Input lifetimes must be recorded with the allocator.
        overwrite = [torch.full_like(tensor, 123) for tensor in case["device"].values()]
        del overwrite
    torch.npu.synchronize()
    for out, lse in outputs:
        assert_fa_close(out.cpu(), refs[0], refs[2], name="out")
        assert_fa_close(lse.cpu(), refs[1], refs[3], name="lse")
