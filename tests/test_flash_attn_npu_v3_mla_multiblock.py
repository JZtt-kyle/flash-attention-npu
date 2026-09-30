"""Dense S1G partitions: source equality, metadata, tails and queued reuse."""

import json

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.common.compare import assert_fa_close
from tests.common.mla_ref import mla_reference_pair
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_dense import check_case, make_case
from tests.test_flash_attn_npu_v3_mla_dense import probes as probes
from tests.test_flash_attn_npu_v3_mla_forward import public
from tests.test_flash_attn_npu_v3_mla_kvcache import cached, generate


SHAPES = [
    (1, 257, 1, 1),
    (1, 1, 257, 17),
    (2, 7, 73, 65),  # 511 rows; blocks cut across query/head boundaries.
    (1, 4, 128, 512),
    (2, 3, 171, 511),  # 513 rows; final block has one active Vector lane.
    (1, 513, 1, 33),
    (19, 3, 171, 17),
    (20, 3, 171, 65),
    (21, 3, 171, 257),  # A core can cross batches and consume different tails.
    (41, 7, 73, 33),
    (10, 2, 129, 65),  # Source-adjusted odd M base.
    (10, 2, 192, 129),
    (10, 2, 255, 33),
    (3, 17, 257, 65),
    (1, 129, 128, 33),  # More M blocks than cores; exercise the three-task ring.
]


def check_public(probes, shape, scale=None, perturb=None):
    case = make_case(probes, *shape, scale=scale, perturb=perturb, offset=1)
    # Independently built fixed-source Kernel and migrated Kernel, with guarded
    # outputs/workspaces. CPU differential tests separately verify the schedule.
    check_case(probes, case)
    lengths = torch.full((shape[0],), shape[3], dtype=torch.int32, device="npu")
    record = generate(case["device"], lengths)
    saved = record.cpu().clone()
    results = []
    with forbid_host_readback():
        for _ in range(3):
            results += [
                public(case, scale=scale),
                cached(case["device"], None, lengths, scale),
                cached(case["device"], record, lengths, scale),
            ]
    torch.npu.synchronize()
    for result in results:
        for got, want in zip(result[:2], case["outputs"]["oracle"]):
            torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(record.cpu(), saved, rtol=0, atol=0)
    for before, device, _ in case["storage"][:4]:
        torch.testing.assert_close(device.cpu(), before, rtol=0, atol=0)
    with (probes[1] / "public-multiblock.jsonl").open("a") as log:
        log.write(
            json.dumps(
                dict(
                    shape=shape,
                    dtype=case["dtype"],
                    scale=scale,
                    perturb=perturb,
                    metadata=saved.tolist(),
                    calls_per_path=3,
                    source_equal=True,
                )
            )
            + "\n"
        )


@pytest.mark.parametrize("shape", SHAPES)
def test_multiblock_accuracy(probes, shape):
    check_public(probes, shape)


@pytest.mark.parametrize("scale", [0.0, -0.23, 0.23])
def test_multiblock_runtime_scale(probes, scale):
    check_public(probes, (2, 3, 171, 129), scale)


@pytest.mark.parametrize("perturb", range(4))
def test_multiblock_content_rope(probes, perturb):
    check_public(probes, (10, 2, 129, 33), perturb=perturb)


def test_multiblock_stream_reuse(probes):
    cases = [make_case(probes, *s) for s in ((10, 2, 129, 17), (21, 3, 171, 33))]
    records = [generate(case["device"]) for case in cases]
    snapshots = [record.cpu().clone() for record in records]
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    for stream in streams:
        stream.wait_stream(torch.npu.current_stream())
    results = []
    with forbid_host_readback():
        for iteration in range(16):
            index = iteration % 2
            device = {name: t.clone() for name, t in cases[index]["device"].items()}
            record = records[index].clone()
            streams[index].wait_stream(torch.npu.current_stream())
            with torch.npu.stream(streams[index]):
                results.append((index, cached(device, record, scale=-0.23)))
            del device, record
            poison = [torch.full_like(t, 123) for t in cases[index]["device"].values()]
            poison.append(torch.full_like(records[index], -1))
            del poison
    torch.npu.synchronize()
    refs = [mla_reference_pair(**case["inputs"], softmax_scale=-0.23) for case in cases]
    for index, result in results:
        for i, name in enumerate(("out", "lse")):
            assert_fa_close(result[i].cpu(), refs[index][i], refs[index][i + 2], name=name)
    for record, snapshot in zip(records, snapshots):
        torch.testing.assert_close(record.cpu(), snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 257, 1, 17), (2, 3, 171, 33), (10, 2, 129, 65)])
def test_multiblock_compile_and_fake(dtype, shape):
    b, sq, h, sk = shape

    def run(q, k, v, qv, lengths):
        record = generate(dict(q=q, k=k), lengths)
        return cached(dict(q=q, k=k, v=v, qv=qv), record, lengths)

    def inputs():
        return (
            torch.zeros((b, sq, h, 64), dtype=dtype, device="npu"),
            torch.zeros((b, sk, 1, 64), dtype=dtype, device="npu"),
            torch.ones((b, sk, 1, 512), dtype=dtype, device="npu"),
            torch.zeros((b, sq, h, 512), dtype=dtype, device="npu"),
            torch.full((b,), sk, dtype=torch.int32, device="npu"),
        )

    with FakeTensorMode():
        fake = run(*inputs())
        assert fake[0].shape == (b, sq, h, 512) and fake[1].shape == (b, h, sq)
    result = torch.compile(run, backend="eager", fullgraph=True)(*inputs())
    torch.testing.assert_close(result[0].cpu(), torch.ones((b, sq, h, 512), dtype=dtype))
    torch.testing.assert_close(result[1].cpu(), torch.full((b, h, sq), float(sk)).log())


@pytest.mark.parametrize("shape", [(1, 2**30 + 1, 1), (1, 1, 2**23), (1, 2**30 - 63, 4)])
def test_fake_forward_rejects_address_overflow(shape):
    b, sq, h = shape
    with FakeTensorMode():
        device = dict(
            q=torch.empty((b, sq, h, 64), dtype=torch.float16, device="npu"),
            qv=torch.empty((b, sq, h, 512), dtype=torch.float16, device="npu"),
            k=torch.empty((b, 1, 1, 64), dtype=torch.float16, device="npu"),
            v=torch.empty((b, 1, 1, 512), dtype=torch.float16, device="npu"),
        )
        with pytest.raises(RuntimeError, match="address/field bounds"):
            cached(device)
