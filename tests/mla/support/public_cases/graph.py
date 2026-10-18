"""Graph fixtures over existing five-layout inputs and cached dual-precision math."""

import torch

from tests.mla.support.public_cases import gqa_options as public
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback

LAYOUTS = tuple(public.LAYOUTS)
CASES = [
    dict(id="legacy-single-none", pack=None, splits=1),
    dict(id="legacy-single-causal", pack=None, splits=1, mask="causal"),
    dict(id="legacy-single-window", pack=None, splits=1, mask="window"),
    dict(id="legacy-explicit-causal", pack=True, splits=3, mask="causal", strided=True),
    dict(id="legacy-auto-window", pack=None, splits=0, mask="window"),
    dict(id="legacy-softcap-explicit", pack=None, splits=3, mask="causal", cap=15),
    dict(id="unpacked-auto-causal", pack=False, margin=1, splits=0, mask="causal"),
    dict(id="unpacked-softcap-single", pack=False, splits=1, mask="window", cap=15),
    dict(id="packed-softcap-auto", pack=True, margin=1, splits=0, mask="window", cap=30),
]
DAILY = {"unpacked-auto-causal"}


def make_case(layout, representation, dtype="fp16", **spec):
    spec = dict(spec)
    spec.setdefault("q", (7, 3))
    spec.setdefault("k", (65, 33))
    return public.make_case(layout, representation, dtype, **spec)


def metadata(c):
    # Public varlen always creates its own TND Q/KV metadata inside forward.
    return None if c["layout"] == "tnd" else public.metadata(c)


def invoke(c, record=None):
    return public.invoke(c, record)


def compare(c, result):
    # Final O/LSE do not depend on split endpoints or packing. Reuse the existing
    # independent full-attention math with one explicit interval per Q head.
    # Accumulators are separately compared bit-for-bit with eager below.
    oracle = dict(c, options=dict(c["options"], num_splits=1, pack_gqa=False, sm_margin=0))
    record = public.metadata(oracle, geometry=c["layout"] == "tnd")
    return public.compare(oracle, result[:2], record)


def equal(actual, expected):
    assert len(actual) == len(expected)
    for x, y in zip(actual, expected):
        assert (x.shape, x.stride(), x.dtype, x.device) == (
            y.shape,
            y.stride(),
            y.dtype,
            y.device,
        )
        torch.testing.assert_close(x.cpu(), y.cpu(), rtol=0, atol=0)


def capture(call, stream=None, *, warmup=True):
    if warmup:
        call()
        torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        with forbid_host_readback():
            result = call()
    return graph, result


def kv_lengths(c):
    if isinstance(c["lengths"], dict):
        if c["lengths"].get("seqused_k") is None:
            c["lengths"]["seqused_k"] = torch.tensor(c["klens"], dtype=torch.int32, device="npu")
        return c["lengths"]["seqused_k"]
    return c["lengths"]


def static_replay(c):
    record = metadata(c)
    eager = invoke(c, record)
    compare(c, eager)
    graph, result = capture(lambda: invoke(c, record))
    for _ in range(3):
        graph.replay()
        torch.npu.synchronize()
        equal(result, eager)
    compare(c, result)
    return result


def dynamic_replay(c):
    target = kv_lengths(c)
    donor = target.clone()

    def call():
        target.copy_(donor)
        record = metadata(c)
        # One generated record, two public layers, independent output/scratch.
        return record, invoke(c, record), invoke(c, record)

    graph, (record, first, second) = capture(call)
    for lengths in ([0, 1], [33, 7], list(c["klens"])):
        c["klens"] = list(lengths)
        donor.copy_(donor.new_tensor(lengths))
        graph.replay()
        torch.npu.synchronize()
        fresh = metadata(c)
        eager = invoke(c, fresh)
        equal(first, eager)
        equal(second, eager)
        assert first[0].data_ptr() != second[0].data_ptr()
        if record is not None:
            torch.testing.assert_close(record.cpu(), fresh.cpu(), rtol=0, atol=0)
        compare(c, first)
    return first
