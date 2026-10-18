"""Replay ordering, graph-owned temporaries and public metadata reuse."""

import gc
import pytest
import torch

from tests.mla.support.public_cases.graph import (
    LAYOUTS,
    make_case,
    metadata,
    invoke,
    compare,
    equal,
    capture,
    kv_lengths,
)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_multistream_reuse(layout, representation, dtype):
    c = make_case(layout, representation, dtype, pack=None, splits=1)
    other = dict(c, t={name: value.clone() for name, value in c["t"].items()})
    lengths = kv_lengths(c)
    donor = lengths.clone()
    first_stream, second_stream = torch.npu.Stream(), torch.npu.Stream()

    def produce():
        lengths.copy_(donor)
        record = metadata(c)
        return record, invoke(c, record)

    first_graph, (record, first) = capture(produce, first_stream)
    first_graph.replay()
    torch.npu.synchronize()
    second_graph, second = capture(lambda: invoke(other, record), second_stream)
    for values in ([0, 1], [33, 7], [65, 33]):
        c["klens"] = other["klens"] = list(values)
        donor.copy_(donor.new_tensor(values))
        first_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(first_stream):
            first_graph.replay()
        second_stream.wait_stream(first_stream)
        with torch.npu.stream(second_stream):
            second_graph.replay()
        torch.npu.current_stream().wait_stream(second_stream)
        equal(first, invoke(c))
        equal(second, invoke(other))
        compare(c, first)
        compare(other, second)
        assert first[0].data_ptr() != second[0].data_ptr()
    first_graph.reset()
    second_graph.reset()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation,dtype", [("compressed", "bf16"), ("prefill", "fp16")])
def test_temporary_lifetime(layout, representation, dtype):
    c = make_case(layout, representation, dtype, pack=False, splits=3, mask="window", cap=15)
    eager = invoke(c)

    # Neither the generated record, cloned Q nor intermediate attention output
    # escapes the call. Only final public outputs remain alive in Python.
    def call():
        temporary = dict(c, t=dict(c["t"], q=c["t"]["q"].clone()))
        outputs = invoke(temporary, metadata(temporary))
        return tuple(value.clone() for value in outputs)

    graph, result = capture(call)
    gc.collect()
    for _ in range(3):
        pressure = [torch.full((262144,), 99.0, device="npu") for _ in range(8)]
        graph.replay()
        torch.npu.synchronize()
        equal(result, eager)
        del pressure
    # Input storage stays fixed, while replay must read its new content through
    # the temporary clone captured above.
    c["raw"]["q"].mul_(0.5)
    c["t"]["q"].copy_(c["raw"]["q"])
    eager = invoke(c)
    graph.replay()
    torch.npu.synchronize()
    equal(result, eager)
    compare(c, result)
    graph.reset()
    gc.collect()
    pressure = [torch.full((262144,), -99.0, device="npu") for _ in range(8)]
    equal(result, eager)
    del pressure
