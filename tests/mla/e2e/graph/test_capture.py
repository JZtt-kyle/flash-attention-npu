"""Real graph capability checks before the MLA capture acceptance matrix."""

import pytest
import torch
import torch_npu  # noqa: F401
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.mla.support.public_cases.gqa_options import invoke as invoke_entry
from tests.mla.support.public_cases.graph import (
    LAYOUTS,
    CASES,
    DAILY,
    make_case,
    metadata,
    invoke,
    compare,
    equal,
    capture,
    static_replay,
    dynamic_replay,
)


def test_runtime_capture():
    """A graph must replay changed device input even without a custom operator."""
    value = torch.ones(4, device="npu")
    value + 1
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = value + 1
    for number in (2, 5):
        value.fill_(number)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result.cpu(), torch.full((4,), number + 1.0))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "layout,representation,spec",
    [
        pytest.param("bsnd", "compressed", dict(splits=1, pack=None), id="bsnd-single"),
        pytest.param(
            "paged_tnd",
            "prefill",
            dict(splits=0, mask="window", cap=15, margin=1),
            id="paged-tnd-auto-window",
        ),
    ],
)
def test_mla_capture(layout, representation, dtype, spec):
    c = make_case(layout, representation, dtype, **spec)
    static_replay(c)
    dynamic_replay(c)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("spec", [pytest.param(s, id=s["id"]) for s in CASES])
def test_behavior(layout, representation, dtype, spec):
    c = make_case(layout, representation, dtype, **spec)
    static_replay(c)
    dynamic_replay(c)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_legacy_producers(layout, dtype):
    # Each compressed default producer previously used auxiliary-stream events;
    # retain every layout/dtype independently of the protocol pair selection.
    c = make_case(layout, "compressed", dtype, pack=None, splits=1)
    static_replay(c)
    dynamic_replay(c)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("spec", [pytest.param(s, id=s["id"]) for s in (CASES[0], CASES[6])])
def test_func_capture(representation, dtype, spec):
    # func has no length tensors: the default compressed path captures the
    # dense producer, distinct from cache's device-KV-length producer.
    c = make_case("bsnd", representation, dtype, q=(7, 7), k=(65, 65), **spec)

    def call():
        return invoke_entry(c, entry="func")

    graph, result = capture(call)
    for _ in range(3):
        c["raw"]["q"].mul_(0.5)
        c["t"]["q"].copy_(c["raw"]["q"])
        graph.replay()
        torch.npu.synchronize()
        equal(result, call())
        compare(c, result)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("spec", [pytest.param(s, id=s["id"]) for s in CASES if s["id"] in DAILY])
def test_daily(layout, representation, dtype, spec):
    c = make_case(layout, representation, dtype, **spec)
    dynamic_replay(c)


@pytest.mark.parametrize("layout", ["tnd", "mixed", "paged_tnd"])
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_prefix_updates(layout, representation, dtype):
    c = make_case(
        layout, representation, dtype, pack=False, splits=0, mask="window", maxima=(10, 65)
    )
    prefixes = c["lengths"]["cu_seqlens_q"]
    donor = prefixes.clone()

    def call():
        prefixes.copy_(donor)
        return invoke(c, metadata(c))

    graph, result = capture(call)
    for allocation in ([0, 10], [5, 5], [7, 3]):
        c["q"] = c["qlens"] = allocation
        donor.copy_(donor.new_tensor([0, allocation[0], sum(allocation)]))
        graph.replay()
        torch.npu.synchronize()
        equal(result, invoke(c, metadata(c)))
        compare(c, result)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_kv_prefix_updates(representation, dtype):
    c = make_case("tnd", representation, dtype, pack=None, splits=1)
    prefix = c["lengths"]["cu_seqlens_k"]
    donor = prefix.clone()

    def call():
        prefix.copy_(donor)
        return invoke(c)

    graph, result = capture(call)
    for allocation in ([33, 65], [65, 33]):
        c["k"] = c["klens"] = allocation
        donor.copy_(donor.new_tensor([0, allocation[0], sum(allocation)]))
        graph.replay()
        torch.npu.synchronize()
        equal(result, invoke(c))
        compare(c, result)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_query_used_update(representation, dtype):
    c = make_case("tnd", representation, dtype, pack=False, splits=0, mask="window", used_q=(3, 1))
    used = c["lengths"]["seqused_q"]
    donor = used.clone()

    def call():
        used.copy_(donor)
        return invoke(c)

    graph, result = capture(call)
    for values in ([0, 0], [1, 1], [3, 1]):
        c["qlens"] = values
        donor.copy_(donor.new_tensor(values))
        graph.replay()
        torch.npu.synchronize()
        equal(result, invoke(c))
        compare(c, result)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("empty", ["query", "kv", "batch"])
def test_empty(layout, representation, dtype, empty):
    spec = (
        dict(q=(0, 0))
        if empty == "query"
        else dict(k=(0, 0))
        if empty == "kv"
        else dict(q=(), k=())
    )
    c = make_case(layout, representation, dtype, splits=0, mask="causal", **spec)
    static_replay(c)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("representation,dtype", [("compressed", "fp16"), ("prefill", "bf16")])
def test_fake_and_compile(layout, representation, dtype):
    spec = dict(pack=False, splits=3, mask="window", cap=15)
    c = make_case(layout, representation, dtype, **spec)
    # Each capture has fixed shapes and scalar options. Do not let preceding
    # parameter cases generalize their different scales into symbolic floats.
    torch._dynamo.reset()
    compiled = torch.compile(lambda: invoke(c), backend="eager", fullgraph=True, dynamic=False)
    graph, result = capture(compiled)
    graph.replay()
    torch.npu.synchronize()
    compare(c, result)
    with FakeTensorMode():
        fake = make_case(layout, representation, dtype, **spec)
        outputs = invoke(fake, metadata(fake))
        for x, y in zip(result, outputs):
            assert (x.shape, x.stride(), x.dtype, x.device) == (
                y.shape,
                y.stride(),
                y.dtype,
                y.device,
            )
