"""Minimal real graph workload for diagnosing CANN profiler collection.

Run directly with TorchNPU profiling, or pass --external under msprof. This
diagnostic does not provide MLA correctness or performance acceptance evidence.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import torch
import torch_npu
from torch.utils._pytree import tree_flatten


def run(
    directory,
    external=False,
    keep_model=False,
    workload="arithmetic",
    pre_capture=False,
    fresh_stream=False,
    layout="bsnd",
):
    directory.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    x = torch.ones((32, 32), dtype=torch.float16, device="npu")

    def call():
        return torch.mm(x, x) + 1

    if workload == "mla":
        from tests.mla.support.public_cases.graph import make_case, metadata, invoke

        case = make_case(layout, "compressed", q=(3, 1), k=(65, 33), pack=None, splits=1)
        record = metadata(case)

        def call():
            return invoke(case, record)

    expected = call()
    torch.npu.synchronize()
    prior = None
    if pre_capture:
        prior = torch.npu.NPUGraph()
        with torch.npu.graph(prior):
            prior_result = call()
    profiler = (
        nullcontext()
        if external
        else torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(directory / "profiler")
            ),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                data_simplification=False,
            ),
        )
    )
    with profiler:
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream() if fresh_stream else None
        with torch.npu.graph(graph, stream=stream):
            result = call()
        for _ in range(5):
            graph.replay()
            torch.npu.synchronize()
        actual_values, actual_spec = tree_flatten(result)
        expected_values, expected_spec = tree_flatten(expected)
        assert actual_spec == expected_spec
        for actual, reference in zip(actual_values, expected_values):
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0, atol=0)
        if not keep_model:
            graph.reset()
    if keep_model:
        graph.reset()
    if prior is not None:
        prior.reset()
        del prior_result
    (directory / "workload.json").write_text(
        json.dumps(
            dict(
                external=external,
                keep_model=keep_model,
                replays=5,
                workload=workload,
                pre_capture=pre_capture,
                fresh_stream=fresh_stream,
                layout=layout,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--keep-model", action="store_true")
    parser.add_argument("--workload", choices=("arithmetic", "mla"), default="arithmetic")
    parser.add_argument("--pre-capture", action="store_true")
    parser.add_argument("--fresh-stream", action="store_true")
    parser.add_argument("--layout", choices=("bsnd", "tnd"), default="bsnd")
    args = parser.parse_args()
    run(
        args.output,
        args.external,
        args.keep_model,
        args.workload,
        args.pre_capture,
        args.fresh_stream,
        args.layout,
    )
