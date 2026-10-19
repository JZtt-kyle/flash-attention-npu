"""Cold graph and cross-thread ownership of the immutable TND causal mask."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import torch

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.graph import capture
from tests.mla.support.public_cases.int8_tnd import make_case, invoke, equal, check_math
from tests.mla.support.mla_validation_build import sha256


def run(output):
    c = make_case(uniform=True)
    # Neither captured mask has executed when the second graph is built.
    cold = [capture(lambda: invoke(c), stream=torch.npu.Stream(), warmup=False) for _ in range(2)]
    eager = invoke(c)
    check_math(c, eager)
    for graph, result in cold:
        graph.replay()
        torch.npu.synchronize()
        equal(result, eager)
        graph.reset()

    def in_thread():
        torch.npu.set_device(c["tensors"]["q"].device)
        return capture(lambda: invoke(c), stream=torch.npu.Stream(), warmup=False)

    with ThreadPoolExecutor(max_workers=1) as executor:
        graph, result = executor.submit(in_thread).result()
    # The creating host thread has ended; the fixed mask must still be owned.
    graph.replay()
    torch.npu.synchronize()
    equal(result, eager)
    graph.reset()
    extension = Path(api.flash_attn_npu_3.__file__)
    Path(output).write_text(
        json.dumps(
            dict(
                cold_graphs=2, thread_exit=True, extension=str(extension), sha256=sha256(extension)
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*sys.argv[1:])
