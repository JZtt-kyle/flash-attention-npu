"""Reject stale external metadata or invalid dynamic prefixes during real replay."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import torch

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.graph import make_case, metadata, invoke, capture, kv_lengths
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback


def run(layout, representation, dtype, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = make_case(layout, representation, dtype, pack=None, splits=1)
    lengths = kv_lengths(c)
    record = metadata(c)
    graph, result = capture(lambda: invoke(c, record))
    graph.replay()
    torch.npu.synchronize()
    if layout == "tnd":
        c["lengths"]["cu_seqlens_q"][1] = -1
    else:
        lengths[0] -= 1
    stage = "replay"
    try:
        with forbid_host_readback():
            graph.replay()
        stage = "event"
        event = torch.npu.Event()
        event.record()
        event.synchronize()
    except RuntimeError as error:
        path = Path(api.flash_attn_npu_3.__file__)
        (Path(directory) / "runtime-error.json").write_text(
            json.dumps(
                dict(
                    stage=stage,
                    error=str(error),
                    extension=str(path),
                    extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
        os._exit(0)
    raise AssertionError(f"Invalid graph lengths completed: {layout}: {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
