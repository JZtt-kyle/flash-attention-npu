"""One v9 release Trap per process, including empty-capacity descriptor validation."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.public_cases.causal_mixed_public import prepare, spec, call, metadata


def run(dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    s = spec(uq=[5, 0, 3]) if fault == "q-used" else spec()
    if fault == "empty-q":
        s = spec(q=(0, 0, 0), k=(33, 0, 17), sq=7)
    if fault == "empty-k":
        s = spec(q=(7, 0, 3), k=(0, 0, 0), capacity=0)
    args = prepare(dtype, s)["args"]
    call(args, "op")
    torch.npu.synchronize()
    if fault in (
        "stale-prefix",
        "stale-length",
        "old-record",
        "v8-record",
        "capacity",
        "signature",
    ):
        args["scheduler_metadata"] = metadata(args)
        torch.npu.synchronize()
    name, index, value = {
        "q-start": ("cu_seqlens_q", 0, 1),
        "q-descending": ("cu_seqlens_q", 2, 6),
        "q-tail": ("cu_seqlens_q", 3, 9),
        "q-used": ("seqused_q", 0, 8),
        "k-used": ("cache_seqlens", 1, -1),
        "empty-q": ("cache_seqlens", 0, 34),
        "empty-k": ("cu_seqlens_q", 0, 1),
        "stale-prefix": ("cu_seqlens_q", 1, 6),
        "stale-length": ("cache_seqlens", 0, 2),
        "old-record": ("scheduler_metadata", 1, 4),
        "v8-record": ("scheduler_metadata", 1, 8),
        "capacity": ("scheduler_metadata", 408, 34),
        "signature": ("scheduler_metadata", 4, 2 if dtype == torch.float16 else 1),
    }[fault]
    args[name][index] = value
    start = time.monotonic()
    stage = "launch"
    try:
        with forbid_host_readback():
            result = call(args, "op")
        stage = "event"
        event = torch.npu.Event()
        event.record()
        event.synchronize()
    except RuntimeError as error:
        path = Path(api.flash_attn_npu_3.__file__)
        report = dict(
            fault=fault,
            stage=stage,
            error=str(error),
            seconds=time.monotonic() - start,
            extension=str(path),
            extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        (Path(directory) / "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
        os._exit(0)
    raise AssertionError(f"Invalid call completed: {fault}, {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
