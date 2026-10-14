"""One release v12 Trap per process; preserve the poisoned context until exit."""

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
from tests.mla.e2e.window_bsnd.test_flash_attn_npu_v3_mla_window_public import generate, cached
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import _inputs


def run(dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    inputs = _inputs(dtype=dtype)
    device = {name: inputs[name] for name in ("q", "k", "v", "qv")}
    if fault == "empty":
        device["q"] = device["q"][:, :0]
        device["qv"] = device["qv"][:, :0]
    lengths = torch.tensor([3, 17], dtype=torch.int32, device="npu")
    record = generate(device, lengths)
    cached(device, lengths, record)
    torch.npu.synchronize()
    if fault in ("stale", "negative", "over"):
        lengths[-1] = {"stale": 16, "negative": -1, "over": 18}[fault]
        if fault != "stale":
            record = None
    elif fault not in ("q-used", "window-stale"):
        word, value = {
            "version": (1, 5),
            "kind": (2, 5),
            "snapshot": (425, 16),
            "endpoints": (23, 0),
            "window-kind": (-3, 0),
            "window-left": (-2, 1),
            "window-right": (-1, 0),
            "empty": (1, 5),
        }[fault]
        record[word] = value
    start = time.monotonic()
    stage = "launch"
    try:
        with forbid_host_readback():
            if fault == "q-used":
                result = torch.ops.flash_attn_npu_3._mla_window_bsnd_forward(
                    **device,
                    window_left=2,
                    window_right=1,
                    causal=False,
                    cache_seqlens=lengths,
                    seqused_q=torch.tensor([3, 4], dtype=torch.int32, device="npu"),
                )
            else:
                result = cached(
                    device, lengths, record, window=(0, 0) if fault == "window-stale" else (2, 1)
                )
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
