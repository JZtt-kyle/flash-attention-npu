"""One release v21 Trap per process; retain the failed context until exit."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.e2e.split_mixed.test_public import prepare, metadata, call
from tests.mla.support.mla_split_mixed_cases import spec
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback


def run(dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = prepare(
        {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype],
        spec(
            q=(0, 0) if fault == "empty" else (7, 3),
            k=(31, 15),
            capacity=33,
            h=2,
        ),
    )
    record = metadata(c)
    call(c["args"], "op", record)
    torch.npu.synchronize()
    a = c["args"]
    if fault in ("stale-q", "stale-k"):
        a["cu_seqlens_q" if fault == "stale-q" else "cache_seqlens"][1] -= 1
    elif fault in ("negative", "over", "q-used"):
        if fault == "q-used":
            a["seqused_q"] = torch.tensor([7, 3], dtype=torch.int32, device="npu")
        a["seqused_q" if fault == "q-used" else "cache_seqlens"][-1] = {
            "negative": -1,
            "over": 34,
            "q-used": 4,
        }[fault]
        record = None
    elif fault == "split-stale":
        a["num_splits"] = 2
    elif fault == "window-stale":
        a["window_size"] = (0, 0)
    else:
        word, value = {
            "version": (1, 19),
            "kind": (2, 20),
            "prefix": (-2, 100),
            "prefix-high": (-1, 1),
            "split": (429, 1),
            "window": (427, 0),
            "snapshot": (410, 1),
            "empty": (1, 17),
        }[fault]
        record[word] = value
    started = time.monotonic()
    stage = "launch"
    try:
        with forbid_host_readback():
            result = call(c["args"], "op", record)
        stage = "event"
        event = torch.npu.Event()
        event.record()
        event.synchronize()
    except RuntimeError as error:
        p = Path(api.flash_attn_npu_3.__file__)
        report = dict(
            fault=fault,
            stage=stage,
            error=str(error),
            seconds=time.monotonic() - started,
            extension=str(p),
            extension_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        (Path(directory) / "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
        os._exit(0)
    raise AssertionError(f"invalid call completed: {fault} {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
