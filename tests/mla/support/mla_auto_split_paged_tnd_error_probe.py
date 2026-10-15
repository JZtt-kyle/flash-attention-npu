"""One release v26 Trap per process; retain the failed context until exit."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.auto_split_paged_tnd_public import prepare, metadata, call
from tests.mla.support.mla_auto_split_paged_tnd_cases import spec
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback


def run(dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = prepare(
        {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype],
        spec(q=0 if fault == "empty" else 7, lengths=(31, 15), capacity=112, heads=2),
    )
    record = metadata(c)
    call(c["args"], "op", record)
    torch.npu.synchronize()
    a = c["args"]
    if fault == "stale-q":
        a["cu_seqlens_q"][1] -= 1
    elif fault == "prefix-zero":
        a["cu_seqlens_q"][0] = 1
        record = None
    elif fault == "prefix-over":
        a["cu_seqlens_q"][-1] += 1
        record = None
    elif fault == "bsnd-version":
        record[1] = record[2] = 24
    elif fault == "stale-k":
        a["cache_seqlens"][1] -= 1
    elif fault.startswith("page"):
        if fault == "page-zero-used":
            a["seqused_q"] = torch.zeros(2, dtype=torch.int32, device="npu")
            record = None
        a["page_table"][0, 0] = -1 if fault == "page-negative" else a["k"].shape[0]
    elif fault in ("negative", "over", "q-used"):
        if fault == "q-used":
            a["seqused_q"] = torch.tensor([7, 7], dtype=torch.int32, device="npu")
        a["seqused_q" if fault == "q-used" else "cache_seqlens"][-1] = {
            "negative": -1,
            "over": 113,
            "q-used": 8,
        }[fault]
        record = None
    elif fault == "version-stale":
        record[1] = record[2] = 25
    elif fault == "window-stale":
        a["window_size"] = (0, 0)
    else:
        word, value = {
            "version": (1, 19),
            "kind": (2, 20),
            "prefix": (-2, 100),
            "prefix-high": (-1, 1),
            "aic": (422, 100),
            "fd": (500, 27),
            "vector": (502, 53),
            "window": (419, 0),
            "snapshot": (412, 1),
            "empty": (1, 17),
            "budget": (367, 1),
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
