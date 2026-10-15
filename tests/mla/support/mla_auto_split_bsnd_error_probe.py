"""One release v18 Trap per process; retain the failed context until exit."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.auto_split_bsnd_public import make_case, metadata, invoke
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback


def run(dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = make_case(dtype, shape=(2, 0, 2, 33) if fault == "empty" else (2, 7, 2, 1057))
    record = metadata(c)
    invoke(c, record)
    torch.npu.synchronize()
    if fault in ("stale", "negative", "over"):
        c["lengths"][-1] = {"stale": 3, "negative": -1, "over": 1058}[fault]
        if fault != "stale":
            record = None
    elif fault == "q-used":
        c["usedq"] = torch.tensor([8, 7], dtype=torch.int32, device="npu")
        record = None
    elif fault == "window-stale":
        c["options"]["window_size"] = (0, 0)
    else:
        word, value = {
            "version": (1, 5),
            "kind": (2, 5),
            "prefix": (-2, 100),
            "prefix-high": (-1, 1),
            "aic-endpoint": (430, 99),
            "fd-count": (508, 99),
            "fd-split": (564, 99),
            "fd-vector": (668, 99),
            "window": (427, 0),
            "snapshot": (417, 1056),
            "empty": (1, 5),
        }[fault]
        record[word] = value
    started = time.monotonic()
    stage = "launch"
    try:
        with forbid_host_readback():
            result = invoke(c, record)
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
