"""One release v27 rejection in a fresh device process."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.prefill_bsnd import make_case, metadata, invoke
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.public_cases.contract import _forward


def run(dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = make_case(
        dtype, shape=(2, 0 if fault == "empty" else 7, 6, 1057, 2), splits=0, mask="window"
    )
    record = metadata(c)
    invoke(c, record)
    torch.npu.synchronize()
    if fault in ("stale", "negative", "over"):
        c["lengths"][-1] = {"stale": 3, "negative": -1, "over": 1058}[fault]
        if fault != "stale":
            record = None
    elif fault == "none-stale":
        c["lengths"][-1] = 3
        record = metadata(c)
        c["lengths"] = None
    elif fault == "window-stale":
        c["options"]["window_size"] = (0, 0)
    elif fault == "split-stale":
        c["options"]["num_splits"] = 2
    elif fault == "q-used":
        c["usedq"] = torch.tensor([8, 7], dtype=torch.int32, device="npu")
    else:
        word, value = {
            "version": (1, 18),
            "kind": (2, 18),
            "reserved": (19, 1),
            "prefix": (-2, 99),
            "prefix-high": (-1, 1),
            "aic-endpoint": (28, 99),
            "fd-count": (106, 99),
            "fd-split": (162, 99),
            "fd-vector": (266, 99),
            "snapshot": (21, 1056),
            "kv-heads": (10, 1),
            "empty": (1, 18),
        }[fault]
        record[word] = value
    started, stage = time.monotonic(), "launch"
    try:
        with forbid_host_readback():
            if fault == "q-used":
                result = _forward(
                    "cpp",
                    dict(**c["t"], num_splits=0, seqused_k=c["lengths"], seqused_q=c["usedq"]),
                )
            else:
                result = invoke(c, record)
        stage = "event"
        event = torch.npu.Event()
        event.record()
        event.synchronize()
    except RuntimeError as error:
        p = Path(api.flash_attn_npu_3.__file__)
        (Path(directory) / "runtime-error.json").write_text(
            json.dumps(
                dict(
                    fault=fault,
                    stage=stage,
                    error=str(error),
                    seconds=time.monotonic() - started,
                    extension=str(p),
                    extension_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
        os._exit(0)
    raise AssertionError(f"invalid prefill completed: {fault} {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
