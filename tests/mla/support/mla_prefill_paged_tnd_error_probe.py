"""One v31 live-page/length/record rejection in an isolated NPU process."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.prefill_paged_tnd import make_case, metadata, invoke
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback


def run(dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = make_case(
        dtype,
        (2, 0 if fault == "empty-q-page" else 7, 8, 1057, 2),
        0,
        "window",
        True,
        [1057, 31],
        page=112,
    )
    record = metadata(c)
    invoke(c, record)
    torch.npu.synchronize()
    base = 40
    if fault in ("page-negative", "empty-q-page"):
        c["page_table"][0, 0] = -1
    elif fault == "page-over":
        c["page_table"][1, 0] = c["t"]["k"].shape[0]
    elif fault == "page-after-zero-q":
        c["lengths"]["seqused_q"] = torch.zeros(2, dtype=torch.int32, device="npu")
        record = None
        c["page_table"][1, 0] = -1
    elif fault.startswith("prefix-"):
        index, value = {
            "prefix-first": (0, 1),
            "prefix-negative": (1, -1),
            "prefix-end": (2, 13),
            "prefix-over": (1, 15),
            "prefix-stale": (1, 6),
        }[fault]
        c["lengths"]["cu_seqlens_q"][index] = value
        if fault != "prefix-stale":
            record = None
    elif fault in ("q-used-over", "q-used-negative"):
        c["lengths"]["seqused_q"] = torch.tensor(
            [8 if fault == "q-used-over" else -1, 7], dtype=torch.int32, device="npu"
        )
        record = None
    elif fault == "physical-empty":
        c["t"]["k"] = c["t"]["k"][:0]
        c["t"]["v"] = c["t"]["v"][:0]
    elif fault.startswith("used-") and fault != "used-snapshot":
        c["lengths"]["seqused_k"][0] = {"used-over": 1121, "used-negative": -1, "used-stale": 513}[
            fault
        ]
        if fault != "used-stale":
            record = None
    elif fault == "split-stale":
        c["options"]["num_splits"] = 2
    elif fault == "window-stale":
        c["options"]["window_size"] = (0, 0)
    else:
        word, value = {
            "version": (1, 29),
            "reserved": (22, 1),
            "page-size": (21, 16),
            "tile": (14, 33),
            "q-start": (32, 6),
            "q-capacity": (28, 6),
            "q-total": (20, 13),
            "used-snapshot": (31, 513),
            "partition-prefix": (-2, 99),
            "aic-endpoint": (base + 4, 99),
            "fd-count": (base + 82, 99),
            "fd-split": (base + 138, 99),
            "fd-vector": (base + 242, 99),
        }[fault]
        record[word] = value
    start, stage = time.monotonic(), "launch"
    try:
        with forbid_host_readback():
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
                    seconds=time.monotonic() - start,
                    extension=str(p),
                    extension_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
        os._exit(0)
    raise AssertionError(f"invalid PA prefill completed: {fault} {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
