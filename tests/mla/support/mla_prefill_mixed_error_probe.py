"""One release v29 rejection in a fresh device process."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.prefill_mixed import make_case, metadata, invoke
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback


def run(dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = make_case(
        dtype,
        q=[0, 0] if fault == "empty" else [7, 3],
        k=[8193, 513] if fault == "fd-start" else [1057, 31],
        splits=0,
        mask="none" if fault == "fd-start" else "window",
        # The stale-prefix case keeps used lengths valid after moving the
        # allocation boundary, so it specifically checks snapshot binding.
        used_q=[0, 0] if fault == "empty" else [6, 3] if fault == "prefix-stale" else [7, 3],
        used_k=[8193, 513] if fault == "fd-start" else [1057, 31],
    )
    record = metadata(c)
    invoke(c, record)
    torch.npu.synchronize()
    if fault == "fd-start":
        assert record[205].item() > 0  # core 1 owns a later fragment of this M.
    if fault.startswith("prefix-"):
        index, value = {
            "prefix-start": (0, 1),
            "prefix-decreasing": (1, -1),
            "prefix-end": (2, 9),
            "prefix-max": (1, 8),
            "prefix-stale": (1, 6),
        }[fault]
        c["lengths"]["cu_seqlens_q"][index] = value
        if fault != "prefix-stale":
            record = None
    elif fault.startswith("used-") and fault != "used-snapshot":
        c["lengths"]["seqused_k"][0] = {"used-over": 1058, "used-negative": -1, "used-stale": 513}[
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
            "version": (1, 27),
            "reserved": (22, 1),
            "total": (20, 9),
            "allocation": (32, 6),
            "allocation-high": (25, 1),
            "capacity": (28, 6),
            "used-snapshot": (30, 6),
            "partition-prefix": (-2, 99),
            "aic-endpoint": (44, 99),
            "fd-count": (122, 99),
            "fd-split": (178, 99),
            "fd-start": (205, 99),
            "fd-vector": (282, 99),
            "empty": (1, 27),
        }[fault]
        record[word] = value
    started, stage = time.monotonic(), "launch"
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
