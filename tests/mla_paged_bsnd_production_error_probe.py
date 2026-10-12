"""One real release Trap through a production BSND/TND PA forward per process."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla_metadata_error_probe import forbid_host_readback


def run(kind, fault, directory, entry="internal"):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[kind]
    if entry.startswith("tnd-"):
        from tests.test_flash_attn_npu_v3_mla_paged_tnd_production import (
            small_inputs,
            forward,
            generate,
        )
    else:
        from tests.test_flash_attn_npu_v3_mla_paged_bsnd_production import (
            small_inputs,
            forward,
            generate,
        )
    if entry in ("public", "tnd-public"):
        from tests.test_flash_attn_npu_v3_mla_paged_public import public

        def call(args, metadata=None):
            return public(dict(args, scheduler_metadata=metadata))
    else:
        call = forward
    args = small_inputs(
        dtype,
        sq=0 if fault == "empty_q_page" else 3,
        pool=0 if fault == "empty_pool" else 7,
        used=fault in ("bad_q", "zero_k_bad_q", "unused_q_page"),
    )
    if fault == "empty_pool":
        args["cache_seqlens"].zero_()
    with forbid_host_readback():
        call(args)
    torch.npu.synchronize()
    record = None
    if fault in ("page_negative", "page_over", "unused_q_page", "empty_q_page"):
        # The last effective page is checked even when the sample has no Q.
        args["page_table"][0, 1] = 7 if fault == "page_over" else -1
        if fault == "unused_q_page":
            args["seqused_q"].zero_()
    elif fault in ("stale_k", "stale_prefix"):
        record = generate(args)
        torch.npu.synchronize()
        if fault == "stale_k":
            args["cache_seqlens"][0] = 17
        else:
            args["cu_seqlens_q"][1] = 2
    elif fault.startswith("prefix_") or fault == "zero_k_prefix":
        prefix = args["cu_seqlens_q"]
        if fault == "prefix_start":
            prefix[0] = 1
        elif fault == "prefix_end":
            prefix[-1] = 4
        elif fault == "prefix_descending":
            prefix[2] = 2
        elif fault == "prefix_max":
            prefix[1:3] = 4
        else:
            args["cache_seqlens"].zero_()
            prefix[0] = -1
    elif fault in ("bad_q", "zero_k_bad_q"):
        args["seqused_q"][0] = 4
        if fault == "zero_k_bad_q":
            args["cache_seqlens"].zero_()
    elif fault in ("bad_k", "empty_pool"):
        args["cache_seqlens"][0] = -1 if fault == "bad_k" else 1
    else:
        if fault == "empty_payload":
            args["cache_seqlens"].zero_()
        record = generate(args)
        torch.npu.synchronize()
        if fault == "magic":
            record[0] = 0
        elif fault == "dtype":
            record[4] = 2 if kind == "fp16" else 1
        elif fault in ("v1", "v2", "v3", "v4", "v5"):
            record[1] = int(fault[1])
        elif fault == "page_size":
            record[406] = 128
        elif fault == "capacity":
            record[407] = 224
        elif fault == "empty_payload":
            record[385] = 16
        elif fault == "bsnd_record":
            record[401] = 0
        else:
            raise ValueError(fault)
    start = time.monotonic()
    try:
        with forbid_host_readback():
            result = call(args, record)
        torch.npu.current_stream().synchronize()
    except RuntimeError as error:
        extension = Path(api.flash_attn_npu_3.__file__)
        report = dict(
            fault=fault,
            entry=entry,
            dtype=kind,
            error=str(error),
            elapsed_seconds=time.monotonic() - start,
            extension=str(extension),
            extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        )
        Path(directory, "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
        assert "trap error" in str(error), str(error)
        os._exit(0)
    raise AssertionError(f"Invalid PA input completed without a Trap: {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
