"""One malformed production varlen used input per fresh process."""

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
from tests.test_flash_attn_npu_v3_mla_varlen_used import inputs


def run(kind, fault, directory, layout="contiguous"):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[kind]
    args = inputs(
        dtype,
        qs=(0, 0) if fault == "empty_q" else (7, 3),
        ks=(0, 0) if fault == "empty_k" else (33, 17),
    )
    if layout == "strided":
        for name in ("q", "qv", "k", "v"):
            original = args[name]
            storage = torch.empty(
                (2 * original.shape[0] + 1, *original.shape[1:]),
                dtype=original.dtype,
                device=original.device,
            )
            args[name] = storage[1 : 1 + 2 * original.shape[0] : 2]
            args[name].copy_(original)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        api.flash_attn_varlen_func(**args, return_attn_probs=True)
        stream.synchronize()
        if fault == "q_start":
            args["cu_seqlens_q"][0] = 1
        elif fault == "k_tail":
            args["cu_seqlens_k"][-1] = 49
        elif fault == "q_decrease":
            args["cu_seqlens_q"][1] = 11
        elif fault == "k_max":
            args["max_seqlen_k"] = 32
        elif fault in ("q_negative", "q_over", "zero_k_bad_q", "empty_q"):
            args["seqused_q"][0] = (
                -1 if fault == "q_negative" else (0 if fault == "empty_q" else 7) + 1
            )
            if fault == "zero_k_bad_q":
                args["seqused_k"].zero_()
        else:
            args["seqused_k"][0] = (
                -1 if fault == "k_negative" else (0 if fault == "empty_k" else 33) + 1
            )
            if fault == "zero_q_bad_k":
                args["seqused_q"].zero_()
        start = time.monotonic()
        try:
            with forbid_host_readback():
                result = api.flash_attn_varlen_func(**args, return_attn_probs=True)
            stream.synchronize()
        except RuntimeError as error:
            extension = Path(api.flash_attn_npu_3.__file__)
            report = dict(
                fault=fault,
                input_layout=layout,
                dtype=kind,
                error=str(error),
                elapsed_seconds=time.monotonic() - start,
                extension=str(extension),
                extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
            )
            Path(directory, "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
            assert "trap error" in str(error), str(error)
            # The trapped context cannot service allocator/event cleanup.
            os._exit(0)
        raise AssertionError(f"Invalid used lengths completed without a Trap: {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
