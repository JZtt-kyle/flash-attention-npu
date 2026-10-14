"""One malformed internal mixed Q-used input per fresh process."""

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
from tests.mla.e2e.mixed.test_flash_attn_npu_v3_mla_mixed_used import inputs


def run(kind, fault, directory, layout="contiguous"):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[kind]
    args = inputs(
        dtype,
        qs=()
        if fault == "empty_batch_prefix"
        else (0, 0)
        if fault.startswith("empty_q")
        else (7, 3),
        capacity=0 if fault.startswith("empty_k") else 33,
    )
    if layout == "strided":
        from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_stride import strides_for

        for name in ("q", "qv", "k", "v"):
            t = args[name]
            shape = (1, *t.shape) if t.ndim == 3 else t.shape
            strides = strides_for(shape, "independent", name)
            span = 0 if 0 in shape else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
            base = torch.zeros(span + 32, dtype=dtype, device="npu")
            view = base.as_strided(shape, strides, 7)
            args[name] = view[0] if t.ndim == 3 else view
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        api._flash_attn_forward(**args)
        stream.synchronize()
        if fault in ("q_start", "empty_batch_prefix"):
            args["cu_seqlens_q"][0] = 1
        elif fault == "q_tail":
            args["cu_seqlens_q"][-1] = 9
        elif fault == "q_decrease":
            args["cu_seqlens_q"][1] = 11
        elif fault == "q_max":
            args["max_seqlen_q"] = 6
        elif fault in ("q_negative", "q_over", "zero_k_bad_q", "empty_q", "empty_k_bad_q"):
            args["seqused_q"][0] = -1 if fault == "q_negative" else (1 if fault == "empty_q" else 8)
            if fault == "zero_k_bad_q":
                args["seqused_k"].zero_()
        else:
            args["seqused_k"][0] = (
                -1 if fault == "k_negative" else (1 if fault == "empty_k" else 34)
            )
            if fault == "zero_q_bad_k":
                args["seqused_q"].zero_()
        start = time.monotonic()
        try:
            with forbid_host_readback():
                result = api._flash_attn_forward(**args)
            stream.synchronize()
        except RuntimeError as error:
            extension = Path(api.flash_attn_npu_3.__file__)
            report = dict(
                input_layout=layout,
                fault=fault,
                dtype=kind,
                error=str(error),
                elapsed_seconds=time.monotonic() - start,
                extension=str(extension),
                extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
            )
            Path(directory, "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
            assert "trap error" in str(error), str(error)
            # A trapped context cannot service allocator/event cleanup.
            os._exit(0)
        raise AssertionError(f"Invalid mixed lengths completed without a Trap: {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
