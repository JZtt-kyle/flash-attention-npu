"""One malformed-prefix Trap per fresh production varlen process."""

import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_contract import _inputs


def run(dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args = _inputs(layout="tnd", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[dtype])
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        api.flash_attn_varlen_func(**args)
        stream.synchronize()
        if fault == "empty_b":
            for n in ("q", "qv", "k", "v"):
                args[n] = args[n][:0]
            args["cu_seqlens_q"] = torch.ones(1, dtype=torch.int32, device="npu")
            args["cu_seqlens_k"] = torch.zeros(1, dtype=torch.int32, device="npu")
        elif fault.startswith("empty_"):
            query = fault == "empty_q"
            for n in ("q", "qv") if query else ("k", "v"):
                args[n] = args[n][:0]
            args["cu_seqlens_" + ("q" if query else "k")].zero_()
            args["cu_seqlens_" + ("q" if query else "k")][-1] = 1
        else:
            name, detail = fault.split("_")
            prefix = args["cu_seqlens_" + name]
            if detail == "max":
                args["max_seqlen_" + name] = 0
            elif detail == "start":
                prefix[0] = 1
            elif detail == "decrease":
                prefix[-1] = -1
            else:
                prefix[-1] -= 1
        start = time.monotonic()
        try:
            with forbid_host_readback():
                result = api.flash_attn_varlen_func(**args, return_attn_probs=True)
            done = torch.npu.Event()
            done.record(stream)
            done.synchronize()
        except RuntimeError as error:
            report = dict(
                fault=fault,
                dtype=dtype,
                elapsed_seconds=time.monotonic() - start,
                error=str(error),
                extension=api.flash_attn_npu_3.__file__,
            )
            report["extension_sha256"] = hashlib.sha256(
                Path(report["extension"]).read_bytes()
            ).hexdigest()
            Path(directory, "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
            assert "trap error" in str(error), str(error)
            return
        raise AssertionError(f"Malformed prefixes completed without a Trap: {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
