"""One invalid v4 snapshot/current length Trap per fresh production process."""

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
from tests.mla.e2e.mixed.test_flash_attn_npu_v3_mla_mixed_public import inputs, metadata, public


def run(dtype, fault, directory, layout="contiguous"):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    kind = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    args = inputs(kind)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        public(args)
        stream.synchronize()
        if fault == "empty_b":
            args = inputs(kind, [], [], 64)
        elif fault == "empty_q":
            args = inputs(kind, [0, 0], [1, 3], 64)
        elif fault == "empty_k":
            args = inputs(kind, [3, 0, 1], [0, 0, 0], 0)
        if layout == "strided":
            from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_stride import strides_for

            for name in ("q", "qv", "k", "v"):
                t = args[name]
                shape = (1, *t.shape) if t.ndim == 3 else t.shape
                strides = strides_for(shape, "independent", name)
                span = 0 if 0 in shape else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
                base = torch.zeros(span + 32, dtype=kind, device="npu")
                view = base.as_strided(shape, strides, 7)
                args[name] = view[0] if t.ndim == 3 else view
        if fault.startswith("stale") or fault in (
            "magic",
            "version",
            "dtype",
            "shape",
            "capacity",
            "kv_high",
            "endpoint",
            "workspace",
        ):
            args["scheduler_metadata"] = metadata(args)
        if fault in ("q_start", "empty_b"):
            args["cu_seqlens_q"][0] = 1
        elif fault == "q_decrease":
            args["cu_seqlens_q"][2] = 2
        elif fault in ("q_tail", "empty_q"):
            args["cu_seqlens_q"][-1] += 1
        elif fault == "q_max":
            args["max_seqlen_q"] = 0
        elif fault == "k_negative":
            args["cache_seqlens"][0] = -1
        elif fault in ("k_over", "empty_k"):
            args["cache_seqlens"][0] = args["k"].shape[1] + 1
        elif fault == "stale_q":
            args["cu_seqlens_q"][1] -= 1
        elif fault == "stale_k":
            args["cache_seqlens"][0] -= 1
        elif fault == "stale_none":
            args["cache_seqlens"] = None
        elif fault == "stale_integer":
            args["cache_seqlens"] = 64
        else:
            word, value = {
                "magic": (0, 0),
                "version": (1, 3),
                "dtype": (4, 3),
                "shape": (401, 3),
                "capacity": (403, 63),
                "kv_high": (408, 1),
                "endpoint": (23, 2147483647),
                "workspace": (368, -1),
            }[fault]
            args["scheduler_metadata"][word] = value
        start = time.monotonic()
        try:
            with forbid_host_readback():
                result = public(args)
            done = torch.npu.Event()
            done.record(stream)
            done.synchronize()
        except RuntimeError as error:
            report = dict(
                input_layout=layout,
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
            # A trapped context cannot service allocator/event cleanup. Like
            # the other fault probes, finish this isolated process only after
            # saving and checking the real device error; the parent checks it.
            os._exit(0)
        raise AssertionError(f"Malformed mixed metadata completed without a Trap: {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
