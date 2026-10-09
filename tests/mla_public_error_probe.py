"""One public MLA device rejection per process, with no enqueue-path D2H."""

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
from tests.test_flash_attn_npu_v3_mla_kvcache import cached, generate


def run(dtype_name, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    directory = Path(directory)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    device_name = torch.npu.get_device_name(0)
    assert "910B" in device_name
    empty, separator, base_fault = fault.partition(":")
    if separator:
        fault = base_fault
    else:
        empty = ""
    # The invalid final batch catches checks that accidentally inspect only
    # the first sample or only the first launch core's own work.
    heads = 171 if fault in ("m_base", "m_end") else 4
    sk = 16384 if fault.startswith("kv_") else 17
    batch = 0 if empty == "emptyb" else 21
    sq = 0 if empty == "emptyq" else 3
    if empty == "emptyk":
        sk = 0
    device = {
        n: torch.zeros(
            (batch, sq if n in ("q", "qv") else sk, heads if n in ("q", "qv") else 1, d),
            dtype=dtype,
            device="npu",
        )
        for n, d in (("q", 64), ("k", 64), ("v", 512), ("qv", 512))
    }
    lengths = torch.full((batch,), sk, dtype=torch.int32, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        record = generate(device, lengths)
        good = cached(device, record, lengths)
        torch.testing.assert_close(good[0].cpu(), torch.zeros(good[0].shape, dtype=dtype))
        expected_lse = (
            torch.full(good[1].shape, float("inf"))
            if sk == 0
            else torch.full(good[1].shape, float(sk)).log()
        )
        torch.testing.assert_close(good[1].cpu(), expected_lse)
        mode, _, detail = fault.partition("-")
        if mode in ("generated", "implicit", "reused"):
            value = dict(negative=-1, zero=0, short=16, over=18)[detail]
            lengths[-1] = value
            if mode == "generated":
                diagnostic = generate(device, lengths).cpu()
                assert diagnostic[5].item() == 2 and diagnostic[8:].count_nonzero().item() == 0
        elif fault == "omitted-lengths":
            lengths[-1] = sk - 1
            record = generate(device, lengths)
            lengths = None
        else:
            mutations = dict(
                version=(1, 1),
                kind=(2, 1),
                wordcount=(3, 401),
                snapshot=(401 + batch - 1, sk - 1),
                magic=(0, 0),
                dtype=(4, 3),
                shape=(8, 22),
                workspace=(368, -1),
                m_base=(15, 129),
                m_end=(49, 3),
                kv_base=(16, 1024),
                kv_end=(75, 1),
                kv_workspace=(367, 16384),
            )
            word, value = mutations[fault]
            record[word] = value
        stage = "launch"
        start = time.monotonic()
        try:
            with forbid_host_readback():
                if mode == "generated":
                    record = generate(device, lengths)
                output = cached(device, None if mode == "implicit" else record, lengths)
            stage = "event"
            done = torch.npu.Event()
            done.record(stream)
            done.synchronize()
        except RuntimeError as error:
            extension = Path(api.flash_attn_npu_3.__file__).resolve()
            report = dict(
                fault=fault,
                empty_capacity=empty,
                dtype=dtype_name,
                shape=list(device["q"].shape),
                sk=sk,
                pid=os.getpid(),
                device=device_name,
                visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                stage=stage,
                elapsed_seconds=time.monotonic() - start,
                error=str(error),
                extension=str(extension),
                extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
            )
            (directory / "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
            # Retain all storage until exit; allocator destructors may enqueue
            # work on the trapped context. The parent cleans our process group.
            os._exit(0)
        raise AssertionError(f"{fault}: invalid call silently completed: {output}")


if __name__ == "__main__":
    run(*sys.argv[1:])
