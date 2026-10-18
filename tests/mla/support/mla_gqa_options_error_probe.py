"""One option/allocation/PA mismatch in an isolated release device process."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.gqa_options import make_case, metadata, invoke
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback


def run(layout, representation, dtype, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = make_case(layout, representation, dtype, splits=0, mask="window", pack=False, margin=1)
    record = None if layout == "tnd" else metadata(c)
    invoke(c, record)
    torch.npu.synchronize()
    if fault == "margin":
        c["options"]["sm_margin"] = 2
    elif fault == "packed-field":
        record[14] = 1
    elif fault == "version":
        record[1] = 46
    elif fault == "stale":
        (c["lengths"]["seqused_k"] if isinstance(c["lengths"], dict) else c["lengths"])[0] -= 1
    elif fault == "prefix":
        c["lengths"]["cu_seqlens_q"][1] = -1
        record = None
    elif fault in ("page", "page-zero-q"):
        c["page_table"][0, 0] = c["t"]["k"].shape[0]
        if fault == "page-zero-q":
            c["lengths"]["seqused_q"] = torch.zeros_like(c["lengths"]["seqused_k"])
            record = None
    elif fault == "softcap":
        c["options"]["softcap"] = 15
    elif fault == "window":
        c["options"]["window_size"] = (0, 0)
    else:
        raise ValueError(fault)
    stage = "launch"
    try:
        with forbid_host_readback():
            result = invoke(c, record, entry="op")
        stage = "event"
        event = torch.npu.Event()
        event.record()
        event.synchronize()
    except RuntimeError as error:
        path = Path(api.flash_attn_npu_3.__file__)
        (Path(directory) / "runtime-error.json").write_text(
            json.dumps(
                dict(
                    stage=stage,
                    fault=fault,
                    error=str(error),
                    extension=str(path),
                    extension_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
        os._exit(0)
    raise AssertionError(f"invalid GQA record completed: {fault}: {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
