"""One feature/length mismatch in an isolated release NPU process."""

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.softcap_paged_bsnd import make_case, metadata, invoke
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback


def run(representation, dtype, splits, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = make_case(representation, dtype, splits=int(splits), mask="window", lengths=[65, 3])
    record = metadata(c)
    invoke(c, record)
    torch.npu.synchronize()
    if fault == "old":
        record = metadata(c, enabled=False)
    elif fault == "reverse":
        c["options"]["softcap"] = 0
    elif fault == "stale":
        c["lengths"][-1] = 1
    elif fault == "negative":
        c["lengths"][-1] = -1
        record = None
    elif fault in ("page", "page-zero-q"):
        c["page_table"][0, 0] = c["t"]["k"].shape[0]
        if fault == "page-zero-q":
            c["usedq"] = torch.zeros_like(c["lengths"])
            record = None
    elif fault == "mask":
        c["options"]["window_size"] = (0, 0)
    else:
        record[1] = 26
    stage = "launch"
    try:
        with forbid_host_readback():
            result = invoke(c, record)
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
    raise AssertionError(f"invalid softcap record completed: {fault}: {result}")


if __name__ == "__main__":
    run(*sys.argv[1:])
