"""One public INT8 release Trap per fresh interpreter/device context."""

import json
from pathlib import Path
import resource
import sys

import torch

from tests.mla.support.public_cases.int8_bsnd import make_case, metadata, invoke
from tests.mla.support.public_cases.graph import capture
from tests.mla.support.mla_validation_build import sha256
from flash_attn_npu_3 import flash_attn_npu_interface as api


def run(fault, output):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = make_case()
    record = metadata(c)
    invoke(c, record)
    torch.npu.synchronize()
    graph = capture(lambda: invoke(c, record))[0] if fault == "graph-stale" else None
    if fault in ("qs", "ks", "vs"):
        c["tensors"][fault].fill_(float("nan"))
    elif fault == "length-negative":
        c["lengths"][0] = -1
    elif fault == "length-capacity":
        c["lengths"][0] = c["tensors"]["table"].shape[1] * 128 + 1
    elif fault == "page-negative":
        c["tensors"]["table"][0, 0] = -1
    elif fault == "page-pool":
        c["tensors"]["table"][0, 0] = c["tensors"]["k"].shape[0]
    elif fault == "metadata-version":
        record[1] = 6
    elif fault == "metadata-tiling":
        record[20] += 1
    elif fault == "metadata-tail":
        record[-1] += 1
    elif fault in ("metadata-length", "graph-stale"):
        c["lengths"][0] -= 1
    else:
        raise ValueError(fault)
    try:
        graph.replay() if graph else invoke(c, record)
        torch.npu.synchronize()
    except RuntimeError as error:
        message = str(error)
        # Graph replay can wrap the vector Trap in model error 507011. Require
        # the device Trap diagnosis, as the other public Trap probes do.
        assert "trap error" in message.lower(), message
        extension = Path(api.flash_attn_npu_3.__file__)
        Path(output).write_text(
            json.dumps(
                dict(
                    fault=fault,
                    rejected=True,
                    error=message,
                    extension=str(extension),
                    sha256=sha256(extension),
                ),
                indent=2,
            )
            + "\n"
        )
        return
    raise AssertionError("Invalid public INT8 values completed")


if __name__ == "__main__":
    run(*sys.argv[1:])
