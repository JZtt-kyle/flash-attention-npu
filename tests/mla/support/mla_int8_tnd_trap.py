"""One public INT8 release Trap per fresh interpreter/device context."""

import json
from pathlib import Path
import resource
import sys

import torch

from tests.mla.support.public_cases.int8_tnd import make_case, metadata, invoke
from tests.mla.support.public_cases.graph import capture
from tests.mla.support.mla_validation_build import sha256
from flash_attn_npu_3 import flash_attn_npu_interface as api


def run(fault, output):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    c = (
        make_case(("page-grid", "TND", (1, 1), 8, (1, 8193), 0))
        if fault in ("page-late", "page-empty-query")
        else make_case()
    )
    record = metadata(c)
    invoke(c, record)
    torch.npu.synchronize()
    graph = (
        capture(lambda: invoke(c, record))[0] if fault in ("graph-stale", "graph-prefix") else None
    )
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
    elif fault in ("page-late", "page-empty-query"):
        if fault == "page-empty-query":
            for name in ("q", "qv", "qs"):
                c["tensors"][name] = c["tensors"][name][:0]
            c["prefix"].zero_()
            c["spec"] = (*c["spec"][:2], (0, 0), *c["spec"][3:])
            record = metadata(c)
            invoke(c, record)
            torch.npu.synchronize()
        # Beyond the first vector-grid stride, in a later batch. For empty Q
        # only the public precheck can observe this invalid used page.
        c["tensors"]["table"][1, 64] = c["tensors"]["k"].shape[0]
    elif fault == "metadata-version":
        record[1] = 6
    elif fault == "metadata-tiling":
        record[20] += 1
    elif fault == "metadata-tail":
        record[-1] += 1
    elif fault in ("metadata-length", "graph-stale"):
        c["lengths"][0] -= 1
    elif fault.startswith("prefix-"):
        index, value = {
            "prefix-start": (0, 1),
            "prefix-negative": (1, -1),
            "prefix-order": (2, 1),
            "prefix-tail": (3, 6),
            "prefix-max": (1, 5),
        }[fault]
        c["prefix"][index] = value
    elif fault in ("metadata-prefix", "graph-prefix"):
        c["prefix"][1] = 3
    elif fault == "metadata-total":
        record[14] += 1
    elif fault == "bsnd-record":
        record[1] = 48
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
