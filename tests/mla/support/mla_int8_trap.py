"""One invalid-scale device execution per subprocess; never substitute CANN."""

import json
import sys

import torch

from tests.mla.support.mla_int8_cases import SOURCE_CASES, make_inputs
from tests.mla.support.mla_int8_component import ComponentCall


def main(field):
    torch.npu.set_device(0)
    spec = SOURCE_CASES[3]
    inputs = make_inputs(spec)
    call = ComponentCall(spec, inputs)
    if field in ("qs", "ks", "vs"):
        call.native.inputs[field].fill_(float("inf") if field == "ks" else float("nan"))
    elif field == "page":
        call.table[0, 0] = call.layout.pool
    else:
        raise ValueError(field)
    try:
        call()
        torch.npu.synchronize()
    except RuntimeError as error:
        message = str(error)
        if not any(
            token in message.lower()
            for token in ("507015", "507035", "vector core", "aicore exception", "aicore error")
        ):
            raise
        print(json.dumps(dict(rejected=field, error=message)), flush=True)
        return
    raise AssertionError("Invalid INT8 scale/page executed successfully")


if __name__ == "__main__":
    main(sys.argv[1])
