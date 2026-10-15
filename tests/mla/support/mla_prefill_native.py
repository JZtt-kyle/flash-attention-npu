"""Small pinned-package preflight for the 192/128 generic nonquant path."""

import argparse
import json
from pathlib import Path

from tests.mla.support.native_package import native_vendor
from tests.mla.support.mla_native_dispatch import parse_dispatch


def run(dtype, output):
    import torch
    import torch_npu  # noqa: F401

    from tests.mla.support.mla_cann_baseline import CannCall, Case, make_inputs
    from tests.mla.support.mla_validation_build import runtime

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime.json").write_text(json.dumps(runtime(), indent=2) + "\n")
    records = []
    for index, (split, causal) in enumerate(((False, False), (True, False), (False, True))):
        spec = Case("prefill-preflight", "prefill", 2, 17, 31, 8, 2, split_rope=split)
        inputs = make_inputs(spec, {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype])
        inputs = {k: None if v is None else v.npu() for k, v in inputs.items()}
        call = CannCall(inputs, split_rope=split, causal=causal)
        print(f"MLA_STRIDE_NATIVE_BEGIN {dtype} {index}", flush=True)
        out, lse = call()
        torch.npu.synchronize()
        print(f"MLA_STRIDE_NATIVE_END {dtype} {index}", flush=True)
        assert torch.isfinite(out).all() and torch.isfinite(lse).all()
        records.append(dict(split_rope=split, causal=causal, call=call.description()))
    (output / "inputs.json").write_text(json.dumps(records, indent=2) + "\n")


def check(log, dtype, output):
    result = parse_dispatch(log, dtype, 3, native_vendor())
    if not all(row["native_template"] in (3, 4) for row in result["cases"]):
        raise RuntimeError("192/128 must execute the pinned generic nonquant kernel")
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.log:
        check(args.log, args.dtype, args.output)
    else:
        run(args.dtype, args.output)
