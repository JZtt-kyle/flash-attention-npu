"""Isolated v11 TND device rejection; failure must happen after submitting forward."""

import os
import sys
import torch
from tests.test_flash_attn_npu_v3_mla_causal_paged_tnd_public import production, metadata, call


def main():
    torch.npu.set_device(0)
    args = production.small_inputs(
        getattr(torch, sys.argv[1]), sq=0 if sys.argv[2] == "zero-query-page" else 7
    )
    mode = sys.argv[2]
    record = metadata(args)
    if mode == "v6":
        record = production.generate(args)
    elif mode == "v10":
        record[1:3].fill_(10)
    elif mode in (
        "stale-prefix",
        "negative-prefix",
        "nonzero-prefix",
        "overflow-prefix",
        "decreasing-prefix",
    ):
        values = {
            "stale-prefix": [0, 2, 7, 9],
            "negative-prefix": [0, -1, 7, 9],
            "nonzero-prefix": [1, 7, 7, 9],
            "overflow-prefix": [0, 7, 7, 10],
            "decreasing-prefix": [0, 7, 6, 9],
        }[mode]
        args["cu_seqlens_q"].copy_(torch.tensor(values, dtype=torch.int32))
    elif mode == "zero-middle-page":
        args["page_table"][1, 0] = -1
    elif mode == "header":
        record[1] = 99
    elif mode == "endpoint":
        record[23] = 999
    elif mode in ("stale-kv", "negative-kv", "overflow-kv"):
        args["cache_seqlens"][0] = {"stale-kv": 17, "negative-kv": -1, "overflow-kv": 449}[mode]
    elif mode in ("negative-page", "overflow-page", "zero-query-page", "zero-used-page"):
        args["page_table"][0, 1] = 7 if mode == "overflow-page" else -1
    if mode in ("zero-used-page", "negative-used", "overflow-used"):
        args["seqused_q"] = torch.tensor(
            [{"zero-used-page": 0, "negative-used": -1, "overflow-used": 8}[mode], 0, 0],
            dtype=torch.int32,
            device="npu",
        )
    else:
        args["scheduler_metadata"] = record
    torch.npu.synchronize()
    os.write(1, b"MLA_PAGED_CAUSAL_LAUNCHED\n")
    if mode == "v11-unmasked":
        torch.ops.flash_attn_npu_3._mla_paged_tnd_forward(**args)
    else:
        call(args, "op")
    torch.npu.synchronize()
    os.write(1, b"MLA_PAGED_CAUSAL_COMPLETED\n")


if __name__ == "__main__":
    main()
