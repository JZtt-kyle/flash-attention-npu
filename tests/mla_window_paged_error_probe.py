"""Isolated v15 device rejection; failure must happen after submitting forward."""

import os
import sys
import torch
from tests.test_flash_attn_npu_v3_mla_window_paged_public import production, metadata, call


def main():
    torch.npu.set_device(0)
    args = production.small_inputs(
        getattr(torch, sys.argv[1]), sq=0 if sys.argv[2] == "zero-query-page" else 7
    )
    mode = sys.argv[2]
    record = metadata(args)
    if mode in ("v6", "v10", "v11"):
        record[1:3] = int(mode[1:])
    elif mode in ("window-kind", "window-left", "window-right"):
        record[{"window-kind": -3, "window-left": -2, "window-right": -1}[mode]] += 1
    elif mode == "stale-window":
        args["window_size"] = (3, 1)
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
            [{"zero-used-page": 0, "negative-used": -1, "overflow-used": 8}[mode], 0],
            dtype=torch.int32,
            device="npu",
        )
    else:
        args["scheduler_metadata"] = record
    torch.npu.synchronize()
    os.write(1, b"MLA_PAGED_WINDOW_LAUNCHED\n")
    call(args, "op")
    torch.npu.synchronize()
    os.write(1, b"MLA_PAGED_WINDOW_COMPLETED\n")


if __name__ == "__main__":
    main()
