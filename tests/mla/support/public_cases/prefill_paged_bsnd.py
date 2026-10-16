"""Independent physical pages, logical reference gather and public PA calls."""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases import prefill_bsnd as bsnd


def make_case(
    dtype="fp16",
    shape=(2, 17, 8, 1057, 2),
    splits=1,
    mask="none",
    strided=False,
    lengths=None,
    page=112,
):
    b, sq, h, sk, hk = shape
    cap = (sk + page - 1) // page * page
    lengths = [sk] * b if lengths is None else lengths
    c = bsnd.make_case(dtype, (b, sq, h, cap, hk), splits, mask, strided, lengths)
    count = cap // page
    physical = b * count + 3 if count else 0
    ids = [[(3 * i + 2 * bb + 1) % physical for i in range(count)] for bb in range(b)]
    table = torch.tensor(ids, dtype=torch.int32).reshape(b, count)
    gen = torch.Generator().manual_seed(9407)
    c["physical"] = {}
    for name, dim in (("k", 192), ("v", 128)):
        raw = torch.randn((physical, page, hk, dim), generator=gen).to(c["raw"]["q"].dtype)
        logical = raw[table.long()].reshape(b, cap, hk, dim)
        c["raw"][name] = logical
        c["physical"][name] = raw
        value = raw.npu()
        if strided:
            storage = value.new_empty((physical * 2 + 1, page * 2 + 1, hk * 2 + 1, dim + 16))
            view = storage[
                1 : 1 + 2 * physical : 2, 1 : 1 + 2 * page : 2, 1 : 1 + 2 * hk : 2, 8 : 8 + dim
            ]
            view.copy_(value)
            value = view
        c["t"][name] = value
    # Poison only entries outside the effective range. Q-empty batches still
    # have all their live KV entries validated by the device.
    for bb, length in enumerate(lengths):
        table[bb, (length + page - 1) // page :] = -1
    c["raw_pages"] = table
    value = table.npu()
    if strided:
        storage = torch.full((b + 1, count + 3), -77, dtype=torch.int32, device="npu")
        view = storage[1:, 1 : count + 1]
        view.copy_(value)
        value = view
    c.update(page_table=value, page=page, metadata_prefix=24)
    return c


def metadata(c):
    b, sq, h, cap, hk = c["shape"]
    return api.get_scheduler_metadata(
        b,
        sq,
        cap,
        h,
        hk,
        192,
        c["lengths"],
        headdim_v=128,
        qkv_dtype=c["t"]["q"].dtype,
        page_size=c["page"],
        **{k: v for k, v in c["options"].items() if k != "softmax_scale"},
    )


def invoke(c, record=None):
    return api.flash_attn_with_kvcache(
        c["t"]["q"],
        c["t"]["k"],
        c["t"]["v"],
        page_table=c["page_table"],
        cache_seqlens=c["lengths"],
        scheduler_metadata=record,
        return_softmax_lse=True,
        **c["options"],
    )


def compare(c, result, record):
    if c["shape"][3] == 0:
        assert (result[0] == 0).all() and result[1].isneginf().all()
        if c["options"]["num_splits"] != 1:
            assert (result[2] == 0).all() and result[3].isneginf().all()
        else:
            assert all(t.numel() == 0 for t in result[2:])
        return {}
    return bsnd.compare(c, result, record)
