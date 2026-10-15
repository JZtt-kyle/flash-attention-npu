"""Shared helpers extracted from tests/mla/e2e/auto_split_tnd/test_public.py.

Native automatic TND: public dispatch, allocation addresses and real FP32 slots.
"""

import copy
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.support.public_cases.window_tnd_public import prepare as prepare_window, effective
from tests.mla.support.public_cases.contract import _forward
from tests.mla.support.mla_auto_split_bsnd_reference import references
from tests.mla.support.public_cases.auto_split_bsnd_public import geometry


def prepare(dtype, s):
    case = prepare_window(dtype, copy.deepcopy(s))
    case["args"]["num_splits"] = s["splits"]
    return case


def op_args(args):
    options = dict(args)
    options["window_left"], options["window_right"] = options.pop("window_size")
    return options


def call(args, entry="public", record=None):
    if entry == "public":
        assert record is None
        return api.flash_attn_varlen_func(**args, return_attn_probs=True)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_split_tnd_forward(
            **op_args(args), scheduler_metadata=record
        )
    options = dict(args, scheduler_metadata=record)
    options["window_size_left"], options["window_size_right"] = options.pop("window_size")
    return _forward(entry, options)


def metadata(c):
    options = op_args(c["args"])
    options.pop("softmax_scale")
    return torch.ops.flash_attn_npu_3._mla_split_tnd_metadata(**options)


def reference(c, tile=None):
    a, s = c["args"], c["spec"]
    # Only tile choice comes from metadata; mask, allocation and used geometry
    # are independently reconstructed from original caller data.
    if tile is None:
        md = metadata(c).cpu()
        tile = geometry(md, len(s["q"]))
    m, n, ends = tile
    padded = {}
    for name, t in c["raw"].items():
        query = name in ("q", "qv")
        lengths = s["q"] if query else s["kv"]
        cap = a["max_seqlen_q"] if query else a["max_seqlen_k"]
        p = torch.zeros((len(lengths), cap, *t.shape[1:]), dtype=t.dtype)
        offset = 0
        for b, length in enumerate(lengths):
            p[b, :length] = t[offset : offset + length]
            offset += length
        padded[name] = p
    refs = references(
        padded,
        *effective(s),
        ends,
        m or 256,
        n or 512,
        s.get("causal", False),
        s["window"],
        s.get("scale") if s.get("scale") is not None else 576**-0.5,
    )
    result = {}
    for name, t in refs.items():
        axis = 0 if name.startswith("out_") else 1 if name.startswith("lse_") else 2
        pieces = [
            t[b, :q] if axis == 0 else t[b, :, :q] if axis == 1 else t[:, b, :, :q]
            for b, q in enumerate(s["q"])
        ]
        shape = (
            [0, s["heads"], 512]
            if axis == 0
            else [s["heads"], 0]
            if axis == 1
            else [26, s["heads"], 0] + ([512] if not name.startswith("accum_lse") else [])
        )
        result[name] = torch.cat(pieces, dim=axis) if pieces else t.new_empty(shape)
        if sum(s["kv"]) == 0 and name.startswith("lse_"):
            result[name].fill_(torch.inf)
    return result


def compare(c, result, refs=None):
    refs = reference(c) if refs is None else refs
    for i, (actual, name) in enumerate(zip(result, ("out", "lse", "accum", "accum_lse"))):
        actual = actual.cpu()
        assert actual.shape == refs[name + "_high"].shape
        assert actual.dtype == (c["raw"]["q"].dtype if i == 0 else torch.float32)
        # Init defines all unused allocation slots as zero/-inf, including
        # padding between samples and every unused static split slot.
        assert_fa_close(actual, refs[name + "_high"], refs[name + "_low"], name="auto-tnd-" + name)
    for before, actual in c["guards"]:
        torch.testing.assert_close(actual.cpu(), before, rtol=0, atol=0, equal_nan=True)
    return refs
