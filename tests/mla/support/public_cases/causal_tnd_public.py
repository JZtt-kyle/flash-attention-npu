"""Shared helpers extracted from tests/mla/e2e/causal_tnd/test_flash_attn_npu_v3_mla_causal_tnd_public.py.

TND causal varlen: allocation gaps, source/math, public boundaries and lifetime.
"""

import hashlib
import itertools
import json
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.support.mla_validation_policy import public_cases
from tests.mla.support.components.bsnd_stride import make_view, strides_for
from tests.mla.support.components.causal_attention import make_case, launch
from tests.mla.support.public_cases.contract import _forward
from tests.mla.support.components.used_schedule import used_case, execute


def spec(q=(7, 0, 3), k=(33, 0, 17), uq=None, uk=None, h=16, **kwargs):
    return dict(q=list(q), kv=list(k), uq=uq, uk=uk, heads=h, **kwargs)


REPRESENTATIVES = [
    spec(uq=[5, 0, 3], uk=[3, 0, 0]),
    spec(q=(257, 3), k=(257, 513), h=1),
    spec(uq=[5, 0, 3], uk=[3, 0, 17], stride="independent"),
]

PERFORMANCE_CASES = public_cases(
    [
        spec(q=(3, 0, 1), k=(17, 0, 33), h=128),
        spec(q=(1, 7, 2), k=(1, 513, 31)),
        spec(q=(0, 3, 1, 0, 7), k=(33, 511, 0, 513, 1025), h=128),
        spec(q=(1, 2, 3), k=(16384, 8193, 17), h=8),
        spec(q=(257, 1, 0, 3), k=(33, 513, 17, 1), h=1),
        spec(q=(3, 0) * 21, k=(0, 513) * 21, h=128),
        *REPRESENTATIVES,
        spec(q=(129, 3), k=(513, 17), uq=[65, 1], uk=[257, 3], h=3),
        spec(q=(3, 2), k=(16384, 8193), uq=[1, 2], uk=[16384, 8193], h=8),
        spec(stride="ntd", h=128),
        spec(q=(5, 1), k=(17, 17), uq=[3, 1], uk=[17, 7], h=5, stride="combined_65536"),
        spec(q=(513, 257), k=(257, 129), h=3),
    ],
    group="tnd",
)


def effective(s):
    return s["q"] if s["uq"] is None else s["uq"], s["kv"] if s["uk"] is None else s["uk"]


def prepare(dtype, s, *, poison=True):
    gen = torch.Generator().manual_seed(20261013)
    raw, device, guards = {}, {}, []
    uq, uk = effective(s)
    for name, dim, offset in (("q", 64, 5), ("qv", 512, 11), ("k", 64, 19), ("v", 512, 29)):
        query = name in ("q", "qv")
        allocation, used = (s["q"], uq) if query else (s["kv"], uk)
        tensor = torch.randn((sum(allocation), s["heads"] if query else 1, dim), generator=gen).to(
            dtype
        )
        if query:
            tensor.mul_(s.get("amplitude", 1))
        start = 0
        if poison:
            for capacity, length in zip(allocation, used):
                tensor[start + length : start + capacity] = torch.nan
                start += capacity
        if s.get("stride"):
            shape = (1, *tensor.shape)
            profile = "bnsd" if s["stride"] == "ntd" else s["stride"]
            view, logical, checks = make_view(
                tensor.unsqueeze(0),
                strides_for(shape, profile, name),
                offset,
                overlap=profile in ("broadcast", "overlap"),
            )
            device[name], raw[name] = view[0], logical[0]
            guards += checks
        else:
            raw[name], device[name] = tensor, tensor.npu()
    args = dict(
        device,
        cu_seqlens_q=torch.tensor(
            list(itertools.accumulate(s["q"], initial=0)), dtype=torch.int32, device="npu"
        ),
        cu_seqlens_k=torch.tensor(
            list(itertools.accumulate(s["kv"], initial=0)), dtype=torch.int32, device="npu"
        ),
        max_seqlen_q=s.get("sq", max(s["q"], default=0)),
        max_seqlen_k=s.get("sk", max(s["kv"], default=0)),
        seqused_q=None
        if s["uq"] is None
        else torch.tensor(s["uq"], dtype=torch.int32, device="npu"),
        seqused_k=None
        if s["uk"] is None
        else torch.tensor(s["uk"], dtype=torch.int32, device="npu"),
        softmax_scale=s.get("scale"),
    )
    return dict(spec=s, raw=raw, args=args, guards=guards)


def call(args, entry="public"):
    if entry == "public":
        return api.flash_attn_varlen_func(**args, causal=True, num_splits=1, return_attn_probs=True)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_causal_tnd_forward(**args)
    return _forward(entry, dict(args, causal=True, num_splits=1))[:2]


def source_tile(probes, case, oracle):
    s = case["spec"]
    details = make_case(probes, dict(shape=(1, 1, 1, 1)))["details"]
    item = used_case(
        1,
        s["q"],
        s["kv"],
        s["uq"],
        s["uk"],
        heads=s["heads"],
        aic=int(details[2]),
        ratio=2,
        sq=case["args"]["max_seqlen_q"],
        sk=case["args"]["max_seqlen_k"],
    )
    row = execute(oracle, [item])[0]
    assert row[0] == 0
    return row[1:3]


def compare(probes, case, results, tile):
    s, raw = case["spec"], case["raw"]
    uq, uk = effective(s)
    start_q = start_k = 0
    reports = []
    outputs = [[t.cpu() for t in result] for result in results]
    for b, (q, k) in enumerate(zip(uq, uk)):
        if q:
            # Independent per-sample source input, using original source-selected
            # tile geometry. Padding preserves capacity without exposing gaps.
            source = make_case(
                probes,
                dict(
                    shape=(1, q, s["heads"], max(s["kv"])),
                    lengths=[k],
                    scale=s.get("scale"),
                    m=tile[0],
                    n=tile[1],
                ),
            )
            logical = {}
            for name, data in raw.items():
                query = name in ("q", "qv")
                start, length = (start_q, q) if query else (start_k, k)
                logical[name] = data[start : start + length].unsqueeze(0).contiguous()
                source["inputs"][name].fill_(torch.nan)
                source["inputs"][name][:, :length].copy_(logical[name])
                source["device"][name].copy_(source["inputs"][name])
            assert launch(probes, source, "oracle") == 0
            ref_source = [t.cpu() for t in source["outputs"]["oracle"]]
            masked = max(0, q - k)
            assert (ref_source[0][:, :masked] == 0).all() and ref_source[1][
                :, :, :masked
            ].isposinf().all()
            ref_source[1][:, :, :masked] = -torch.inf
            refs = mla_reference_pair(**logical, softmax_scale=s.get("scale"), causal=True)
            if not k:
                refs[1].fill_(-torch.inf)
                refs[3].fill_(-torch.inf)
            actual = []
            try:
                for result in outputs:
                    got = [
                        result[0][start_q : start_q + q].unsqueeze(0),
                        result[1][:, start_q : start_q + q].unsqueeze(0),
                    ]
                    actual.append(got)
                    reports.append(
                        {
                            name: assert_mla_source_close(
                                got[i], ref_source[i], refs[i], refs[i + 2], name=name
                            )
                            for i, name in enumerate(("out", "lse"))
                        }
                    )
            except AssertionError:
                torch.save(
                    dict(
                        spec=s,
                        sample=b,
                        inputs=logical,
                        source=ref_source,
                        refs=refs,
                        actual=actual,
                    ),
                    probes[1] / "tnd-source-failure.pt",
                )
                raise
            if any(v["math"] != "passed" for row in reports for v in row.values()):
                identity = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]
                torch.save(
                    dict(
                        spec=s,
                        sample=b,
                        inputs=logical,
                        source=ref_source,
                        refs=refs,
                        actual=actual,
                    ),
                    probes[1] / f"tnd-source-limit-{identity}-{b}.pt",
                )
        start_q += s["q"][b]
        start_k += s["kv"][b]
    identity = hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:16]
    (probes[1] / f"tnd-source-{identity}.json").write_text(
        json.dumps(dict(spec=s, tile=tile, comparisons=reports), indent=2) + "\n"
    )
    return reports
