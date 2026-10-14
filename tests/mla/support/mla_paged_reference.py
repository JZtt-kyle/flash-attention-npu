"""Independent PA math with optional content-addressed CPU golden reuse.

The scalar page/used mapping is shared by component, production and profile
checks. It never reads device metadata or migrated scheduling decisions.
"""

from tests.mla.paths import ROOT

import itertools
import os
from pathlib import Path
import sys

import torch

from tests.common import golden_cache
from tests.mla.support.mla_ref import mla_reference_pair_uncached as mla_reference_pair


def uncached_references(case):
    q = case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"]
    results = [[], [], [], []]
    heads, page = case["context"].heads, case["config"]["page"]
    starts = list(itertools.accumulate([0, *case["item"]["q"]]))
    for b, (qlen, klen) in enumerate(zip(q, case["config"]["lengths"])):
        if not qlen:
            continue
        if klen:
            data = {
                n: case["host"][n][:, starts[b] : starts[b] + qlen]
                if case["item"]["layout"] == 2
                else case["host"][n][b : b + 1, :qlen]
                for n in ("q", "qv")
            }
            # Independent scalar token indexing, never schedule spans/Copy traces.
            for n in ("k", "v"):
                data[n] = torch.stack(
                    [
                        case["host"][n][int(case["table_host"][b, t // page]), t % page]
                        for t in range(klen)
                    ]
                ).unsqueeze(0)
            refs = [
                t[0]
                for t in mla_reference_pair(
                    **data, softmax_scale=case["scale"], causal=case.get("causal", False)
                )
            ]
        else:
            out = torch.zeros((qlen, heads, 512), dtype=case["host"]["v"].dtype)
            lse = torch.full((heads, qlen), -torch.inf)
            refs = [out, lse, out, lse]
        for dst, ref in zip(results, refs):
            dst.append(ref)
    return [
        torch.cat(r, dim=i % 2)
        if r
        else torch.empty(
            (heads, 0) if i % 2 else (0, heads, 512),
            dtype=torch.float32 if i % 2 else case["host"]["v"].dtype,
        )
        for i, r in enumerate(results)
    ]


def tensor_layout(tensor):
    return dict(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        stride=list(tensor.stride()),
        offset=tensor.storage_offset(),
    )


def references(case):
    """Cache only references; all tested device calls remain the caller's job."""
    names = ("out64", "lse64", "out_low", "lse_low")

    def compute():
        return dict(zip(names, uncached_references(case)))

    tensors = dict(case["host"], page_table=case["table_host"])
    # Logical bytes alone do not distinguish stride/offset, used or prefix
    # interpretations. Include raw descriptors and all adapter inputs too.
    metadata = dict(
        item=case["item"],
        config=case["config"],
        scale=case["scale"],
        causal=case.get("causal", False),
        heads=case["context"].heads,
        layout={name: tensor_layout(t) for name, t in tensors.items()},
        device_layout={
            name: tensor_layout(t)
            for name, t in case.get("devices", {}).get("migrated", {}).items()
        },
        table_layout=tensor_layout(case["tables"]["migrated"]) if "tables" in case else None,
        python=sys.version,
        omp=os.environ.get("OMP_NUM_THREADS"),
        mkl=os.environ.get("MKL_NUM_THREADS"),
    )
    sources = [
        str(Path(__file__)),
        str(Path(__file__).with_name("mla_ref.py")),
        golden_cache.__file__,
    ]
    values = golden_cache.get_or_compute_golden(
        nodeid="mla-paged-reference",
        metadata=metadata,
        inputs=tensors,
        compute_fn=compute,
        expected_keys=names,
        source_files=sources,
        test_source_files=[
            str(ROOT / "tests/mla/unit/paged_bsnd/test_flash_attn_npu_v3_mla_paged_attention.py")
        ],
    )
    return [values[name] for name in names]
