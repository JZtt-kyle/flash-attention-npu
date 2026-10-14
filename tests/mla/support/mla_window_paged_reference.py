"""Independent PA window math with content-addressed CPU golden reuse.

The scalar page/used mapping is shared by component, production and profile
checks. It never reads device metadata or migrated scheduling decisions.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

import torch

from tests.common import golden_cache
from tests.mla.support.mla_ref import mla_reference_pair_uncached as mla_reference_pair


def window_for(case):
    spec = case["config"]
    q = max(spec["q"], default=0)
    k = spec["capacity"]
    left, right = spec.get("window", (2, 1))
    if left >= k - 1:
        left = -1
    if right >= q - 1:
        right = -1
    causal = spec.get("causal", False)
    if q == 1 and left == right == -1:
        causal = False
    if causal:
        right = 0
    kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
    return kind, k - 1 if left < 0 else left, q - 1 if right < 0 else right


def uncached_references(case):
    q = case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"]
    heads, page = case["config"]["heads"], case["config"]["page"]
    kind, left, right = window_for(case)

    def one(b):
        qlen, klen = q[b], case["config"]["lengths"][b]
        if not qlen:
            return None
        if not klen:
            out = torch.zeros((qlen, heads, 512), dtype=case["host"]["v"].dtype)
            lse = torch.full((heads, qlen), -torch.inf)
            return [out, lse, out, lse]
        data = {n: case["host"][n][b : b + 1, :qlen].contiguous() for n in ("q", "qv")}
        # Scalar logical token mapping uses raw table/physical pool, never metadata.
        for n in ("k", "v"):
            data[n] = torch.stack(
                [
                    case["host"][n][int(case["table_host"][b, t // page]), t % page]
                    for t in range(klen)
                ]
            ).unsqueeze(0)
        return [
            t[0]
            for t in mla_reference_pair(
                **data,
                softmax_scale=case["scale"],
                causal=kind == 1,
                window_size=(left, right) if kind == 2 else (-1, -1),
            )
        ]

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(q)))) as pool:
        values = [r for r in pool.map(one, range(len(q))) if r is not None]
    return [
        torch.cat([r[i] for r in values], dim=i % 2)
        if values
        else torch.empty(
            (heads, 0) if i % 2 else (0, heads, 512),
            dtype=torch.float32 if i % 2 else case["host"]["v"].dtype,
        )
        for i in range(4)
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
        nodeid="mla-window-paged-reference",
        metadata=metadata,
        inputs=tensors,
        compute_fn=compute,
        expected_keys=names,
        source_files=sources,
        test_source_files=[str(Path(__file__).with_name("mla_window_paged_helpers.py"))],
    )
    return [values[name] for name in names]
