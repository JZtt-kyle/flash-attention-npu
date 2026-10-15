"""Shared helpers extracted from tests/mla/unit/used/test_flash_attn_npu_v3_mla_used_schedule.py.

CPU used-length normalization and pinned-source scheduling differential.

Allocation geometry and effective source lengths are deliberately separate.
The oracle compiles unmodified CANN parser/tiling/SplitCore functions with an
independent FA3 input adapter; it never links the migrated length parser.
No device metadata or public used-length entry is enabled by these tests.
"""

import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.components.schedule import case
from tests.mla.support.components.tnd_schedule import build_oracle, evaluate


@pytest.fixture(scope="session")
def used_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_USED_SCHEDULE_ARTIFACT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-used-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def used_oracle(used_artifacts):
    return build_oracle(used_artifacts, used=True)


def used_case(layout=0, q=(7, 7), kv=(33, 33), uq=None, uk=None, **kwargs):
    """Build raw driver input, independently of any migrated normalization."""
    item = dict(layout=layout, q=list(q), kv=list(kv), uq=uq, uk=uk)
    item.update(kwargs)
    return item


def encode(item):
    layout, q, kv = item["layout"], item["q"], item["kv"]
    cq = item.get("cq", list(itertools.accumulate(q, initial=0)) if layout != 0 else [])
    ck = item.get("ck", list(itertools.accumulate(kv, initial=0)) if layout == 1 else [])
    defaults = dict(batch=len(q), sq=max(q, default=0), sk=max(kv, default=0))
    defaults.update(
        {
            k: v
            for k, v in item.items()
            if k
            in (
                "batch",
                "sq",
                "sk",
                "heads",
                "aic",
                "aiv",
                "ratio",
                "block_dim",
                "libapi",
                "l1",
                "ub",
                "l0a",
                "l0b",
                "l0c",
            )
        }
    )
    encoded = case(**defaults) + [
        int(item.get("has_k", layout == 1)),
        item.get("kcount", len(ck)),
        len(ck),
        *ck,
        item.get("total_q", sum(q)),
        item.get("total_k", sum(kv)),
        int(item.get("has_q", layout != 0)),
        item.get("qcount", len(cq)),
        len(cq),
        *cq,
        layout,
    ]
    for name in ("uq", "uk"):
        values = item[name]
        encoded += [
            int(item.get("has_" + name, values is not None)),
            item.get(name + "count", 0 if values is None else len(values)),
            0 if values is None else len(values),
            *(values or []),
        ]
    return encoded


def execute(binary, items, **kwargs):
    return evaluate(binary, [encode(item) for item in items], **kwargs)


def spans(item):
    q, kv = item["q"], item["kv"]
    uq = q if item["uq"] is None else item["uq"]
    uk = kv if item["uk"] is None else item["uk"]
    starts_q = list(itertools.accumulate(q, initial=0))
    starts_k = list(itertools.accumulate(kv, initial=0))
    return [
        max(uq, default=0),
        max(uk, default=0),
        max(q, default=0),
        max(kv, default=0),
        *itertools.chain.from_iterable(
            (starts_q[b], starts_k[b], q[b], kv[b], uq[b], uk[b]) for b in range(len(q))
        ),
    ]
