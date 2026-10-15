"""Shared helpers extracted from tests/mla/unit/paged_bsnd/test_flash_attn_npu_v3_mla_paged_schedule.py.

Independent PA CPU scheduling; no metadata or public PA entry is enabled.

The oracle uses pinned, verbatim CANN parser/tiling/SplitCore bodies. Its input
adapter is independent of the migrated normalization; KV is always per sample.
Native FD candidates (including all vectors/workspace) are checked separately
against the migrated SplitCore, and retained alongside FA3 whole-M results.
"""

import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.components.tnd_schedule import build_oracle, compile_driver, evaluate
from tests.mla.support.components.used_schedule import encode as encode_used, used_case


@pytest.fixture(scope="session")
def paged_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_PAGED_SCHEDULE_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-paged-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def paged_oracles(paged_artifacts):
    source = build_oracle(paged_artifacts, used=True, paged=True)
    migrated = compile_driver(paged_artifacts, True, used=True, paged=True, migrated_split=True)
    return source, migrated


def paged_case(layout=0, q=(7, 7), lengths=(17, 0), uq=None, page=112, capacity=None, **kwargs):
    if capacity is None:
        capacity = ((max(lengths, default=0) + page - 1) // page + 1) * page
    return used_case(
        layout, q, [capacity] * len(q), uq, list(lengths), page=page, sk=capacity, **kwargs
    )


def encode(item):
    return [*encode_used(item), item["page"]]


def execute(binary, items, **kwargs):
    return evaluate(binary, [encode(item) for item in items], **kwargs)


def spans(item):
    q = item["q"]
    uq = q if item["uq"] is None else item["uq"]
    starts = list(itertools.accumulate(q, initial=0))
    return [
        max(uq, default=0),
        max(item["uk"], default=0),
        max(q, default=0),
        *itertools.chain.from_iterable(
            (starts[b], q[b], uq[b], item["uk"][b]) for b in range(len(q))
        ),
    ]
