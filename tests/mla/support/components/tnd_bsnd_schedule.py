"""Shared helpers extracted from tests/mla/unit/mixed/test_flash_attn_npu_v3_mla_tnd_bsnd_schedule.py.

Independent CPU checks for compact TND Q with a physical BSND KV cache.

The source executable uses pinned CANN functions and numerical int64 Q tails /
noncumulative KV lengths, never the migrated parser. No mixed metadata, device
attention or public entry is enabled. All tests also support full ASan/UBSan.
"""

import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.components.schedule import case
from tests.mla.support.components.tnd_schedule import build_oracle


@pytest.fixture(scope="session")
def mixed_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_TND_BSND_SCHEDULE_ARTIFACT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-tnd-bsnd-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="session")
def mixed_oracle(mixed_artifacts):
    return build_oracle(mixed_artifacts, mixed=True)


def mixed_case(
    qlens=(3, 0, 1),
    klens=(17, 0, 33),
    *,
    capacity=64,
    cq=None,
    qcount=None,
    kcount=None,
    has_q=True,
    has_k=True,
    total_q=None,
    **shape,
):
    cq = list(itertools.accumulate(qlens, initial=0)) if cq is None else list(cq)
    defaults = dict(batch=len(qlens), sq=max(qlens, default=0), sk=capacity)
    defaults.update(shape)
    # The common harness's totalKv field is unused in the mixed API: physical
    # cache capacity is B*Sk, with no int32 cumulative KV array or total bound.
    return case(**defaults) + [
        int(has_k),
        len(klens) if kcount is None else kcount,
        len(klens),
        *klens,
        cq[-1] if total_q is None else total_q,
        0,
        int(has_q),
        len(cq) if qcount is None else qcount,
        len(cq),
        *cq,
    ]


def input_lengths(item):
    kv = item[16 : 16 + item[15]]
    cq = item[21 + item[15] :]
    return cq, [b - a for a, b in zip(cq, cq[1:])], kv


def expected_spans(item):
    cq, q, kv = input_lengths(item)
    return [
        max(q, default=0),
        max(kv, default=0),
        *itertools.chain.from_iterable((cq[b], b * item[3], q[b], kv[b]) for b in range(item[0])),
    ]
