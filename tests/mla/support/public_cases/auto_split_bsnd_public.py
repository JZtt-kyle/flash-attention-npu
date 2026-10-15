"""Shared helpers extracted from tests/mla/e2e/auto_split_bsnd/test_public.py.

Native automatic BSND split: public results, static capacity and lifecycle.
"""

from tests.mla.support.public_cases.split_kv_bsnd_public import (
    make_case as explicit_case,
    metadata as metadata,
    invoke as invoke,
)


def make_case(dtype="fp16", **kwargs):
    return explicit_case(dtype, splits=0, **kwargs)


def geometry(record, batch):
    words = record.cpu().tolist()
    base = 410 + 8 * batch
    count = words[base + 3]
    ends = list(
        zip(
            words[base + 4 : base + 4 + count],
            words[base + 30 : base + 30 + count],
            words[base + 56 : base + 56 + count],
        )
    )
    return words[15], words[16], ends
