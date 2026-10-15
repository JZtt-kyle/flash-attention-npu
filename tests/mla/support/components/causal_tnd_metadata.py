"""Shared helpers extracted from tests/mla/unit/causal_tnd/test_flash_attn_npu_v3_mla_causal_tnd_metadata.py.

v8 TND allocation/used identity, pinned-source schedule and release consumer.
"""

import itertools


def request_for(item, dtype=1, cores=20):
    q, k = item["q"], item["kv"]
    uq, uk = item["uq"], item["uk"]
    return [
        len(q),
        item.get("sq", max(q, default=0)),
        item["heads"],
        item.get("sk", max(k, default=0)),
        sum(q),
        sum(k),
        dtype,
        int(uq is not None) + 2 * int(uk is not None),
        cores,
        *itertools.accumulate(q, initial=0),
        *itertools.accumulate(k, initial=0),
        *itertools.chain.from_iterable(zip(q if uq is None else uq, k if uk is None else uk)),
    ]
