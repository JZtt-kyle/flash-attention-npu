"""Shared helpers extracted from tests/mla/unit/paged_tnd/test_flash_attn_npu_v3_mla_tnd_paged_attention.py.

Independent TND Q / ND PA attention; allocation prefixes never follow used.

Source Q is independently packed before its pinned PA/non-FD invocation. KV
lengths remain non-cumulative. The shared math adapter indexes original pages
and allocation segments, without reading v6 spans or device Copy addresses.
"""

from tests.mla.support.components import paged_attention as bsnd
from tests.mla.support.components.paged_attention import (
    paged_attention as paged_attention,
    paged_probe as paged_probe,
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    make_case,
    launch as launch,
    check_case as check_case,
    check_rejected as check_rejected,
    references as references,
    valid_outputs as valid_outputs,
    load_attention as load_attention,
    configure_producer as configure_producer,
    forbid_host_readback as forbid_host_readback,
    enqueue as enqueue,
)


def config(page=112, profile="independent", q=(5, 2), lengths=None, uq=(3, 1), heads=8, **kwargs):
    return bsnd.config(page, profile, q, lengths, uq, heads, layout=2, **kwargs)


# Frozen before measurement: native PA01/PA02 page strides, independent Q
# allocations, zero-Q/zero-KV samples and enough physical pages for long KV.
PERFORMANCE_CASES = [
    config(p, profile, q, kv, uq, h, pool=2 + sum((n + p - 1) // p for n in kv))
    for p in (16, 112, 128, 256, 528, 1024)
    for profile, q, kv, uq, h in (
        ("contiguous", (1, 0, 1), (257, 17, 0), None, 128),
        ("pa02_both", (7, 0, 3), (8193, 17, 1025), (3, 0, 1), 128),
    )
]


def rejection_case(probes, producer):
    return make_case(
        probes, producer, config(q=(5, 2), uq=(3, 1), lengths=(257, 257), capacity=448, aic=1)
    )
