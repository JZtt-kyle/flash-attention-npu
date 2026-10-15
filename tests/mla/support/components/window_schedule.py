"""Shared helpers extracted from tests/mla/unit/window_bsnd/test_flash_attn_npu_v3_mla_window_schedule.py.

Independent FA3 normalization, CANN band candidates and row/task ownership.
"""


def normalized(config, *, forward=False):
    q, k, left, right, causal = config
    if left >= k - 1:
        left = -1
    if right >= q - 1:
        right = -1
    if q == 1 and left == right == -1:
        causal = False
    if causal:
        right = 0
    if forward:
        left, right = ((v + 2**31) % 2**32 - 2**31 for v in (left, right))
    kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
    return [kind, k - 1 if left < 0 else left, q - 1 if right < 0 else right]
