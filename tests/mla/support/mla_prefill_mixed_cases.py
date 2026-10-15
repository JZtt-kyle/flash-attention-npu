"""Fixed before sampling: mixed allocation, GQA, masks, splits and KV gaps."""

PERFORMANCE_CASES = [
    dict(q=[17, 3], k=[1057, 513], heads=8, kv_heads=2, splits=1, mask="none", window=[-1, -1]),
    dict(q=[65, 1], k=[513, 31], heads=8, kv_heads=8, splits=1, mask="causal", window=[-1, -1]),
    dict(
        q=[17, 3],
        k=[1057, 513],
        heads=8,
        kv_heads=2,
        splits=2,
        mask="window",
        window=[7, 2],
        used_k=[517, 31],
    ),
    dict(q=[7, 1], k=[8193, 513], heads=8, kv_heads=1, splits=0, mask="none", window=[-1, -1]),
    dict(q=[129, 3], k=[1025, 513], heads=8, kv_heads=2, splits=0, mask="causal", window=[-1, -1]),
    dict(
        q=[33, 7],
        k=[2049, 1025],
        heads=8,
        kv_heads=2,
        splits=0,
        mask="window",
        window=[7, 2],
        used_k=[1025, 513],
    ),
]
