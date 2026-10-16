"""Fixed PA inputs: native tiles, GQA, masks, split policies and page boundaries."""

PERFORMANCE_CASES = [
    dict(shape=(2, 17, 8, 1057, 2), splits=1, mask="none", page=16),
    dict(shape=(1, 65, 8, 513, 8), splits=1, mask="causal", page=112),
    dict(shape=(2, 17, 8, 1057, 2), splits=2, mask="window", page=528, lengths=[1057, 517]),
    dict(shape=(1, 7, 8, 8193, 1), splits=0, mask="none", page=1024),
    dict(shape=(1, 129, 8, 1025, 2), splits=0, mask="causal", page=112),
    dict(shape=(2, 33, 8, 2049, 2), splits=0, mask="window", page=528, lengths=[2049, 1025]),
]
