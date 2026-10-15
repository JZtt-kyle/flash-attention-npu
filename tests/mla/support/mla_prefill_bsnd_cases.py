"""Fixed public inputs chosen by GQA, masks, native tiles and split request."""

PERFORMANCE_CASES = [
    dict(shape=(2, 17, 8, 1057, 2), splits=1, mask="none"),
    dict(shape=(1, 65, 8, 513, 8), splits=1, mask="causal"),
    dict(shape=(2, 17, 8, 1057, 2), splits=2, mask="window", lengths=[1057, 517]),
    dict(shape=(1, 7, 8, 8193, 1), splits=0, mask="none"),
    dict(shape=(1, 129, 8, 1025, 2), splits=0, mask="causal"),
    dict(shape=(2, 33, 8, 2049, 2), splits=0, mask="window", lengths=[2049, 1025]),
]
