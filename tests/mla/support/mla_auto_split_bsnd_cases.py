"""Fixed automatic-split performance inputs; chosen before timing, both dtypes."""

PERFORMANCE_CASES = [
    dict(shape=(1, 1, 128, 16384), window=(-1, -1), splits=0),
    dict(shape=(2, 3, 16, 8193), window=(-1, -1), splits=0),
    dict(shape=(2, 7, 16, 1057), window=(-1, -1), causal=True, splits=0),
    dict(shape=(1, 65, 4, 1057), window=(513, 7), splits=0),
    dict(shape=(2, 7, 16, 513), window=(2, 1), lengths=[512, 1], splits=0),
    dict(shape=(2, 7, 16, 513), window=(17, 1), lengths=[513, 17], stride="independent", splits=0),
    dict(shape=(1, 1, 16, 16384), window=(-1, -1), splits=0),
    dict(shape=(2, 7, 16, 33), window=(2, 1), lengths=[0, 0], splits=0),
    dict(shape=(4, 65, 8, 33), window=(-1, -1), splits=0),
]
