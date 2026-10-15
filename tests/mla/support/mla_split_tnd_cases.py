"""TND explicit-split inputs fixed by independent coverage before measurement."""

from tests.mla.support.mla_split_stage_cases import split_stage_cases

from tests.mla.support.mla_window_tnd_cases import spec as window_spec


def spec(*args, splits=8, **kwargs):
    return dict(window_spec(*args, **kwargs), splits=splits)


REPRESENTATIVES = [
    spec(
        q=(7, 0, 65),
        k=(1057, 3, 513),
        uq=[5, 0, 63],
        uk=[1025, 0, 511],
        h=4,
        window=(530, 1),
        stride="independent",
    ),
    spec(q=(7, 3), k=(3, 0), h=2, causal=True, window=(-1, -1), splits=256),
    spec(q=(1, 2), k=(16384, 8193), h=2, window=(-1, -1), splits=2),
    spec(q=(65, 1), k=(1057, 17), h=4, window=(513, 7), splits=256),
]
DAILY_CASES = [
    spec(q=(3, 0, 2), k=(17, 3, 9), h=2, window=(-1, -1), splits=2),
    spec(q=(7, 3), k=(3, 0), h=2, causal=True, window=(-1, -1), splits=8),
    spec(
        q=(7, 0, 3),
        k=(33, 1, 17),
        uq=[5, 0, 1],
        uk=[31, 0, 0],
        h=3,
        window=(2, 1),
        stride="independent",
        splits=8,
    ),
]
PERFORMANCE_CASES = [
    spec(q=(1, 2), k=(16384, 8193), h=16, window=(-1, -1), splits=2),
    spec(q=(3, 0, 7), k=(17, 0, 1057), h=16, causal=True, window=(-1, -1), splits=8),
    spec(q=(65, 3), k=(1057, 513), h=4, window=(513, 7), splits=8),
    spec(
        q=(7, 3),
        k=(513, 33),
        uq=[5, 1],
        uk=[511, 17],
        h=16,
        window=(17, 1),
        stride="independent",
        splits=2,
    ),
    spec(q=(1, 3), k=(16384, 17), h=2, window=(-1, -1), splits=256),
    spec(q=(7, 3), k=(33, 17), uk=[0, 0], h=4, window=(2, 1), splits=8),
]


# Stage selection changes only performance; all correctness cases above stay intact.
PERFORMANCE_CASES = split_stage_cases(PERFORMANCE_CASES, "split-tnd-profile")
