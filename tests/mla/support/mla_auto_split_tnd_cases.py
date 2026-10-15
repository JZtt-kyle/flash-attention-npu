"""Native TND automatic inputs, fixed by coverage before correctness or timing."""

from tests.mla.support.mla_split_stage_cases import split_stage_cases

from tests.mla.support.mla_window_tnd_cases import spec as window_spec


def spec(*args, **kwargs):
    return dict(window_spec(*args, **kwargs), splits=0)


# Each daily input covers a mask; mixed allocation/used, empty spans and stride
# are explicit. The first has multiple S2 tiles per M and exercises native FD.
DAILY_CASES = [
    spec(q=(1, 0, 2), k=(1025, 3, 513), h=2, window=(-1, -1)),
    spec(q=(7, 3), k=(3, 0), h=2, causal=True, window=(-1, -1)),
    spec(
        q=(7, 0, 3),
        k=(33, 1, 17),
        uq=[5, 0, 1],
        uk=[31, 0, 0],
        h=3,
        window=(2, 1),
        stride="independent",
    ),
]
REPRESENTATIVES = [
    # Simultaneous Direct/FD/Empty, nonuniform compact spans and >256 rows.
    spec(q=(65, 0, 7, 3), k=(1057, 3, 0, 17), h=4, window=(-1, -1)),
    spec(
        q=(7, 0, 65),
        k=(1057, 3, 513),
        uq=[5, 0, 63],
        uk=[1025, 0, 511],
        h=4,
        window=(530, 1),
        stride="independent",
    ),
    spec(q=(7, 3), k=(3, 1), uk=[3, 0], h=2, causal=True, window=(-1, -1)),
]
# Longest reference and largest accumulator in this module's fixed inputs.
COST_CASES = [spec(q=(1, 2), k=(16384, 8193), h=128, window=(-1, -1))]
PERFORMANCE_CASES = [
    spec(q=(1, 2), k=(16384, 8193), h=16, window=(-1, -1)),
    spec(q=(3, 0, 7), k=(17, 0, 1057), h=16, causal=True, window=(-1, -1)),
    spec(q=(65, 3), k=(1057, 513), h=4, window=(513, 7)),
    spec(
        q=(7, 3), k=(513, 33), uq=[5, 1], uk=[511, 17], h=16, window=(17, 1), stride="independent"
    ),
    spec(q=(1, 3), k=(16384, 17), h=2, window=(-1, -1)),
    spec(q=(7, 3), k=(33, 17), uk=[0, 0], h=4, window=(2, 1)),
    # Enough direct work to exercise reuse of each core's temporary raw slot.
    spec(q=(3,) * 32, k=(33, 17) * 16, h=2, window=(-1, -1)),
]


# Stage selection changes only performance; all correctness cases above stay intact.
PERFORMANCE_CASES = split_stage_cases(PERFORMANCE_CASES, "auto-split-tnd-profile")
