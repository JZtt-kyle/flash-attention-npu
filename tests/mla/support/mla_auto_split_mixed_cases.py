"""Mixed native automatic inputs fixed by coverage before execution or timing."""

from tests.mla.support.mla_split_stage_cases import split_stage_cases

from tests.mla.support.mla_window_mixed_cases import spec as mixed_spec


def spec(*args, **kwargs):
    return dict(mixed_spec(*args, **kwargs), splits=0)


# Each mask has a small daily input; no-mask has real native FD work.
DAILY_CASES = [
    spec(q=(1, 0, 2), k=(1025, 3, 513), capacity=1057, h=2, window=(-1, -1)),
    spec(q=(7, 3), k=(3, 0), capacity=17, h=2, causal=True, window=(-1, -1)),
    spec(q=(7, 0, 3), k=(31, 0, 0), capacity=33, h=3, window=(2, 1), stride="independent"),
]
REPRESENTATIVES = [
    # Simultaneous Direct/FD/Empty, nonuniform allocation and >256 rows.
    spec(q=(65, 0, 7, 3), k=(1057, 3, 0, 17), capacity=1057, h=4, window=(-1, -1)),
    spec(
        q=(7, 0, 65),
        k=(1025, 0, 511),
        capacity=1057,
        uq=[5, 0, 63],
        h=4,
        window=(530, 1),
        stride="independent",
    ),
    spec(q=(7, 3), k=(3, 0), capacity=17, h=2, causal=True, window=(-1, -1)),
]
# Largest allocated accumulator and longest reference in the fixed real inputs.
COST_CASES = [spec(q=(1, 2), k=(16384, 8193), capacity=16384, h=257, window=(-1, -1))]
# All six public entry/R segments participate for both dtypes. Native FD is
# included; the nonempty zero-used native empty template is reported separately.
PERFORMANCE_CASES = [
    spec(q=(1, 2), k=(16384, 8193), capacity=16384, h=16, window=(-1, -1)),
    spec(q=(3, 0, 7), k=(17, 0, 1057), capacity=1057, h=16, causal=True, window=(-1, -1)),
    spec(q=(65, 3), k=(1057, 513), capacity=1057, h=4, window=(513, 7)),
    spec(q=(7, 3), k=(511, 17), capacity=513, h=16, window=(17, 1), stride="independent"),
    spec(q=(1, 3), k=(16384, 17), capacity=16384, h=2, window=(-1, -1)),
    spec(q=(7, 3), k=(0, 0), capacity=33, h=4, window=(2, 1)),
    spec(q=(3,) * 32, k=(33, 17) * 16, capacity=33, h=2, window=(-1, -1)),
]


# Stage selection changes only performance; all correctness cases above stay intact.
PERFORMANCE_CASES = split_stage_cases(PERFORMANCE_CASES, "auto-split-mixed-profile")
