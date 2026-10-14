"""Fixed BSND window public correctness and performance inputs."""

from tests.mla.support.mla_validation_policy import window_public_cases

# Selected by semantics before measurement; every profile configuration also
# executes the same source/dual-math public correctness check.
PERFORMANCE_CASES = [
    dict(shape=(2, 7, 16, 33), window=(2, 1)),
    dict(shape=(2, 7, 16, 33), window=(2, 1), lengths=[3, 0]),
    dict(shape=(1, 129, 3, 1025), window=(513, 7)),
    dict(shape=(2, 7, 73, 513), window=(2, 1), lengths=[512, 1]),
    dict(shape=(1, 257, 1, 257), window=(17, -1)),
    dict(shape=(2, 513, 3, 257), window=(-1, 7), lengths=[257, 129]),
    dict(shape=(2, 129, 3, 513), window=(16, 7), causal=True, lengths=[17, 513]),
    dict(shape=(1, 1, 128, 16384), window=(512, 0)),
    dict(shape=(10, 3, 86, 8193), window=(112, 1)),
    dict(shape=(2, 7, 16, 33), window=(2, 1), lengths=[33, 17], stride="independent"),
    dict(shape=(2, 7, 16, 33), window=(-1, 2), lengths=[0, 0]),
    dict(shape=(2, 7, 16, 33), window=(32, 6)),
]
USED_CASES = [
    dict(shape=(2, 7, 16, 33), window=(2, 1), used=[5, 3], lengths=[3, 0]),
    dict(
        shape=(2, 129, 3, 1025),
        window=(513, 7),
        used=[65, 129],
        lengths=[17, 1025],
        stride="independent",
    ),
    dict(shape=(2, 7, 16, 33), window=(-1, 2), used=[0, 0], lengths=[0, 33]),
    dict(shape=(2, 7, 16, 33), window=(2, -1), used=[5, 3], implicit_kv=True),
]
REPRESENTATIVES = [PERFORMANCE_CASES[1], PERFORMANCE_CASES[2], USED_CASES[1]]
COST_CASES = [PERFORMANCE_CASES[7], PERFORMANCE_CASES[8]]
ACCURACY_CASES = [
    s for s in PERFORMANCE_CASES + USED_CASES if s not in REPRESENTATIVES + COST_CASES
]
ACCURACY_CASES += [
    dict(shape=(2, 7, 3, 33), window=w, lengths=[3, 17], causal=c, scale=scale)
    for w, c, scale in [
        ((0, 0), False, 0.0),
        ((2, -1), False, -0.1),
        ((-1, 2), False, 0.23),
        ((2, 9), True, None),
        ((2**63 - 1, 2**63 - 1), False, None),
        ((-2, -2), True, None),
        ((31, 5), False, None),
    ]
]


# Only stage performance changes this fixed candidate set; correctness lists above stay intact.

PERFORMANCE_CASES = window_public_cases(PERFORMANCE_CASES, "bsnd")
