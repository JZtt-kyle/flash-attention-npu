"""Fixed mixed window inputs, chosen by semantics before any timing."""

from tests.mla.support.mla_validation_policy import window_public_cases


def spec(
    q=(7, 0, 3), k=(3, 0, 17), capacity=33, uq=None, h=16, implicit_k=False, window=(2, 1), **kw
):
    return dict(
        q=list(q),
        kv=[capacity] * len(q),
        uq=uq,
        uk=None if implicit_k else list(k),
        heads=h,
        capacity=capacity,
        window=list(window),
        **kw,
    )


PERFORMANCE_CASES = [
    spec(),
    spec(q=(129, 3), k=(1025, 17), capacity=1025, h=3, window=(513, 7)),
    spec(q=(257, 1), k=(257, 513), capacity=513, h=1, window=(17, -1)),
    spec(q=(513, 257), k=(257, 129), capacity=513, h=3, window=(-1, 7)),
    spec(q=(129, 3), k=(257, 3), capacity=513, h=3, window=(16, 7), causal=True),
    spec(q=(1, 2, 3), k=(16384, 8193, 17), capacity=16384, h=8, window=(512, 0)),
    spec(q=(3, 0) * 21, k=(8193, 17) * 21, capacity=8193, h=86, window=(112, 1)),
    spec(stride="independent"),
    spec(q=(7, 3), k=(17, 257), capacity=1025, sq=64, window=(31, 5)),
    spec(window=(32, 6)),
    spec(window=(-1, 0)),
    spec(implicit_k=True, stride="ntd"),
]
REPRESENTATIVES = [
    PERFORMANCE_CASES[0],
    PERFORMANCE_CASES[1],
    spec(uq=[5, 0, 3], stride="independent"),
]
COST_CASES = [PERFORMANCE_CASES[5], PERFORMANCE_CASES[6]]
ACCURACY_CASES = (
    [s for s in PERFORMANCE_CASES if s not in REPRESENTATIVES + COST_CASES]
    + [
        spec(window=w, causal=c, scale=scale)
        for w, c, scale in [
            ((0, 0), False, 0.0),
            ((2, -1), False, -0.1),
            ((-1, 2), False, 0.23),
            ((2, 9), True, None),
            ((2**63 - 1, 2**63 - 1), False, None),
            ((-2, -2), False, None),
            ((-2, -2), True, None),
        ]
    ]
    + [
        spec(uq=[0, 0, 0]),
        spec(k=(0, 0, 0)),
        spec(h=257),
        spec(q=(3, 7), k=(17, 257), capacity=1025, uq=[1, 2], sq=64),
    ]
)


# Only stage performance changes this fixed candidate set; correctness lists above stay intact.

PERFORMANCE_CASES = window_public_cases(PERFORMANCE_CASES, "mixed")
