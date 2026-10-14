"""Fixed TND window inputs selected by semantics before performance sampling."""

from tests.mla.support.mla_validation_policy import window_public_cases


def spec(q=(7, 0, 3), k=(33, 0, 17), uq=None, uk=None, h=16, window=(2, 1), **kwargs):
    return dict(q=list(q), kv=list(k), uq=uq, uk=uk, heads=h, window=list(window), **kwargs)


PERFORMANCE_CASES = [
    spec(),
    spec(uq=[5, 0, 3], uk=[3, 0, 0]),
    spec(q=(129, 3), k=(1025, 17), h=3, window=(513, 7)),
    spec(q=(257, 1), k=(257, 513), h=1, window=(17, -1)),
    spec(q=(513, 257), k=(257, 129), h=3, window=(-1, 7)),
    spec(q=(129, 3), k=(513, 17), uq=[65, 1], uk=[257, 3], h=3, window=(16, 7), causal=True),
    spec(q=(1, 2, 3), k=(16384, 8193, 17), h=8, window=(512, 0)),
    spec(q=(3, 0) * 21, k=(8193, 17) * 21, h=86, window=(112, 1)),
    spec(uq=[5, 0, 3], uk=[3, 0, 17], stride="independent"),
    spec(q=(7, 3), k=(33, 513), uq=[3, 2], uk=[17, 257], sq=64, sk=1025, window=(31, 5)),
    spec(window=(32, 6)),
    spec(window=(-1, 0)),
]
REPRESENTATIVES = [PERFORMANCE_CASES[1], PERFORMANCE_CASES[2], PERFORMANCE_CASES[8]]
COST_CASES = [PERFORMANCE_CASES[6], PERFORMANCE_CASES[7]]
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
    + [spec(uq=[0, 0, 0], uk=[33, 0, 17]), spec(uk=[0, 0, 0]), spec(h=257)]
)


# Only stage performance changes this fixed candidate set; correctness lists above stay intact.

PERFORMANCE_CASES = window_public_cases(PERFORMANCE_CASES, "tnd")
