"""TND PA window: nonuniform allocations, used gaps and page/mask boundaries."""

from tests.mla.support.mla_validation_policy import window_public_cases
from tests.mla.support.mla_window_paged_cases import spec as bsnd_spec


def spec(page=112, **kwargs):
    kwargs.setdefault("q", (7, 0, 3))
    kwargs.setdefault("lengths", (3, 17, page + 1))
    kwargs.setdefault("uq", None)
    return bsnd_spec(page, layout=2, **kwargs)


REPRESENTATIVES = [
    spec(112, q=(7, 0, 3), lengths=(3, 17, 225)),
    spec(528, window=(513, 17), q=(257, 3), lengths=(257, 1057), heads=1),
    spec(
        16, window=(17, 2), causal=True, q=(7, 0, 3), lengths=(3, 17, 33), uq=(5, 0, 1), repeat=True
    ),
]
PERFORMANCE_CASES = [
    spec(
        p,
        window=(129, 2),
        q=(3, 0, 1),
        lengths=(8193, 17, 1025),
        heads=128,
        pool=2 + (8193 + p - 1) // p + (17 + p - 1) // p + (1025 + p - 1) // p,
        profile="pa02_both",
    )
    for p in (16, 112, 128, 256, 528, 1024)
] + [
    *REPRESENTATIVES[:2],
    spec(48, window=(-1, 1), q=(7, 3), lengths=(97, 3)),
    spec(112, window=(1, -1), q=(7, 3), lengths=(3, 7)),
    spec(128, window=(2, 7), causal=True, q=(7, 3), lengths=(257, 129), heads=3),
    spec(1024, window=(0, 0), q=(1, 0), lengths=(8193, 0), heads=128),
]
COST_CASES = [PERFORMANCE_CASES[0]]
ACCURACY_CASES = [s for s in PERFORMANCE_CASES if s not in REPRESENTATIVES + COST_CASES] + [
    spec(112, window=(3, 1), q=(513, 259), lengths=(257, 129), uq=(513, 257), heads=3),
    spec(48, q=(7, 3), lengths=(3, 129), uq=(0, 0)),
    spec(16, window=(10000, 10000)),
    spec(112, window=(-1, 0)),
]


# Only stage performance changes this fixed candidate set; correctness lists above stay intact.

PERFORMANCE_CASES = window_public_cases(PERFORMANCE_CASES, "paged-tnd")
