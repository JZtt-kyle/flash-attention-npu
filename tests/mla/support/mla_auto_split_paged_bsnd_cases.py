"""Fixed native automatic PA inputs; every shape has an independent coverage purpose."""

from tests.mla.support.mla_split_stage_cases import split_stage_cases

from tests.mla.support.mla_split_paged_bsnd_cases import spec as explicit_spec


def spec(**kwargs):
    return explicit_spec(**dict(kwargs, splits=0))


DAILY_CASES = [
    spec(page=16, q=3, lengths=(17, 0), heads=2, window=(-1, -1)),
    spec(page=112, q=7, lengths=(3, 129), heads=2, causal=True, window=(-1, -1)),
    spec(page=528, q=5, lengths=(529, 0), heads=3, window=(2, 1), stride=True),
]
REPRESENTATIVES = [
    spec(page=16, q=3, lengths=(17, 1), heads=257, window=(1, -1)),
    spec(page=512, q=65, lengths=(8193, 1, 0), heads=4, window=(-1, -1)),
    spec(page=112, q=65, lengths=(1025, 511), heads=4, window=(530, 1), used=[63, 1], stride=True),
    spec(page=528, q=257, lengths=(1057, 3), heads=1, causal=True, window=(-1, -1)),
    spec(page=1024, q=2, lengths=(16384, 8193), heads=2, window=(-1, -1)),
]
BOUNDARIES = [
    spec(page=16, q=7, lengths=(3, 33), used=[0, 0]),
    spec(page=112, q=7, lengths=(0, 0), capacity=224),
    spec(page=16, q=128, lengths=(33,), heads=1, window=(2, 1)),
    spec(page=16, q=249, lengths=(33,), heads=1, causal=True, window=(-1, -1)),
    spec(page=512, q=3, lengths=(511, 513), window=(-1, -1)),
    spec(page=528, q=3, lengths=(529, 3), window=(-1, 1)),
    spec(page=16, q=3, lengths=(33, 17) * 16, heads=2, window=(-1, -1)),
]
PERFORMANCE_CASES = [
    spec(page=16, q=3, lengths=(8193, 1025), heads=16, window=(-1, -1)),
    spec(page=112, q=7, lengths=(17, 1057), heads=16, causal=True, window=(-1, -1)),
    spec(page=528, q=65, lengths=(1057, 513), heads=4, window=(513, 7)),
    spec(page=512, q=7, lengths=(511, 17), heads=16, window=(17, 1), stride=True),
    spec(page=1024, q=3, lengths=(16384, 17), heads=2, window=(-1, -1)),
    spec(page=112, q=7, lengths=(0, 0), heads=4, capacity=224),
]


# Stage selection changes only performance; all correctness cases above stay intact.
PERFORMANCE_CASES = split_stage_cases(PERFORMANCE_CASES, "auto-split-paged-bsnd-profile")
