"""BSND PA explicit split inputs fixed before correctness and timing runs."""

from tests.mla.support.mla_split_stage_cases import split_stage_cases


def spec(
    page=112,
    q=7,
    lengths=(225, 3),
    heads=3,
    splits=8,
    window=(2, 1),
    causal=False,
    used=None,
    capacity=None,
    stride=False,
    scale=None,
):
    capacity = (
        ((max(lengths, default=0) + page - 1) // page) * page if capacity is None else capacity
    )
    return dict(
        page=page,
        q=q,
        lengths=list(lengths),
        heads=heads,
        splits=splits,
        window=window,
        causal=causal,
        used=used,
        capacity=capacity,
        stride=stride,
        scale=scale,
    )


DAILY_CASES = [
    spec(page=16, q=3, lengths=(17, 0), heads=2, splits=2, window=(-1, -1)),
    spec(page=112, q=7, lengths=(3, 129), heads=2, causal=True, window=(-1, -1)),
    spec(page=528, q=5, lengths=(529, 0), heads=3, window=(2, 1), stride=True),
]
REPRESENTATIVES = [
    spec(
        page=112,
        q=65,
        lengths=(1025, 511),
        heads=4,
        splits=8,
        window=(530, 1),
        used=[63, 1],
        stride=True,
    ),
    spec(page=528, q=7, lengths=(1057, 3), heads=2, splits=256, causal=True, window=(-1, -1)),
    spec(page=1024, q=2, lengths=(16384, 8193), heads=2, splits=2, window=(-1, -1)),
    spec(page=16, q=65, lengths=(513, 17), heads=4, splits=256, window=(513, 7)),
]
BOUNDARIES = [
    spec(page=512, q=3, lengths=(511, 513), splits=2, window=(-1, -1)),
    spec(page=112, q=257, lengths=(449, 0), heads=1, splits=193, window=(0, 0)),
    spec(page=528, q=3, lengths=(529, 3), splits=255, window=(-1, 1)),
    spec(page=16, q=7, lengths=(3, 33), used=[0, 0]),
    spec(page=112, q=7, lengths=(0, 0), capacity=224),
    spec(page=16, q=3, lengths=(17, 1), heads=257, splits=2, window=(1, -1)),
    spec(page=16, q=128, lengths=(33,), heads=1, splits=2, window=(2, 1)),
    spec(page=16, q=249, lengths=(33,), heads=1, splits=2, causal=True, window=(-1, -1)),
]
PERFORMANCE_CASES = [
    spec(page=16, q=3, lengths=(8193, 1025), heads=16, splits=2, window=(-1, -1)),
    spec(page=112, q=7, lengths=(17, 1057), heads=16, splits=8, causal=True, window=(-1, -1)),
    spec(page=528, q=65, lengths=(1057, 513), heads=4, splits=8, window=(513, 7)),
    spec(page=512, q=7, lengths=(511, 17), heads=16, splits=2, window=(17, 1), stride=True),
    spec(page=1024, q=3, lengths=(16384, 17), heads=2, splits=256, window=(-1, -1)),
    spec(page=112, q=7, lengths=(0, 0), heads=4, splits=8, capacity=224),
]


# Stage selection changes only performance; all correctness cases above stay intact.
PERFORMANCE_CASES = split_stage_cases(PERFORMANCE_CASES, "split-paged-bsnd-profile")
