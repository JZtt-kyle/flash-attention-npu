"""Fixed window/PA inputs chosen before timing; all legal pages are tested separately."""


def config(
    page=112,
    profile="independent",
    q=(3, 3),
    lengths=None,
    uq=(3, 1),
    heads=8,
    capacity=None,
    pool=7,
    aic=None,
    repeat=False,
    layout=0,
):
    return dict(
        page=page,
        profile=profile,
        q=list(q),
        lengths=list(lengths if lengths is not None else (page + 1, 2 * page - 3)),
        uq=None if uq is None else list(uq),
        heads=heads,
        capacity=capacity,
        pool=pool,
        aic=aic,
        repeat=repeat,
        layout=layout,
    )


def spec(page=112, window=(2, 1), causal=False, **kwargs):
    return dict(config(page, **kwargs), window=window, causal=causal)


REPRESENTATIVES = [
    spec(112, q=(7, 7), lengths=(3, 225), uq=None, heads=8),
    spec(528, window=(513, 17), q=(257, 257), lengths=(257, 1057), uq=None, heads=1),
    spec(16, window=(17, 2), causal=True, q=(7, 7), lengths=(3, 33), uq=(5, 1), repeat=True),
]
PERFORMANCE_CASES = [
    spec(
        p,
        window=(129, 2),
        q=(3, 3),
        lengths=(8193, 1025),
        uq=None,
        heads=128,
        pool=2 + (8193 + p - 1) // p + (1025 + p - 1) // p,
        profile="pa02_both",
    )
    for p in (16, 112, 128, 256, 528, 1024)
] + [
    *REPRESENTATIVES[:2],
    spec(48, window=(-1, 1), q=(7, 7), lengths=(97, 3), uq=None, heads=8),
    spec(112, window=(1, -1), q=(7, 7), lengths=(3, 7), uq=None, heads=8),
    spec(128, window=(2, 7), causal=True, q=(7, 7), lengths=(257, 129), uq=None, heads=3),
    spec(1024, window=(0, 0), q=(1, 1), lengths=(8193, 0), uq=None, heads=128),
]
ACCURACY_CASES = [s for s in PERFORMANCE_CASES if s not in REPRESENTATIVES] + [
    spec(112, window=(3, 1), q=(513, 513), lengths=(257, 129), uq=(513, 257), heads=3),
    spec(48, q=(7, 7), lengths=(3, 129), uq=(0, 0)),
    spec(16, window=(10000, 10000), q=(7, 7), lengths=(3, 33), uq=None),
    spec(112, window=(-1, 0), q=(7, 7), lengths=(3, 225), uq=None),
]
COST_CASES = [PERFORMANCE_CASES[0]]
ACCURACY_CASES = [s for s in ACCURACY_CASES if s not in COST_CASES]
