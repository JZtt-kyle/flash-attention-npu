"""Named partial-attention inputs; each exercises a distinct dispatch/numeric boundary."""

CHECK_CASES = [
    dict(id="nonzero-start-tail", shape=(1, 3, 8, 1031), splits=2, m=16, n=512),
    dict(
        id="band-row-empty",
        shape=(2, 9, 4, 1057),
        lengths=[1057, 3],
        splits=8,
        window=(530, 1),
        m=17,
        n=128,
    ),
    dict(id="causal-tail", shape=(1, 17, 8, 1031), splits=2, causal=True, m=129, n=512),
]
COST_CASES = [
    dict(id="longest-kv", shape=(1, 3, 128, 16384), splits=2, m=256, n=512),
]
PERFORMANCE_CASES = [CHECK_CASES[0], CHECK_CASES[2], COST_CASES[0]]
BOUNDARY_CASES = [
    dict(id="one-whole-fragment", shape=(1, 2, 8, 513), splits=1, m=16, n=512),
    dict(id="256-splits", shape=(1, 1, 8, 16383), splits=256, m=16, n=512),
    dict(id="many-m-many-cores", shape=(2, 33, 8, 1057), splits=2, m=16, n=512, cores=3),
    dict(id="ragged-m255", shape=(2, 33, 8, 1025), lengths=[1025, 0], splits=8, m=255, n=512),
    dict(id="n32-both-sides", shape=(1, 17, 2, 65), splits=2, window=(33, 1), m=16, n=32),
    dict(id="n-tail-minus", shape=(1, 2, 16, 511), splits=2, m=32, n=128),
    dict(id="m-tail-one", shape=(1, 1, 1, 33), splits=2, m=16, n=32),
    dict(id="left-unbounded", shape=(1, 9, 8, 1057), splits=2, window=(-1, 1), m=128, n=512),
    dict(id="right-unbounded", shape=(1, 9, 8, 1057), splits=2, window=(530, -1), m=128, n=512),
    dict(id="zero-scale", shape=(1, 3, 8, 1025), splits=2, scale=0.0, m=16, n=512),
    dict(
        id="negative-scale-rope",
        shape=(1, 3, 8, 1025),
        splits=2,
        scale=-0.23,
        perturb=2,
        m=16,
        n=512,
    ),
    dict(id="content-upper", shape=(1, 3, 8, 1025), splits=2, perturb=1, m=16, n=512),
    dict(id="amla-max-change", shape=(1, 2, 8, 2049), splits=2, max_change=True, m=16, n=512),
]
EMPTY_CASES = [
    dict(id="zero-kv", shape=(2, 3, 8, 0), splits=2, m=16, n=512),
    dict(
        id="all-zero-used", shape=(2, 3, 8, 513), lengths=[0, 0], splits=8, causal=True, m=16, n=512
    ),
    dict(id="causal-full-mask-blocks", shape=(1, 33, 8, 3), splits=8, causal=True, m=16, n=32),
    dict(id="local-full-mask-blocks", shape=(1, 33, 8, 3), splits=2, window=(0, 0), m=16, n=32),
]
