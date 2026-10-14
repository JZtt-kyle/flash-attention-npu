"""Fixed window inputs chosen before execution or timing; every timed case has math coverage."""

CHECK_CASES = [
    dict(shape=(2, 7, 17, 33), lengths=[3, 0], window=(1, 1)),
    dict(shape=(1, 257, 3, 1025), window=(7, 11), cores=1, m=129),
    dict(shape=(2, 513, 3, 257), lengths=[257, 129], window=(-1, 2)),
    dict(shape=(1, 129, 17, 1025), window=(13, -1), cores=1),
    dict(shape=(1, 7, 129, 7), window=(0, 0)),
    dict(shape=(2, 7, 3, 33), lengths=[17, 33], window=(5, 9), causal=True),
]
COST_CASES = [
    dict(shape=(1, 1, 128, 16384), window=(513, 0)),
    dict(shape=(10, 3, 86, 8193), window=(1023, 1)),
]
PERFORMANCE_CASES = [
    CHECK_CASES[1],
    CHECK_CASES[2],
    CHECK_CASES[3],
    CHECK_CASES[5],
    *COST_CASES,
    dict(shape=(2, 129, 17, 1025), window=(511, 1)),
    dict(shape=(21, 7, 3, 513), window=(31, 1)),
]
BOUNDARY_CASES = [
    *[
        dict(shape=(1, 7, h, 33), window=(left, right), cores=1)
        for h, left, right in ((1, 0, 0), (3, 1, 0), (15, 0, 2), (73, 7, -1), (171, -1, 1))
    ],
    *[
        dict(shape=(1, 171, 3, sk), window=(17, 3), m=m, cores=1)
        for m, sk in ((128, 511), (129, 512), (255, 513), (256, 1025), (129, 33))
    ],
    dict(shape=(2, 7, 17, 33), lengths=[3, 33], window=(32, 6)),
    dict(shape=(2, 7, 17, 33), lengths=[3, 33], window=(31, 5)),
    dict(shape=(2, 7, 17, 33), lengths=[3, 33], window=(-1, -1), causal=True),
    dict(shape=(1, 1, 17, 33), window=(-1, -1), causal=True),
    dict(shape=(1, 1, 17, 33), window=(0, -1), causal=True),
    dict(shape=(1, 7, 17, 33), window=(3, 9), max_q=17, max_k=65),
    dict(shape=(1, 7, 17, 33), window=(2**63 - 1, 2**63 - 1)),
    *[
        dict(shape=(2, 7, 17, 1025), lengths=[3, 1025], window=(513, 2), scale=scale, perturb=i)
        for i, scale in enumerate((None, 0.0, -0.23, 0.23))
    ],
    *[
        dict(
            shape=(2, 129, 3, 1025),
            lengths=[17, 1025],
            window=(513, 7),
            amplitude=amplitude,
            max_change=True,
        )
        for amplitude in (0.125, 4.0, 8.0)
    ],
]
EMPTY_CASES = [
    dict(shape=(0, 7, 8, 33), window=(1, 1)),
    dict(shape=(2, 0, 8, 33), window=(1, 1)),
    dict(shape=(2, 7, 8, 0), window=(1, 1)),
    dict(shape=(2, 7, 8, 33), lengths=[0, 0], window=(1, 1)),
    dict(shape=(41, 7, 73, 513), lengths=([0, 1, 3, 513] * 10 + [0]), window=(2, 1)),
    dict(shape=(5, 171, 3, 513), lengths=[0, 17, 0, 513, 0], cores=1, window=(3, 2)),
]
