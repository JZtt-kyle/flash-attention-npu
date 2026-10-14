"""Each named tuple has a distinct merge, mask, address or resource purpose."""

CHECK_CASES = [
    dict(id="merge-tail", shape=(1, 3, 8, 1031), splits=2, m=16, n=512),
    dict(id="causal-partial-empty", shape=(1, 17, 8, 1031), splits=2, causal=True, m=129, n=512),
    dict(
        id="band-partial-empty",
        shape=(2, 9, 4, 1057),
        lengths=[1057, 3],
        splits=8,
        window=(530, 1),
        m=17,
        n=128,
    ),
]
COST_CASES = [
    dict(id="longest-kv", shape=(1, 3, 128, 16384), splits=2, m=256, n=512),
    dict(id="256-explicit", shape=(1, 1, 8, 16383), splits=256, m=16, n=512),
]
BOUNDARY_CASES = [
    dict(id="one-no-accumulator", shape=(1, 2, 8, 513), splits=1, m=16, n=512),
    dict(id="ragged-m255", shape=(2, 33, 8, 1025), lengths=[1025, 0], splits=8, m=255, n=512),
    dict(id="m128-head-boundary", shape=(1, 17, 8, 1025), splits=8, m=128, n=512),
    dict(id="single-row", shape=(1, 1, 1, 33), splits=2, m=16, n=32),
    dict(id="zero-kv", shape=(2, 3, 8, 0), splits=2, m=16, n=512),
    dict(id="fully-masked-blocks", shape=(1, 33, 8, 3), splits=8, causal=True, m=16, n=32),
    dict(id="fully-masked-local", shape=(1, 33, 8, 3), splits=2, window=(0, 0), m=16, n=32),
    dict(id="zero-scale", shape=(1, 3, 8, 1025), splits=2, scale=0.0, m=16, n=512),
    dict(id="negative-scale", shape=(1, 3, 8, 1025), splits=2, scale=-0.23, perturb=2, m=16, n=512),
    dict(id="amla-max-change", shape=(1, 2, 8, 2049), splits=2, max_change=True, m=16, n=512),
]
# Source's native 6 KiB / uint8 repeat boundary versus the explicit extension;
# equal maxima also detect skipped split 255 and incorrect re-normalization.
SYNTHETIC_CASES = [
    dict(id="weight-span", splits=8, rows=17, mode="span"),
    dict(id="equal-maxima", splits=2, rows=9, mode="equal"),
    dict(id="source-buffer-boundary", splits=192, rows=1, mode="equal"),
    dict(id="repeat-255-boundary", splits=255, rows=1, mode="equal"),
    dict(id="repeat-256-boundary", splits=256, rows=1, mode="last"),
    dict(id="valid-below-mask-sentinel", splits=8, rows=9, mode="sentinel"),
    dict(id="empty-poisoned-out", splits=8, rows=9, mode="empty"),
]
PERFORMANCE_CASES = [CHECK_CASES[0], CHECK_CASES[2], COST_CASES[0]]
