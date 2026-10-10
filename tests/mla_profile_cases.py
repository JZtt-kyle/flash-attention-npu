"""Fixed MLA performance cases shared by component and public profilers.

These cases were selected before sampling. Changing the comparison scope does
not remove cases on which native CANN automatically chooses FlashDecode.
"""

CASES = [
    ((2, 1, 8, 257), None),
    ((2, 1, 8, 257), 0.23),
    ((1, 7, 16, 65), None),
    ((1, 1, 128, 512), None),
    ((2, 2, 128, 511), None),
]


S1G_CASES = [
    ((1, 257, 1, 257), None),
    ((2, 7, 73, 65), None),
    ((1, 4, 128, 512), None),
    ((2, 171, 3, 511), None),
    ((10, 3, 86, 65), None),
    ((1, 129, 128, 33), None),
]


KV_CASES = [
    ((2, 1, 8, 513), None),
    ((2, 1, 8, 513), 0.23),
    ((1, 7, 16, 769), None),
    ((1, 1, 128, 1024), None),
    ((2, 2, 128, 1023), None),
    ((10, 3, 86, 769), None),
    ((21, 171, 3, 1024), None),
]


LONG_KV_CASES = [
    ((2, 1, 8, 1025), None),
    ((2, 1, 8, 8193), None),
    ((2, 1, 8, 8193), 0.23),
    ((1, 7, 16, 8193), None),
    ((1, 1, 128, 16384), None),
    ((2, 2, 128, 4097), None),
    ((10, 3, 86, 8193), None),
    ((21, 1, 8, 16384), None),
]


# M06b4 public BSND: (allocated shape, scale, actual lengths). All cases
# remain in reports, including native FD and all-zero length cases.
BSND_CASES = [
    ((2, 1, 8, 257), None, [17, 129]),
    ((2, 1, 8, 257), 0.23, [17, 129]),
    ((1, 7, 16, 65), None, [33]),
    ((2, 7, 16, 33), None, [17, 0]),
    ((2, 2, 128, 1025), None, [511, 513]),
    ((2, 1, 8, 16384), None, [8193, 16383]),
    ((16, 1, 1, 513), None, [0, 1, 511, 512, 513] * 3 + [0]),
    ((1, 1, 1, 1), None, [0]),
    ((41, 3, 128, 65), None, [0] * 41),
    ((41, 2, 128, 65), None, [0, 17, 0, 33] * 10 + [0]),
    ((2, 10, 128, 513), None, [17, 513]),
]

# M06e1c: same strided views go to both public entries. Native CANN's
# materialization stays inside the timed call; FD dispatch is never excluded
# based on timing. Metadata generation, reuse and R=1/4 are all measured.
STRIDE_CASES = [
    ((2, 1, 8, 257), None, None, "contiguous"),
    ((2, 7, 16, 513), None, None, "bnsd"),
    ((2, 2, 128, 1025), None, [511, 513], "independent"),
    ((2, 7, 16, 33), None, [17, 0], "independent"),
    ((2, 3, 8, 16384), None, [16384, 8193], "independent"),
    ((1, 7, 8, 33), None, None, "matrix_65536"),
    ((1, 7, 8, 33), None, None, "head_65536"),
    ((2, 7, 8, 33), None, None, "broadcast"),
    ((2, 7, 8, 33), 0.23, [17, 0], "overlap"),
    ((2, 7, 8, 33), None, [0, 0], "bnsd"),
]
