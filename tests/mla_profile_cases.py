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
