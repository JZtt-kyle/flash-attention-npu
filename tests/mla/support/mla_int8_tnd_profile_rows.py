"""Fixed TND configurations: ragged/decode/FD/empty/maximum-Q and storage views."""

from tests.mla.support.mla_int8_cases import SOURCE_CASES
from tests.mla.support.mla_int8_bsnd_profile_rows import (
    modes as modes,
    summarize as summarize,
    partition_kernels as partition_kernels,
)

PERFORMANCE = tuple(
    dict(layout="paged_tnd", index=i, view="contiguous")
    for i, spec in enumerate(SOURCE_CASES)
    if spec[1] == "TND" or i == 16
) + tuple(
    dict(layout="paged_tnd", index=i, view=view)
    for i, view in (
        (11, "row-gap"),
        (12, "page-gap"),
        (12, "query-gap"),
        (12, "scale-gap"),
        (12, "allocation-gap"),
        (13, "table-gap"),
        (13, "row-gap"),
        (15, "scale-gap"),
    )
)


def selected(check):
    return (
        [
            dict(layout="paged_tnd", index=i, view=view)
            for i, view in (
                (13, "contiguous"),
                (12, "allocation-gap"),
                (15, "scale-gap"),
                (16, "table-gap"),
            )
        ]
        if check
        else PERFORMANCE
    )
