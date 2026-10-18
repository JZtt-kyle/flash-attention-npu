"""BSND PA adapters for the validated softcap public timing protocol."""

from tests.mla.support import mla_softcap_mixed_profile as shared
from tests.mla.support.public_cases.softcap_paged_bsnd import make_case, metadata, invoke, compare

PERFORMANCE = [
    dict(
        id="compressed-single-none",
        representation="compressed",
        splits=1,
        mask="none",
        shape=(2, 17, 8, 1057),
        page=16,
    ),
    dict(
        id="compressed-explicit-window",
        representation="compressed",
        splits=3,
        mask="window",
        shape=(2, 17, 8, 1057),
        page=48,
        strided=True,
        lengths=[1025, 7],
    ),
    dict(
        id="compressed-auto-causal",
        representation="compressed",
        splits=0,
        mask="causal",
        shape=(2, 1, 8, 16384),
        page=1024,
        lengths=[16384, 513],
    ),
    dict(
        id="prefill-single-causal",
        representation="prefill",
        splits=1,
        mask="causal",
        shape=(2, 17, 8, 1057),
        page=48,
    ),
    dict(
        id="prefill-auto-window",
        representation="prefill",
        splits=0,
        mask="window",
        shape=(2, 17, 8, 1057),
        page=48,
        strided=True,
        lengths=[1025, 7],
    ),
]
MODES = shared.MODES


def partition_rows(rows, raw, labels, representation, dtype, spec):
    # Validate actual symbols, including C++ ABI lengths, without rewriting rows.
    return shared.partition_rows(
        rows, raw, labels, representation, dtype, spec, layout="paged_bsnd"
    )


def profile(representation, dtype, directory, *, check=False):
    return shared.profile(
        representation,
        dtype,
        directory,
        check=check,
        cases=PERFORMANCE,
        make_case=make_case,
        metadata=metadata,
        invoke=invoke,
        compare=compare,
        partition=partition_rows,
        layout="BSND PA cache",
    )
