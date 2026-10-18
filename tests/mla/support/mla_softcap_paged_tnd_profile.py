"""TND PA adapters for the validated softcap public timing protocol."""

from tests.mla.support import mla_softcap_mixed_profile as shared
from tests.mla.support.public_cases.softcap_paged_tnd import make_case, metadata, invoke, compare

PERFORMANCE = [
    dict(
        id="compressed-single-none",
        representation="compressed",
        splits=1,
        mask="none",
        q=(17, 3),
        k=(1057, 513),
        page=16,
    ),
    dict(
        id="compressed-explicit-window",
        representation="compressed",
        splits=3,
        mask="window",
        q=(17, 3),
        k=(1057, 513),
        page=48,
        strided=True,
        used_k=(1025, 7),
    ),
    dict(
        id="compressed-auto-causal",
        representation="compressed",
        splits=0,
        mask="causal",
        q=(1, 1),
        k=(16384, 513),
        page=1024,
    ),
    dict(
        id="prefill-single-causal",
        representation="prefill",
        splits=1,
        mask="causal",
        q=(17, 3),
        k=(1057, 513),
        page=48,
    ),
    dict(
        id="prefill-auto-window",
        representation="prefill",
        splits=0,
        mask="window",
        q=(17, 3),
        k=(1057, 513),
        page=48,
        strided=True,
        used_k=(1025, 7),
    ),
]
MODES = shared.MODES


def partition_rows(rows, raw, labels, representation, dtype, spec):
    # Validate actual symbols, including C++ ABI lengths, without rewriting rows.
    return shared.partition_rows(rows, raw, labels, representation, dtype, spec, layout="paged_tnd")


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
        layout="TND PA cache",
    )
