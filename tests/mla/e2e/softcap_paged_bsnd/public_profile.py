"""Explicit public timing only; excluded from daily/detail correctness collection."""

import os
from pathlib import Path
import pytest
from tests.mla.support.mla_softcap_paged_bsnd_profile import profile


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_profile(representation, dtype):
    check = os.environ.get("MLA_SOFTCAP_PAGED_BSND_PROFILE_CHECK") == "1"
    report = profile(
        representation,
        dtype,
        Path(os.environ["MLA_VALIDATION_RESULT"]) / f"{representation}-{dtype}",
        check=check,
    )
    assert report["status"] == ("fixtures-passed" if check else "sampled")
