"""M14 uses every existing prefill performance input, fixed before sampling."""

import json

from tests.mla.support.mla_prefill_bsnd_cases import PERFORMANCE_CASES as BSND
from tests.mla.support.mla_prefill_tnd_cases import PERFORMANCE_CASES as TND
from tests.mla.support.mla_prefill_mixed_cases import PERFORMANCE_CASES as MIXED
from tests.mla.support.mla_prefill_paged_bsnd_cases import PERFORMANCE_CASES as PAGED_BSND
from tests.mla.support.mla_prefill_paged_tnd_cases import PERFORMANCE_CASES as PAGED_TND

PREFILL_STAGE_POLICY = "prefill-five-layouts-all-inputs-v1"
PREFILL_STAGE_SUITES = {
    "prefill-bsnd-profile": ("bsnd", BSND),
    "prefill-tnd-profile": ("tnd", TND),
    "prefill-mixed-profile": ("mixed", MIXED),
    "prefill-paged-bsnd-profile": ("paged-bsnd", PAGED_BSND),
    "prefill-paged-tnd-profile": ("paged-tnd", PAGED_TND),
}


def stage_selection():
    specs = {suite: cases for suite, (_, cases) in PREFILL_STAGE_SUITES.items()}
    if not all(specs.values()) or sum(map(len, specs.values())) * 2 > 100:
        raise ValueError("Prefill stage requires nonempty layouts and at most 100 configurations")
    return json.loads(json.dumps(specs))
