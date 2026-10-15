"""Shared helpers extracted from tests/mla/unit/bsnd/test_flash_attn_npu_v3_mla_bsnd_schedule.py.

Independent CPU checks for BSND actual KV lengths; no production/NPU entry.

The oracle compiles pinned CANN SplitCore plus verbatim host length/base/workspace
functions. Native source output and the FA3 num_splits=1 mapping are counted
separately. MLA_LENGTH_SCHEDULE_ARTIFACT_DIR preserves inputs/digests/manifests;
MLA_SCHEDULE_SANITIZE=1 enables ASan/UBSan for both executables.
"""

from tests.mla.support.components.schedule import case


def length_case(lengths=None, *, count=None, **shape):
    values = case(**shape)
    supplied = [] if lengths is None else list(lengths)
    if count is None:
        count = len(supplied)
    return values + [int(lengths is not None), count, len(supplied), *supplied]
