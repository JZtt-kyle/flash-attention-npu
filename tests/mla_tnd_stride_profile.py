"""Profile independent TND/mixed v3/v4/v5 strides against fixed CANN non-FD.

Run as python3 -m tests.mla_tnd_stride_profile PROBES PRODUCER_MANIFEST OUTPUT.
Source packing/materialization precedes timing; this is not public metadata E2E.
"""

import argparse

from tests.mla_bsnd_stride_profile import profile
from tests import test_flash_attn_npu_v3_mla_tnd_stride as tnd_stride


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir")
    parser.add_argument("producer_manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    profile(
        args.probe_dir, args.producer_manifest, args.output, suite=tnd_stride, kernel_prefix="tnd"
    )
