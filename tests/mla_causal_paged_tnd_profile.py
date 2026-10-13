"""TND PA causal shares the fixed timing/dispatch protocol with BSND."""

import sys

from tests.mla_causal_paged_profile import profile


if __name__ == "__main__":
    profile(*sys.argv[1:], layout="tnd")
