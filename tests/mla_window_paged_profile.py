"""Native ND PA band mask with unchanged public conversion/timing boundaries."""

from tests.mla_paged_profile import CannPagedCall
from tests.common.mla_window_paged_reference import window_for


class WindowPagedCall(CannPagedCall):
    def __init__(self, case):
        super().__init__(
            case["args"], case["config"]["lengths"], case["scale"], case["config"]["q"]
        )
        kind, left, right = window_for(case)
        self.kwargs.update(sparse_mode=0)
        if kind:
            self.kwargs.update(
                sparse_mode=3 if kind == 1 else 4,
                atten_mask=case["mask"],
                pre_tokens=2**31 - 1 if kind == 1 else left,
                next_tokens=0 if kind == 1 else right,
            )

    def description(self):
        return {name: self.kwargs[name] for name in ("sparse_mode", "pre_tokens", "next_tokens")}


if __name__ == "__main__":
    import sys
    from tests.mla_causal_paged_profile import profile
    from tests import test_flash_attn_npu_v3_mla_window_paged_public as suite

    profile(
        *sys.argv[1:],
        suite=suite,
        native_class=WindowPagedCall,
        mask="window",
        metadata_name="ComputeMlaWindowPagedBsndMetadata",
    )
