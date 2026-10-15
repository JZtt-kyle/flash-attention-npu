from tests.mla.support.components.split_kv_merge import merge_probes as merge_probes
from tests.mla.support.mla_split_kv_merge_profile import profile


def test_component_profile(merge_probes):  # noqa: F811
    profile(merge_probes, merge_probes["directory"] / "profile")
