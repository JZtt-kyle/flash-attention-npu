"""TND PA window native/public timing, preserving the PA conversion boundary."""

from tests.mla.support.mla_window_paged_profile import WindowPagedCall

if __name__ == "__main__":
    import sys
    from tests.mla.support.mla_causal_paged_profile import profile
    from tests.mla.support.public_cases import window_paged_tnd_public as suite

    profile(
        *sys.argv[1:],
        layout="tnd",
        suite=suite,
        native_class=WindowPagedCall,
        mask="window",
        metadata_name="ComputeMlaWindowPagedTndMetadata",
    )
