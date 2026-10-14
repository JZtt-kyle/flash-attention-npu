"""Explicit daily public behaviors. Growth follows capabilities, not a quota."""

# capability: (physical layout, mask)
BEHAVIORS = {
    "dense": ("func", "none"),
    "bsnd": ("bsnd", "none"),
    "tnd": ("tnd", "none"),
    "mixed": ("mixed", "none"),
    "paged_bsnd": ("paged_bsnd", "none"),
    "paged_tnd": ("paged_tnd", "none"),
    "causal_bsnd": ("bsnd", "causal"),
    "causal_tnd": ("tnd", "causal"),
    "causal_mixed": ("mixed", "causal"),
    "causal_paged": ("paged_bsnd", "causal"),
    "causal_paged_tnd": ("paged_tnd", "causal"),
    "window_bsnd": ("bsnd", "window"),
    "window_tnd": ("tnd", "window"),
    "window_mixed": ("mixed", "window"),
    "window_paged": ("paged_bsnd", "window"),
    "window_paged_tnd": ("paged_tnd", "window"),
}
SCENARIOS = ("lengths", "strided_reuse", "empty")
COMPILE_BEHAVIORS = ("dense", "causal_tnd", "window_mixed", "causal_paged_tnd", "window_paged_tnd")


def daily_cases():
    return [
        (name, dtype, scenario)
        for name in BEHAVIORS
        for dtype in ("fp16", "bf16")
        for scenario in SCENARIOS
    ]


def daily_suites():
    path = "tests/mla/e2e/test_regression.py"
    suites = {
        name: [
            f"{path}::test_public_behavior[{'-'.join(case)}]"
            for case in daily_cases()
            if case[0] == name
        ]
        for name in BEHAVIORS
    }
    for name in suites:
        suites[name].append(f"{path}::test_public_fake[{name}]")
        if name in COMPILE_BEHAVIORS:
            suites[name].append(f"{path}::test_public_compile[{name}]")
    suites["dense"] += [
        f"{path}::test_unopened_options[{name}]" for name in ("num_splits", "softcap")
    ]
    split = "tests/mla/e2e/split_kv_bsnd/test_public.py"
    suites["split_kv_bsnd"] = [
        f"{split}::test_daily",
        f"{split}::test_empty",
        f"{split}::test_fake_and_compile",
    ]
    auto = "tests/mla/e2e/auto_split_bsnd/test_public.py"
    suites["auto_split_bsnd"] = [
        f"{auto}::test_daily",
        f"{auto}::test_empty",
        f"{auto}::test_fake_and_compile",
    ]
    tnd_split = "tests/mla/e2e/split_tnd/test_public.py"
    suites["split_tnd"] = [
        f"{tnd_split}::test_daily",
        f"{tnd_split}::test_empty",
        f"{tnd_split}::test_fake",
        f"{tnd_split}::test_compile",
    ]
    return suites
