"""M10's fixed performance subset, selected by coverage before measurement."""

import json
import os
from pathlib import Path


SPLIT_STAGE_POLICY = 2
# suite: (layout, split mode, original candidate indices)
# Single-split probes retain length/stride/page and masked boundaries. Each
# explicit/automatic set retains causal, strided window and long unmasked KV.
SPLIT_STAGE_SUITES = {
    "window-bsnd-profile": ("bsnd", "single", (1, 6, 9)),
    "window-tnd-profile": ("tnd", "single", (1, 9, 11)),
    "window-mixed-profile": ("mixed", "single", (0, 7, 10)),
    "window-paged-profile": ("paged-bsnd", "single", (0, 6, 10)),
    "window-paged-tnd-profile": ("paged-tnd", "single", (0, 6, 10)),
    # used=512 retains a non-FD native representative; the other three are FD.
    "split-bsnd-profile": ("bsnd", "explicit", (2, 4, 5, 6)),
    "split-tnd-profile": ("tnd", "explicit", (1, 3, 4)),
    "split-mixed-profile": ("mixed", "explicit", (1, 3, 4)),
    "split-paged-bsnd-profile": ("paged-bsnd", "explicit", (1, 3, 4)),
    "split-paged-tnd-profile": ("paged-tnd", "explicit", (1, 3, 4)),
    "auto-split-bsnd-profile": ("bsnd", "automatic", (2, 5, 6)),
    "auto-split-tnd-profile": ("tnd", "automatic", (1, 3, 4)),
    "auto-split-mixed-profile": ("mixed", "automatic", (1, 3, 4)),
    "auto-split-paged-bsnd-profile": ("paged-bsnd", "automatic", (1, 3, 4)),
    "auto-split-paged-tnd-profile": ("paged-tnd", "automatic", (1, 3, 4)),
}


def split_stage_selection(specs, suite):
    layout, mode, indices = SPLIT_STAGE_SUITES[suite]
    candidates = json.loads(json.dumps(specs))
    identities = [json.dumps(spec, sort_keys=True) for spec in candidates]
    if len(identities) != len(set(identities)) or max(indices) >= len(candidates):
        raise ValueError("Incomplete or duplicate M10 performance candidates")
    return dict(
        policy=SPLIT_STAGE_POLICY,
        suite=suite,
        layout=layout,
        split_mode=mode,
        dtypes=["fp16", "bf16"],
        indices=list(indices),
        candidates=candidates,
        selected=[s for i, s in enumerate(candidates) if i in indices],
        deselected=[s for i, s in enumerate(candidates) if i not in indices],
    )


def split_stage_cases(specs, suite):
    if os.environ.get("MLA_SPLIT_STAGE_PERFORMANCE") != "1":
        return list(specs)
    selection = split_stage_selection(specs, suite)
    directory = Path(os.environ["MLA_VALIDATION_RESULT"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"split-stage-selection-{suite}.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    return selection["selected"]


def check_split_stage_dispatch(cases):
    """Reject a stage group with no applicable native cases before timing."""
    if os.environ.get("MLA_SPLIT_STAGE_PERFORMANCE") != "1":
        return
    suite = os.environ.get("MLA_SPLIT_STAGE_SUITE")
    if suite not in SPLIT_STAGE_SUITES:
        raise ValueError("Missing M10 performance suite for native dispatch")
    automatic = SPLIT_STAGE_SUITES[suite][1] == "automatic"
    if not any(c["native_template"] == 5 and (automatic or not c["native_fd"]) for c in cases):
        raise ValueError(f"No applicable M10 native cases before timing: {suite}")
