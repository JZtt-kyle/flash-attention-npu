"""Deterministic daily case limits, independent of test or timing results."""

from tests.mla.paths import original_nodeid

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path


DEFAULT_MAX_CASES = 100
PUBLIC_MAX_CASES = 100
POLICY_VERSION = 1
CAUSAL_STAGE_POLICY = 1
CAUSAL_STAGE_GROUPS = ("bsnd", "tnd", "mixed", "paged-bsnd", "paged-tnd")
CAUSAL_STAGE_SUITES = dict(
    zip(
        (
            "causal-bsnd-profile",
            "causal-tnd-profile",
            "causal-mixed-profile",
            "causal-paged-profile",
            "causal-paged-tnd-profile",
        ),
        CAUSAL_STAGE_GROUPS,
    )
)


def rank(value):
    return hashlib.sha256(("mla-daily-v1:" + original_nodeid(value)).encode()).hexdigest()


def select_nodeids(nodeids, limit=DEFAULT_MAX_CASES):
    """Round-robin test functions/dtypes; retain the original execution order."""
    if limit < 0:
        raise ValueError("Case limit must be nonnegative")
    if len(nodeids) != len(set(nodeids)):
        raise ValueError("Duplicate collected case IDs")
    if not limit or len(nodeids) <= limit:
        return list(nodeids)
    families = defaultdict(list)
    for nodeid in nodeids:
        function, _, parameters = original_nodeid(nodeid).partition("[")
        dtype = next((d for d in ("fp16", "bf16") if d in parameters), "")
        families[(function, dtype)].append(nodeid)
    queues = [sorted(families[k], key=rank) for k in sorted(families, key=str)]
    chosen = []
    while queues and len(chosen) < limit:
        for queue in queues:
            if len(chosen) == limit:
                break
            chosen.append(queue.pop(0))
        queues = [queue for queue in queues if queue]
    selected = set(chosen)
    return [nodeid for nodeid in nodeids if nodeid in selected]


def select_e2e_nodeids(nodeids, limit=DEFAULT_MAX_CASES):
    """Round-robin module/dtype first, then functions, so later modules survive the cap."""
    if limit < 0 or len(nodeids) != len(set(nodeids)):
        raise ValueError("E2E requires a nonnegative limit and unique case IDs")
    if not limit:
        return list(nodeids)
    nodeids = [node for node in nodeids if "::test_cost_boundary" not in node]
    if len(nodeids) <= limit:
        return list(nodeids)
    modules = defaultdict(list)
    for nodeid in nodeids:
        parts = nodeid.split("::", 1)[0].split("/")
        module = parts[parts.index("e2e") + 1]
        parameters = nodeid.partition("[")[2]
        dtype = next((d for d in ("fp16", "bf16") if d in parameters), "contract")
        modules[(module, dtype)].append(nodeid)
    queues = []
    for key in sorted(modules):
        families = defaultdict(list)
        for node in modules[key]:
            function = node.split("[", 1)[0]
            # Cost boundaries remain explicit acceptance options, not daily regression.
            if "::test_cost_boundary" not in function:
                families[function].append(node)

        def priority(function):
            name = function.split("::")[-1]
            numeric = any(
                token in name
                for token in ("representative", "source", "accuracy", "all_page", "d01_d02")
            )
            return (not numeric, function)

        families = [sorted(families[f], key=rank) for f in sorted(families, key=priority)]
        queue = []
        while families:
            for family in families:
                queue.append(family.pop(0))
            families = [family for family in families if family]
        if queue:
            queues.append(queue)
    chosen = set()
    while queues and len(chosen) < limit:
        for queue in queues:
            if len(chosen) < limit:
                chosen.add(queue.pop(0))
        queues = [queue for queue in queues if queue]
    return [node for node in nodeids if node in chosen]


def stage_selection(specs, group):
    """Allocate ten inputs to each layout, independently of measured timings."""
    if group not in CAUSAL_STAGE_GROUPS:
        raise ValueError("Unknown causal layout")
    limit = PUBLIC_MAX_CASES // (2 * len(CAUSAL_STAGE_GROUPS))
    identities = [json.dumps(spec, sort_keys=True) for spec in specs]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate public inputs")
    families = defaultdict(list)
    for i, spec in enumerate(specs):
        family = (
            (spec["page"],)
            if group.startswith("paged-")
            else (
                spec.get("stride", "contiguous"),
                any(spec.get(k) is not None for k in ("uq", "uk", "used")),
                spec.get("lengths") is not None,
            )
        )
        families[family].append(i)
    queues = [
        sorted(families[k], key=lambda i: rank(identities[i])) for k in sorted(families, key=str)
    ]
    chosen = set()
    while queues and len(chosen) < limit:
        for queue in queues:
            if len(chosen) < limit:
                chosen.add(queue.pop(0))
        queues = [queue for queue in queues if queue]
    return dict(
        policy=CAUSAL_STAGE_POLICY,
        group=group,
        max_configurations=PUBLIC_MAX_CASES,
        dtypes=["fp16", "bf16"],
        candidates=list(specs),
        selected=[s for i, s in enumerate(specs) if i in chosen],
        deselected=[s for i, s in enumerate(specs) if i not in chosen],
    )


def public_cases(specs, *, group=None):
    """Both dtypes use the same fixed inputs, at most 100 configurations total."""
    if group is not None and os.environ.get("MLA_CAUSAL_STAGE_PERFORMANCE") == "1":
        selection = stage_selection(specs, group)
        directory = Path(os.environ["MLA_VALIDATION_RESULT"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"public-stage-selection-{group}.json"
        path.write_text(json.dumps(selection, indent=2) + "\n")
        return selection["selected"]
    limit = PUBLIC_MAX_CASES // 2
    if len(specs) <= limit:
        return list(specs)
    identities = [json.dumps(spec, sort_keys=True) for spec in specs]
    selected = set(sorted(range(len(specs)), key=lambda i: rank(identities[i]))[:limit])
    return [spec for i, spec in enumerate(specs) if i in selected]


def check_public_performance(result):
    # Stage acceptance is the equal configuration mean across all five layouts.
    # The separate stage summarizer requires complete, hashed group evidence.
    if os.environ.get("MLA_CAUSAL_STAGE_PERFORMANCE") != "1":
        assert result["passed"], result["mean_ratio"]
