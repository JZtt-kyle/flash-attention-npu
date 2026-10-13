"""Deterministic daily case limits, independent of test or timing results."""

from collections import defaultdict
import hashlib
import json


DEFAULT_MAX_CASES = 100
PUBLIC_MAX_CASES = 100
POLICY_VERSION = 1


def rank(value):
    return hashlib.sha256(("mla-daily-v1:" + value).encode()).hexdigest()


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
        function, _, parameters = nodeid.partition("[")
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


def public_cases(specs):
    """Both dtypes use the same fixed inputs, at most 100 configurations total."""
    limit = PUBLIC_MAX_CASES // 2
    if len(specs) <= limit:
        return list(specs)
    identities = [json.dumps(spec, sort_keys=True) for spec in specs]
    selected = set(sorted(range(len(specs)), key=lambda i: rank(identities[i]))[:limit])
    return [spec for i, spec in enumerate(specs) if i in selected]
