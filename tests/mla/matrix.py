"""Representative parameter coverage, without a numerical case budget.

Keep every distinct validation value and protocol pair for each dtype. Shape
axes retain every boundary and extreme corner without multiplying all tails.
Explicit hand-written tuple cases stay indivisible.
The exhaustive page sweep and duplicate internal API adapters are diagnostics.
"""

from itertools import combinations
import os


# Minimum, non-power-of-two, either side of 128/512, and maximum legal page.
PAGES = (16, 112, 128, 144, 496, 512, 528, 1024)
DIMENSIONS = {"m", "sq", "sk", "s1", "s2", "seqlen_q", "seqlen_k", "batch", "heads"}
ADAPTERS = {
    "public",
    "op",
    "eager",
    "cpp",
    "dispatch",
    "fake",
    "public_fake",
    "op_fake",
    "eager_fake",
    "dispatch_fake",
    "fake-public",
    "fake-op",
}


def full_matrix():
    return os.environ.get("MLA_FULL_MATRIX") == "1"


def covering_rows(rows, *, contexts=(), independent=()):
    """Choose existing rows that cover all available singles/pairs per context.

    Rows contain parameter indices, not tensors. Existing order breaks ties;
    retained cases keep pytest's IDs, marks, fixtures and reproducible RNG seed.
    """
    if len(rows) < 2:
        return list(range(len(rows)))
    keys = sorted(set(rows[0]) - set(contexts))
    extremes = {key: {min(r[key] for r in rows), max(r[key] for r in rows)} for key in independent}
    bits, masks = {}, []
    for row in rows:
        context = tuple((key, row[key]) for key in contexts)
        values = [(key, row[key]) for key in keys]
        tokens = [context + (value,) for value in values]
        tokens += [
            context + pair
            for pair in combinations(values, 2)
            if not all(key in independent for key, _ in pair)
            or all(value in extremes[key] for key, value in pair)
        ]
        if not tokens:
            tokens = [context]
        mask = 0
        for token in tokens:
            if token not in bits:
                bits[token] = len(bits)
            mask |= 1 << bits[token]
        masks.append(mask)
    remaining = (1 << len(bits)) - 1
    chosen = []
    while remaining:
        index = max(range(len(rows)), key=lambda i: (masks[i] & remaining).bit_count())
        chosen.append(index)
        remaining &= ~masks[index]
    return sorted(chosen)


def compact_calls(calls, *, public=False):
    if not calls:
        return []
    keep = list(calls)
    for name in calls[0].params:
        values = {str(call.params[name]) for call in calls}
        if name in {"page", "page_size"} and values == {str(p) for p in range(16, 1025, 16)}:
            keep = [call for call in keep if call.params[name] in PAGES]
        if public and name in {"entry", "kind", "metadata_entry", "forward_entry"}:
            if values <= ADAPTERS:
                primary = "public" if "public" in values else "eager" if "eager" in values else None
                fake = next(
                    (
                        v
                        for v in ("public_fake", "fake-public", "fake", "eager_fake")
                        if v in values
                    ),
                    None,
                )
                if primary:
                    keep = [call for call in keep if call.params[name] in {primary, fake}]
    contexts = [
        name
        for name in calls[0].params
        if {str(call.params[name]) for call in calls}
        in ({"fp16", "bf16"}, {"torch.float16", "torch.bfloat16"})
    ]
    selected = covering_rows(
        [call.indices for call in keep],
        contexts=contexts,
        independent=DIMENSIONS.intersection(calls[0].params),
    )
    return [keep[i] for i in selected]
