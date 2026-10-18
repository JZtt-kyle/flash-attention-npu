"""Fixed graph/eager inputs and the existing public equal-weight timing format."""

from decimal import Decimal

from tests.mla.support.mla_gqa_options_profile_rows import modes, segments, summarize

PERFORMANCE = [
    dict(layout=layout, **spec)
    for layout in ("bsnd", "tnd", "mixed", "paged_bsnd", "paged_tnd")
    for spec in (
        dict(id="legacy-single-none", pack=None, splits=1, q=(7, 3), k=(65, 33)),
        dict(
            id="unpacked-auto-window",
            pack=False,
            margin=1,
            splits=0,
            mask="window",
            q=(3, 1),
            k=(4097, 1057),
            window=(256, 2),
        ),
    )
]


def selected(check):
    # Check both launch families in every native layout with bounded inputs.
    return [dict(s, q=(3, 1), k=(65, 33)) for s in PERFORMANCE] if check else PERFORMANCE


def select_model_rows(rows, raw):
    if len(rows) != len(raw):
        raise ValueError("Missing raw model task identities")
    result = [
        dict(row, model_id=int(source["Model ID"]))
        for row, source in zip(rows, raw)
        if 0 <= int(source["Model ID"]) < 0xFFFFFFFF
    ]
    if not result:
        raise ValueError("No captured model tasks in profiler output")
    return result


def partition_kernels(rows, labels, specs, dtype, representation):
    pos, result = 0, []
    model_ids = {}
    dt = {"fp16": "Fp16", "bf16": "Bf16"}[dtype]
    for label in labels:
        spec = specs[label["case"]]
        ours = label["name"] == "fa3"
        legacy = spec["pack"] is None
        if not legacy:
            prefix = "MlaGqa" + dt
            expected = numerical = [prefix + "Attention"]
        elif representation == "prefill":
            expected = ["MlaPrefillTnd" + dt + "Init"] if spec["layout"] != "bsnd" else []
            expected += ["MlaPrefillBsnd" + dt + "Attention"]
            numerical = expected[-1:]
        else:
            prefix = {
                "bsnd": "mla_dense",
                "tnd": "mla_tnd",
                "mixed": "mla_tnd_bsnd",
                "paged_bsnd": "mla_paged_bsnd",
                "paged_tnd": "mla_paged_tnd",
            }[spec["layout"]]
            expected = numerical = [prefix + "_" + dtype]
        group = []
        while pos < len(rows):
            row = rows[pos]
            pos += 1
            group.append(row)
            if (
                (expected[-1] in row["Name"])
                if ours
                else row["Name"].endswith("_FusedInferAttentionScore")
            ):
                break
        else:
            raise ValueError(f"Missing graph/eager terminal task for {label}: {expected[-1]}")
        if "model_id" in group[0]:
            identities = {r["model_id"] for r in group}
            key = (label["case"], label["name"])
            if (
                len(identities) != 1
                or model_ids.setdefault(key, group[0]["model_id"]) not in identities
            ):
                raise ValueError("Mixed or changed captured model tasks")
        if ours:
            if not legacy and any(
                prefix in r["Name"] and expected[0] not in r["Name"] for r in group
            ):
                raise ValueError("Unexpected separate MLA phase in fused invocation")
            kernels = [r for r in group if any(name in r["Name"] for name in numerical)]
            phases = [r for r in group if any(name in r["Name"] for name in expected)]
            if len(phases) != len(expected) or not all(
                n in r["Name"] for n, r in zip(expected, phases)
            ):
                raise ValueError("Missing or repeated graph/eager phase")
            producers = [r for r in group if "ComputeMla" in r["Name"]]
            if len(producers) != int(spec["layout"] == "tnd"):
                raise ValueError("Changed metadata lifecycle in reuse sample")
        else:
            if any("ComputeMla" in r["Name"] or "MlaGqa" in r["Name"] for r in group):
                raise ValueError("Foreign MLA work in native sample")
            kernels = [group[-1]]
        for a, b in zip(group, group[1:]):
            if a["Device_id"] != b["Device_id"] or Decimal(a["Start Time(us)"]) + Decimal(
                a["Duration(us)"]
            ) > Decimal(b["Start Time(us)"]):
                raise ValueError("Graph/eager phases overlap or change device")
        result.append(
            dict(
                **label,
                tasks=group,
                kernel_us=sum(float(r["Duration(us)"]) for r in kernels),
                adaptation_us=sum(float(r["Duration(us)"]) for r in group if r not in kernels),
            )
        )
    if pos != len(rows):
        raise ValueError("Unclaimed graph/eager profiler tasks")
    if len(set(model_ids.values())) != len(model_ids):
        raise ValueError("Different captured calls share a model identity")
    return result


__all__ = ["PERFORMANCE", "selected", "modes", "segments", "summarize", "partition_kernels"]
