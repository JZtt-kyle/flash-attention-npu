"""Fixed GQA option matrix and strict reuse of the public timing evidence format."""

from decimal import Decimal
import math
import statistics
from tests.mla.support.mla_native_dispatch import e2e_applicable
from tests.mla.support.mla_validation_policy import PUBLIC_MAX_CASES

PERFORMANCE = [
    dict(layout=layout, **spec)
    for layout in ("bsnd", "tnd", "mixed", "paged_bsnd", "paged_tnd")
    for spec in (
        dict(
            id="unpacked-single-window",
            pack=False,
            margin=0,
            splits=1,
            mask="window",
            q=(17, 3),
            k=(1057, 513),
            window=(256, 2),
        ),
        dict(
            id="unpacked-explicit-causal",
            pack=False,
            margin=1,
            splits=3,
            mask="causal",
            q=(17, 3),
            k=(1057, 513),
            strided=True,
        ),
        dict(
            id="packed-auto-window",
            pack=True,
            margin=2,
            splits=0,
            mask="window",
            q=(3, 1),
            k=(4097, 1057),
            window=(256, 2),
        ),
        dict(
            id="unpacked-capped-auto",
            pack=False,
            margin=1,
            splits=0,
            mask="causal",
            q=(17, 3),
            k=(1057, 513),
            cap=15,
        ),
    )
]


# M16 also measures 40 graph inputs. This complementary 56-input collection
# covers every option on BSND/TND and explicit/unpacked + auto/packed calls on
# the other layouts; graph inputs cover their auto/unpacked path. Selection is
# by dispatch/phase coverage, independent of timing results.
SCHEDULE_PERFORMANCE = [
    s
    for s in PERFORMANCE
    if s["layout"] in ("bsnd", "tnd")
    or s["id"] in ("unpacked-explicit-causal", "packed-auto-window")
]


def selected(check, *, schedule=False):
    # Cheap preflight covers every native physical layout and every split/mask.
    if check:
        return [s for i, s in enumerate(PERFORMANCE) if i % 4 == i // 4 % 3]
    return SCHEDULE_PERFORMANCE if schedule else PERFORMANCE


def modes(spec):
    return (
        ["fa3", "native_cann", "implicit_4", "native_cann_4"]
        if spec["layout"] == "tnd"
        else [
            "fa3",
            "native_cann",
            "metadata",
            "generate_1",
            "generate_4",
            "native_cann_4",
            "reuse_4",
            "implicit",
            "implicit_4",
        ]
    )


def segments(spec):
    return (
        {"public_1": ("fa3", "native_cann"), "public_4": ("implicit_4", "native_cann_4")}
        if spec["layout"] == "tnd"
        else {
            "reuse_1": ("fa3", "native_cann"),
            "reuse_4": ("reuse_4", "native_cann_4"),
            "generate_1": ("generate_1", "native_cann"),
            "generate_4": ("generate_4", "native_cann_4"),
            "implicit_1": ("implicit", "native_cann"),
            "implicit_4": ("implicit_4", "native_cann_4"),
        }
    )


def partition_kernels(rows, labels, specs, dtype):
    pos, result = 0, []
    prefix = "MlaGqa" + {"fp16": "Fp16", "bf16": "Bf16"}[dtype]
    for label in labels:
        ours = label["name"] == "fa3"
        group = []
        while pos < len(rows):
            row = rows[pos]
            pos += 1
            group.append(row)
            if (
                (prefix + "Attention" in row["Name"])
                if ours
                else row["Name"].endswith("_FusedInferAttentionScore")
            ):
                break
        else:
            raise ValueError("Missing GQA/native terminal task")
        if ours:
            expected = (
                ["ComputeMlaGqaMetadata"] if specs[label["case"]]["layout"] == "tnd" else []
            ) + [prefix + "Attention"]
            if len(group) != len(expected) or not all(
                n in r["Name"] for n, r in zip(expected, group)
            ):
                raise ValueError("Incomplete MLA metadata/fused attention phases")
            kernels = [group[-1]]
        else:
            if any("MlaGqa" in r["Name"] or "ComputeMla" in r["Name"] for r in group):
                raise ValueError("Foreign GQA task in native sample")
            kernels = [group[-1]]
        for a, b in zip(group, group[1:]):
            if a["Device_id"] != b["Device_id"] or Decimal(a["Start Time(us)"]) + Decimal(
                a["Duration(us)"]
            ) > Decimal(b["Start Time(us)"]):
                raise ValueError("GQA phases overlap or change device")
        result.append(
            dict(
                **label,
                tasks=group,
                kernel_us=sum(float(r["Duration(us)"]) for r in kernels),
                adaptation_us=sum(float(r["Duration(us)"]) for r in group if r not in kernels),
            )
        )
    if pos != len(rows):
        raise ValueError("Unclaimed GQA profiler tasks")
    return result


def summarize(reports, specs=PERFORMANCE):
    if (
        len(specs) * 4 > PUBLIC_MAX_CASES
        or len(reports) != 4
        or {(r["representation"], r["dtype"]) for r in reports}
        != {(r, d) for r in ("compressed", "prefill") for d in ("fp16", "bf16")}
    ):
        raise ValueError("Incomplete GQA matrix or configuration limit exceeded")
    cases, excluded = [], []
    for report in reports:
        if (
            report["warmup"],
            report["rounds"],
            report["wall_iterations"],
            report["profiler_iterations"],
        ) != (5, 5, 20, 10):
            raise ValueError("Changed timing protocol")
        if len(report["results"]) != len(specs):
            raise ValueError("Incomplete GQA cases")
        for i, (spec, row) in enumerate(zip(specs, report["results"])):
            import json

            if row["spec"] != json.loads(json.dumps(spec)):
                raise ValueError("Changed GQA input identity")
            applicable = not spec.get("cap", 0) and e2e_applicable(
                row["native_template"],
                row["native_fd"],
                representation=report["representation"],
                include_native_fd=spec["splits"] == 0,
            )
            if row["e2e_applicable"] != applicable:
                raise ValueError("Changed native classification")
            ratios = {}
            for name, (a, b) in segments(spec).items():
                a, b = row[a]["median_e2e_us"], row[b]["median_e2e_us"]
                if not all(math.isfinite(x) and x > 0 for x in (a, b)):
                    raise ValueError("Nonpositive or invalid timing")
                ratios[name] = a / b
            item = dict(
                case=i,
                spec=row["spec"],
                representation=report["representation"],
                dtype=report["dtype"],
                native_template=row["native_template"],
                native_fd=row["native_fd"],
                ratios=ratios,
                mean_ratio=statistics.fmean(ratios.values()),
            )
            (cases if applicable else excluded).append(item)
    if not cases:
        raise ValueError("No applicable GQA public configurations")
    mean = statistics.fmean(x["mean_ratio"] for x in cases)
    segment_names = {name for item in cases for name in item["ratios"]}
    return dict(
        passed=mean <= 1,
        mean_ratio=mean,
        threshold=1,
        max_cases=PUBLIC_MAX_CASES,
        measured_cases=4 * len(specs),
        applicable_cases=len(cases),
        excluded_cases=len(excluded),
        regressed_cases=sum(x["mean_ratio"] > 1 for x in cases),
        regressed_segments=sum(value > 1 for item in cases for value in item["ratios"].values()),
        segment_means={
            name: statistics.fmean(item["ratios"][name] for item in cases if name in item["ratios"])
            for name in sorted(segment_names)
        },
        weighting="equal input/dtype weight; equal entry/R weight within each input",
        cases=cases,
        excluded=excluded,
    )
