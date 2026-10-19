"""Fixed BSND INT8 public configurations and equal-weight timing checks."""

import math
import statistics

from tests.mla.support.mla_int8_cases import SOURCE_CASES
from tests.mla.support.mla_gqa_options_profile_rows import modes, segments


PERFORMANCE = tuple(
    dict(layout="paged_bsnd", index=i, view="contiguous")
    for i, spec in enumerate(SOURCE_CASES)
    if spec[1] == "BSND"
) + tuple(
    dict(layout="paged_bsnd", index=i, view=view)
    for i, view in (
        (3, "row-gap"),
        (8, "page-gap"),
        (8, "query-gap"),
        (8, "scale-gap"),
        (16, "table-gap"),
        (6, "row-gap"),
        (2, "query-gap"),
        (14, "scale-gap"),
    )
)


def selected(check):
    # Decode FD, multi-Q tail/stride, fast FD and the largest public allocation.
    return (
        [
            dict(layout="paged_bsnd", index=i, view=view)
            for i, view in (
                (6, "contiguous"),
                (8, "query-gap"),
                (16, "table-gap"),
                (14, "scale-gap"),
            )
        ]
        if check
        else PERFORMANCE
    )


def summarize(samples, specs):
    if not specs or len(specs) > 100:
        raise ValueError("Empty or excessive INT8 public configuration set")
    expected = {
        (i, name, mode, r, it)
        for i, spec in enumerate(specs)
        for name in modes(spec)
        for mode in ("eager", "graph")
        for r in range(5)
        for it in range(20)
    }
    actual = {(x["case"], x["name"], x["mode"], x["round"], x["iteration"]) for x in samples}
    if actual != expected or len(samples) != len(expected):
        raise ValueError("Incomplete, duplicate or changed timing protocol")
    if any(not math.isfinite(x["e2e_us"]) or x["e2e_us"] <= 0 for x in samples):
        raise ValueError("Invalid wall timing")
    reports = {}
    for mode in ("eager", "graph"):
        rows = []
        for i, spec in enumerate(specs):
            timings = {}
            for name in modes(spec):
                values = [
                    x for x in samples if (x["case"], x["name"], x["mode"]) == (i, name, mode)
                ]
                timings[name] = dict(
                    median_e2e_us=statistics.median(x["e2e_us"] for x in values),
                    round_mean_e2e_us=[
                        statistics.fmean(x["e2e_us"] for x in values if x["round"] == r)
                        for r in range(5)
                    ],
                )
            ratios = {
                key: timings[a]["median_e2e_us"] / timings[b]["median_e2e_us"]
                for key, (a, b) in segments(spec).items()
            }
            rows.append(
                dict(
                    case=i,
                    spec=spec,
                    timings=timings,
                    ratios=ratios,
                    mean_ratio=statistics.fmean(ratios.values()),
                )
            )
        mean = statistics.fmean(x["mean_ratio"] for x in rows)
        reports[mode] = dict(
            passed=mean <= 1,
            mean_ratio=mean,
            threshold=1,
            max_cases=100,
            measured_cases=len(rows),
            applicable_cases=len(rows),
            excluded_cases=0,
            regressed_cases=sum(x["mean_ratio"] > 1 for x in rows),
            regressed_segments=sum(r > 1 for x in rows for r in x["ratios"].values()),
            segments={
                name: statistics.fmean(x["ratios"][name] for x in rows)
                for name in segments(specs[0])
            },
            cases=rows,
            warmup=5,
            rounds=5,
            wall_iterations=20,
        )
    return reports


def partition_kernels(rows, labels, specs, dtype):
    if dtype != "int8":
        raise ValueError("Wrong INT8 profiler dtype")
    pos, result, models = 0, [], {}
    for label in labels:
        ours = label["name"] == "fa3"
        target = (
            "mla_int8_tnd" if specs[label["case"]]["layout"] == "paged_tnd" else "mla_int8_bsnd"
        )
        terminal = target if ours else "_FusedInferAttentionScore"
        group = []
        while pos < len(rows):
            row = rows[pos]
            pos += 1
            group.append(row)
            if terminal in row["Name"]:
                break
        else:
            raise ValueError("Missing INT8/native terminal task")
        if any(
            "ComputeMla" in row["Name"]
            or ("mla_int8_" in row["Name"] and (not ours or target not in row["Name"]))
            or ("FusedInferAttentionScore" in row["Name"] and ours)
            for row in group
        ):
            raise ValueError("Foreign work or unexpected metadata in reuse segment")
        if "model_id" in group[0]:
            ids = {x["model_id"] for x in group}
            key = label["case"], label["name"]
            if len(ids) != 1 or models.setdefault(key, group[0]["model_id"]) not in ids:
                raise ValueError("Mixed captured model identities")
        result.append(
            dict(
                **label,
                tasks=group,
                kernel_us=float(group[-1]["Duration(us)"]),
                adaptation_us=sum(float(x["Duration(us)"]) for x in group[:-1]),
            )
        )
    if pos != len(rows) or len(set(models.values())) != len(models):
        raise ValueError("Unclaimed tasks or shared graph model identity")
    return result
