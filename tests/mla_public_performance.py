"""Average public causal E2E acceptance; saved reports can be re-evaluated."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from tests.mla_validation_policy import PUBLIC_MAX_CASES


SEGMENTS = {
    "e2e_ratio": ("fa3", "native_cann"),
    "generate_1_ratio": ("generate_1", "native_cann"),
    "generate_4_ratio": ("generate_4", "native_cann_4"),
    "reuse_4_ratio": ("reuse_4", "native_cann_4"),
    "implicit_ratio": ("implicit", "native_cann"),
    "implicit_4_ratio": ("implicit_4", "native_cann_4"),
}
FUNC_SEGMENTS = {
    "func_ratio": ("func", "native_cann"),
    "func_4_ratio": ("func_4", "native_cann_4"),
}


def input_identity(row):
    return {name: row[name] for name in ("shape", "scale", "lengths", "stride_profile")}


def summarize(reports, cases_per_dtype):
    """Equal weight per input/dtype, then equal weight per defined entry/R."""
    if cases_per_dtype <= 0 or cases_per_dtype * 2 > PUBLIC_MAX_CASES:
        raise ValueError("Expected 1..50 input cases per dtype")
    if len(reports) != 2 or {r["dtype"] for r in reports} != {"fp16", "bf16"}:
        raise ValueError("Expected exactly one FP16 and one BF16 report")
    identities = []
    cases, excluded = [], []
    segments = defaultdict(list)
    for report in reports:
        if (
            not report["causal"]
            or len(report["results"]) != cases_per_dtype
            or (report["warmup"], report["rounds"], report["wall_iterations"]) != (5, 5, 20)
            or report["metadata_lifecycle"]["complete_segments"] != [1, 4]
        ):
            raise ValueError("Incomplete matrix or changed timing protocol")
        identities.append([input_identity(row) for row in report["results"]])
        for index, row in enumerate(report["results"]):
            applicable = row["native_template"] == 5 and not row["native_fd"]
            if row["e2e_applicable"] != applicable:
                raise ValueError("Inconsistent native dispatch classification")
            item = dict(dtype=report["dtype"], index=index, **input_identity(row))
            if not applicable:
                excluded.append(
                    dict(item, native_fd=row["native_fd"], native_template=row["native_template"])
                )
                continue
            modes = dict(SEGMENTS)
            if row["lengths"] is None:
                modes.update(FUNC_SEGMENTS)
            ratios = {}
            for name, (ours, source) in modes.items():
                a, b = (row[key]["median_e2e_us"] for key in (ours, source))
                if not all(math.isfinite(x) and x > 0 for x in (a, b, row[name])):
                    raise ValueError("Non-finite or nonpositive timing")
                value = a / b
                if not math.isclose(value, row[name], rel_tol=1e-12):
                    raise ValueError("Recorded ratio differs from measured timings")
                ratios[name] = value
                segments[name].append(value)
            cases.append(dict(item, ratios=ratios, mean_ratio=statistics.fmean(ratios.values())))
    if identities[0] != identities[1]:
        raise ValueError("Dtype input matrices differ")
    if not cases:
        raise ValueError("No applicable public E2E cases")
    mean = statistics.fmean(item["mean_ratio"] for item in cases)
    return dict(
        passed=mean <= 1.0,
        threshold=1.0,
        mean_ratio=mean,
        weighting="equal input/dtype weight; equal entry/R weight within each input",
        max_cases=PUBLIC_MAX_CASES,
        measured_cases=2 * cases_per_dtype,
        applicable_cases=len(cases),
        excluded_cases=len(excluded),
        regressed_cases=sum(item["mean_ratio"] > 1 for item in cases),
        regressed_segments=sum(value > 1 for values in segments.values() for value in values),
        segment_means={key: statistics.fmean(values) for key, values in segments.items()},
        cases=cases,
        excluded=excluded,
    )


def summarize_varlen(reports, specs):
    """Varlen has one public entry; average R=1/4 within each fixed input/dtype."""
    if not 0 < len(specs) <= PUBLIC_MAX_CASES // 2:
        raise ValueError("Expected 1..50 fixed varlen inputs")
    if len(reports) != 2 or {r["dtype"] for r in reports} != {"fp16", "bf16"}:
        raise ValueError("Both dtypes are required")
    cases, excluded = [], []
    for report in reports:
        if not report["causal"] or (
            report["warmup"],
            report["rounds"],
            report["wall_iterations"],
            report["repeats"],
        ) != (5, 5, 20, [1, 4]):
            raise ValueError("Changed varlen timing protocol")
        if [row["spec"] for row in report["results"]] != specs:
            raise ValueError("Incomplete or changed varlen input matrix")
        for index, row in enumerate(report["results"]):
            if row["case"] != index:
                raise ValueError("Inconsistent case index")
            item = dict(dtype=report["dtype"], case=index, spec=row["spec"])
            applicable = row["native_template"] == 5 and not row["native_fd"]
            if applicable != row["e2e_applicable"]:
                raise ValueError("Inconsistent native dispatch classification")
            if not applicable:
                excluded.append(
                    dict(item, native_fd=row["native_fd"], native_template=row["native_template"])
                )
                continue
            ratios = []
            for repeats in (1, 4):
                segment = row[f"r{repeats}"]
                a, b, recorded = (segment[k] for k in ("fa3", "cann", "ratio"))
                if not all(math.isfinite(x) and x > 0 for x in (a, b, recorded)):
                    raise ValueError("Non-finite or nonpositive timing")
                ratio = a / b
                if not math.isclose(ratio, recorded, rel_tol=1e-12):
                    raise ValueError("Recorded ratio differs from measured timings")
                ratios.append(ratio)
            cases.append(dict(item, ratios=ratios, mean_ratio=statistics.fmean(ratios)))
    if not cases:
        raise ValueError("No applicable public E2E cases")
    mean = statistics.fmean(c["mean_ratio"] for c in cases)
    return dict(
        passed=mean <= 1,
        threshold=1.0,
        mean_ratio=mean,
        weighting="equal input/dtype weight; equal R=1/4 weight within each input",
        measured_cases=2 * len(specs),
        applicable_cases=len(cases),
        excluded_cases=len(excluded),
        regressed_cases=sum(c["mean_ratio"] > 1 for c in cases),
        regressed_segments=sum(x > 1 for c in cases for x in c["ratios"]),
        segment_means={
            f"r{r}": statistics.fmean(c["ratios"][i] for c in cases) for i, r in enumerate((1, 4))
        },
        cases=cases,
        excluded=excluded,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs=2)
    parser.add_argument("--cases-per-dtype", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(p.read_text()) for p in args.reports], args.cases_per_dtype)
    result["reports"] = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reports
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"cases", "excluded"}}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
