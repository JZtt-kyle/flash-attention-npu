"""Recompute M10's full stage or explicitly selected groups from fixed inputs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from tests.mla.support.mla_causal_stage_performance import summarize_stage
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_public_performance import (
    input_identity,
    summarize,
    summarize_mixed,
    summarize_varlen,
)
from tests.mla.support.mla_split_stage_cases import (
    SPLIT_STAGE_POLICY,
    SPLIT_STAGE_SUITES,
    split_stage_selection,
)
from tests.mla.support.mla_validation import digest, reusable
from tests.mla.support.mla_validation_policy import PUBLIC_MAX_CASES


def recompute(suite, reports, specs):
    layout, mode, _ = SPLIT_STAGE_SUITES[suite]
    automatic = mode == "automatic"
    if layout == "bsnd":
        expected = [
            dict(
                shape=s["shape"],
                scale=s.get("scale"),
                lengths=s.get("lengths"),
                stride_profile=s.get("stride", "contiguous"),
                window=s["window"],
                causal=s.get("causal", False),
                **(dict(num_splits=s["splits"]) if mode != "single" else {}),
            )
            for s in specs
        ]
        if any([input_identity(row) for row in r["results"]] != expected for r in reports):
            raise ValueError("BSND inputs differ from fixed M10 selection")
        return summarize(reports, len(specs), include_native_fd=automatic)
    function = summarize_varlen if layout == "tnd" else summarize_mixed
    return function(reports, specs, include_native_fd=automatic)


def load_evidence(paths, suites=None):
    required = list(SPLIT_STAGE_SUITES) if suites is None else list(suites)
    if (
        not required
        or len(set(required)) != len(required)
        or not set(required) <= set(SPLIT_STAGE_SUITES)
    ):
        raise ValueError("Expected distinct known M10 performance suites")
    groups, source_identity, evidence = {}, {}, []
    runtime = None
    for path in paths:
        summary = json.loads(path.read_text())
        if (
            not summary.get("split_stage_performance")
            or summary.get("split_stage_policy") != SPLIT_STAGE_POLICY
        ):
            raise ValueError("Expected fixed M10 stage evidence")
        current_runtime = summary["runtime_identity"]
        if not current_runtime or (runtime is not None and runtime != current_runtime):
            raise ValueError("Stage runtime identities differ or are missing")
        runtime = current_runtime
        for suite in SPLIT_STAGE_SUITES:
            if suite not in required:
                continue
            record = summary["groups"].get(suite)
            if record is None or record.get("status") == "not_run":
                continue
            if suite in groups:
                raise ValueError("Duplicate M10 group evidence")
            if not reusable(record, record.get("identity")):
                raise ValueError("Failed, missing or damaged M10 evidence: " + suite)
            directory = Path(record["directory"])

            def read(relative):
                file = directory / relative
                if str(file) not in record["evidence"]:
                    raise ValueError("Unbound evidence: " + str(file))
                return json.loads(file.read_text())

            snapshot = read("sources.json")
            if not isinstance(snapshot, dict) or not snapshot:
                raise ValueError("Missing source identity")
            for source, fingerprint in snapshot.items():
                if (
                    not isinstance(fingerprint, str)
                    or len(fingerprint) != 64
                    or any(c not in "0123456789abcdef" for c in fingerprint)
                ):
                    raise ValueError("Invalid source fingerprint")
                if source in source_identity and source_identity[source] != fingerprint:
                    raise ValueError("Conflicting shared source identity: " + source)
                source_identity[source] = fingerprint
            selection = read(f"split-stage-selection-{suite}.json")
            if selection != split_stage_selection(selection["candidates"], suite):
                raise ValueError("M10 selection differs from fixed policy")
            reports = [read(f"public-profile/{dtype}/profile.json") for dtype in ("fp16", "bf16")]
            result = recompute(suite, reports, selection["selected"])
            if result != read("public-profile/average.json"):
                raise ValueError("Saved M10 average differs from raw timings")
            groups[suite] = result
            evidence.append(
                dict(
                    summary=str(path),
                    summary_sha256=sha256(path),
                    suite=suite,
                    identity=record["identity"],
                    directory=str(directory),
                )
            )
    if set(groups) != set(required):
        raise ValueError(
            "All fifteen M10 layout/split groups are required"
            if suites is None
            else "All requested M10 groups are required"
        )
    layouts = defaultdict(lambda: dict(cases=[], excluded=[], regressed_segments=0))
    segments = defaultdict(list)
    for suite, group in groups.items():
        layout, mode, _ = SPLIT_STAGE_SUITES[suite]
        target = layouts[layout]
        for key in ("cases", "excluded"):
            target[key].extend(dict(c, split_mode=mode, suite=suite) for c in group[key])
        target["regressed_segments"] += group["regressed_segments"]
        for case in group["cases"]:
            ratios = case["ratios"]
            if isinstance(ratios, list):
                ratios = dict(zip(("varlen_r1", "varlen_r4"), ratios))
            for name, ratio in ratios.items():
                segments[name].append(ratio)
    for target in layouts.values():
        target.update(
            measured_cases=len(target["cases"]) + len(target["excluded"]),
            mean_ratio=statistics.fmean(c["mean_ratio"] for c in target["cases"]),
        )
    if suites is None:
        result = summarize_stage(layouts)
    else:
        # Incremental regressions may affect fewer than five layouts. Retain
        # each selected input/dtype's equal weight, including slower groups.
        cases = [dict(c, layout=name) for name, r in layouts.items() for c in r["cases"]]
        excluded = [dict(c, layout=name) for name, r in layouts.items() for c in r["excluded"]]
        measured = len(cases) + len(excluded)
        if not cases or not 0 < measured <= PUBLIC_MAX_CASES:
            raise ValueError("Invalid selected configuration count")
        mean = statistics.fmean(c["mean_ratio"] for c in cases)
        result = dict(
            scope="selected-groups",
            selected_suites=required,
            passed=mean <= 1,
            threshold=1.0,
            mean_ratio=mean,
            max_configurations=PUBLIC_MAX_CASES,
            weighting="equal input/dtype weight; equal entry/R weight within each input",
            measured_cases=measured,
            applicable_cases=len(cases),
            excluded_cases=len(excluded),
            regressed_cases=sum(c["mean_ratio"] > 1 for c in cases),
            regressed_segments=sum(r["regressed_segments"] for r in groups.values()),
            cases=cases,
            excluded=excluded,
        )
    return dict(
        result,
        policy=SPLIT_STAGE_POLICY,
        groups={
            s: {k: v for k, v in r.items() if k not in {"cases", "excluded"}}
            for s, r in groups.items()
        },
        segment_means={name: statistics.fmean(values) for name, values in segments.items()},
        evidence=evidence,
        runtime_identity=runtime,
        source_identity=dict(files=len(source_identity), sha256=digest(source_identity)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=SPLIT_STAGE_SUITES,
        help="Explicit regression groups; omit to require the complete M10 stage",
    )
    args = parser.parse_args()
    try:
        result = load_evidence(args.evidence, args.suite)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = dict(passed=False, error=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in {"passed", "mean_ratio", "measured_cases", "applicable_cases", "error"}
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
