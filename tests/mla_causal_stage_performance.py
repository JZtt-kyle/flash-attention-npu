"""Validate and combine the five capped causal public performance groups."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

from tests.mla_dense_build import sha256
from tests.mla_public_performance import (
    input_identity,
    summarize,
    summarize_mixed,
    summarize_varlen,
)
from tests.mla_validation import digest, reusable
from tests.mla_validation_policy import (
    CAUSAL_STAGE_GROUPS,
    CAUSAL_STAGE_POLICY,
    CAUSAL_STAGE_SUITES,
    PUBLIC_MAX_CASES,
    stage_selection,
)


def summarize_stage(averages):
    if set(averages) != set(CAUSAL_STAGE_GROUPS):
        raise ValueError("All five causal layouts are required")
    cases, excluded = [], []
    layouts = {}
    for group in CAUSAL_STAGE_GROUPS:
        result = averages[group]
        count = len(result["cases"]) + len(result["excluded"])
        if (
            not 0 < count <= PUBLIC_MAX_CASES // len(CAUSAL_STAGE_GROUPS)
            or count != result["measured_cases"]
        ):
            raise ValueError("Invalid layout configuration count")
        values = [c["mean_ratio"] for c in result["cases"]]
        if not values or not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("Invalid configuration mean")
        if not math.isclose(statistics.fmean(values), result["mean_ratio"], rel_tol=1e-12):
            raise ValueError("Inconsistent layout mean")
        cases.extend(dict(c, layout=group) for c in result["cases"])
        excluded.extend(dict(c, layout=group) for c in result["excluded"])
        layouts[group] = {k: v for k, v in result.items() if k not in {"cases", "excluded"}}
    mean = statistics.fmean(c["mean_ratio"] for c in cases)
    return dict(
        policy=CAUSAL_STAGE_POLICY,
        passed=mean <= 1,
        threshold=1.0,
        mean_ratio=mean,
        max_configurations=PUBLIC_MAX_CASES,
        weighting="equal input/dtype weight across layouts; equal entry/R weight within each input",
        measured_cases=len(cases) + len(excluded),
        applicable_cases=len(cases),
        excluded_cases=len(excluded),
        regressed_cases=sum(c["mean_ratio"] > 1 for c in cases),
        regressed_segments=sum(r["regressed_segments"] for r in averages.values()),
        layouts=layouts,
        cases=cases,
        excluded=excluded,
    )


def load_evidence(paths):
    averages, evidence = {}, []
    source_identity = {}
    runtime = None
    for path in paths:
        summary = json.loads(path.read_text())
        if (
            not summary.get("causal_stage_performance")
            or summary.get("causal_stage_policy") != CAUSAL_STAGE_POLICY
        ):
            raise ValueError("Expected capped causal stage evidence")
        current_runtime = summary["runtime_identity"]
        if runtime is not None and runtime != current_runtime:
            raise ValueError("Stage runtime identities differ")
        runtime = current_runtime
        for suite, group in CAUSAL_STAGE_SUITES.items():
            record = summary["groups"].get(suite)
            if record is None or record.get("status") == "not_run":
                continue
            if group in averages:
                raise ValueError("Duplicate layout evidence")
            if not reusable(record, record.get("identity")):
                raise ValueError("Failed, missing or damaged layout evidence: " + suite)
            directory = Path(record["directory"])

            def read(relative):
                file = directory / relative
                if str(file) not in record["evidence"]:
                    raise ValueError("Unbound evidence: " + str(file))
                return json.loads(file.read_text())

            snapshot = read("sources.json")
            if not isinstance(snapshot, dict) or not snapshot:
                raise ValueError("Missing layout source identity")
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
            selection = read(f"public-stage-selection-{group}.json")
            if selection != stage_selection(selection["candidates"], group):
                raise ValueError("Stage input selection differs from policy")
            specs = selection["selected"]
            reports = [read(f"public-profile/{dtype}/profile.json") for dtype in ("fp16", "bf16")]
            if group == "bsnd":
                expected = [
                    dict(
                        shape=s["shape"],
                        scale=s.get("scale"),
                        lengths=s.get("lengths"),
                        stride_profile=s.get("stride", "contiguous"),
                    )
                    for s in specs
                ]
                if any([input_identity(row) for row in r["results"]] != expected for r in reports):
                    raise ValueError("BSND inputs differ from fixed selection")
                result = summarize(reports, len(specs))
            else:
                result = (summarize_varlen if group == "tnd" else summarize_mixed)(reports, specs)
            if result != read("public-profile/average.json"):
                raise ValueError("Saved average differs from raw timings")
            averages[group] = result
            evidence.append(
                dict(
                    summary=str(path),
                    summary_sha256=sha256(path),
                    suite=suite,
                    identity=record["identity"],
                    directory=str(directory),
                )
            )
    result = summarize_stage(averages)
    return dict(
        result,
        evidence=evidence,
        runtime_identity=runtime,
        source_identity=dict(files=len(source_identity), sha256=digest(source_identity)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = load_evidence(args.evidence)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = dict(passed=False, error=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"cases", "excluded", "layouts", "runtime_identity"}
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
