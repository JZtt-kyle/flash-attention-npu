"""Recompute M14's full stage or selected groups from bound raw timing samples."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from tests.mla.support.mla_causal_stage_performance import summarize_stage
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_prefill_stage_cases import (
    PREFILL_STAGE_POLICY,
    PREFILL_STAGE_SUITES,
    stage_selection,
)
from tests.mla.support.mla_public_performance import (
    SEGMENTS,
    FUNC_SEGMENTS,
    summarize,
    summarize_varlen,
)
from tests.mla.support.mla_validation import digest, reusable
from tests.mla.support.mla_validation_build import matching_files
from tests.mla.support.mla_validation_policy import PUBLIC_MAX_CASES


def sample_medians(samples, expected, iterations, *, repeats=False, value="e2e_us"):
    groups = defaultdict(dict)
    for sample in samples:
        key = (sample["case"], sample["name"])
        if repeats:
            key += (sample["repeats"],)
        position = (sample["round"], sample["iteration"])
        number = sample[value]
        if (
            key not in expected
            or position in groups[key]
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError("Unexpected, duplicate or invalid raw timing sample")
        groups[key][position] = number
    positions = {(r, i) for r in range(5) for i in range(iterations)}
    if set(groups) != set(expected) or any(set(v) != positions for v in groups.values()):
        raise ValueError("Incomplete raw timing matrix")
    return {key: statistics.median(values.values()) for key, values in groups.items()}


def cache_identity(layout, spec):
    """Describe the existing profile rows without importing device input helpers."""
    shape = (
        [len(spec["q"]), max(spec["q"]), spec["heads"], max(spec["k"]), spec["kv_heads"]]
        if layout == "mixed"
        else list(spec["shape"])
    )
    b, _, _, sk, _ = shape
    lengths = spec.get("lengths")
    stride = None
    if layout == "mixed":
        lengths = spec.get("used_k", spec["k"])
        stride = dict(query_allocation=spec["q"])
    elif layout.startswith("paged-"):
        page = spec["page"]
        shape[3] = (sk + page - 1) // page * page
        lengths = spec.get("lengths", [sk] * b)
        stride = dict(page_size=page, table_shape=[b, shape[3] // page])
    return dict(
        shape=shape,
        scale=192**-0.5,
        lengths=lengths,
        stride_profile=stride,
        window=[7, 2] if spec["mask"] == "window" else [-1, -1],
        causal=spec["mask"] == "causal",
        num_splits=spec["splits"],
    )


def recompute_report(layout, specs, report, wall, kernels, proof):
    if (
        report.get("representation") != "prefill"
        or proof.get("representation") != "prefill"
        or proof["dtype"] != report["dtype"]
        or proof["specs"] != specs
        or len(proof["cases"]) != len(specs)
        or len(report["results"]) != len(specs)
        or report["profiler_iterations"] != 10
    ):
        raise ValueError("Prefill input, dtype, dispatch or sampling protocol changed")
    expected = set()
    for i, (row, spec, dispatch) in enumerate(zip(report["results"], specs, proof["cases"])):
        if dispatch["case"] != i or any(
            row[key] != dispatch[key] for key in ("native_template", "native_fd")
        ):
            raise ValueError("Profile differs from actual native dispatch")
        if layout == "tnd":
            if row["spec"] != spec:
                raise ValueError("Changed compact input")
            expected.update((i, name, r) for name in ("fa3", "native_cann") for r in (1, 4))
        else:
            if any(row[key] != value for key, value in cache_identity(layout, spec).items()):
                raise ValueError("Changed cache input")
            names = {name for pair in SEGMENTS.values() for name in pair} | {"generate"}
            if row["lengths"] is None:
                names.update(name for pair in FUNC_SEGMENTS.values() for name in pair)
            expected.update((i, name) for name in names)
    medians = sample_medians(wall, expected, 20, repeats=layout == "tnd")
    kernel_medians = sample_medians(
        kernels,
        {(i, name) for i in range(len(specs)) for name in ("fa3", "native_cann")},
        10,
        value="kernel_us",
    )
    for i, row in enumerate(report["results"]):
        for key in (key for key in expected if key[0] == i):
            if layout == "tnd":
                saved = row[f"r{key[2]}"]["fa3" if key[1] == "fa3" else "cann"]
            else:
                saved = row[key[1]]["median_e2e_us"]
            if not math.isclose(saved, medians[key], rel_tol=1e-12):
                raise ValueError("Profile median differs from raw wall samples")
            if layout == "tnd":
                row[f"r{key[2]}"]["fa3" if key[1] == "fa3" else "cann"] = medians[key]
            else:
                row[key[1]]["median_e2e_us"] = medians[key]
        for name in ("fa3", "native_cann"):
            if not math.isclose(row["kernel_us"][name], kernel_medians[i, name], rel_tol=1e-12):
                raise ValueError("Profile kernel median differs from raw samples")
            row["kernel_us"][name] = kernel_medians[i, name]
    # Recompute with the raw medians, even if saved values differ only within
    # the consistency tolerance: rounding must not move the <=1 stage gate.
    return report


def load_evidence(paths, suites=None):
    required = list(PREFILL_STAGE_SUITES) if suites is None else list(suites)
    if (
        not required
        or len(set(required)) != len(required)
        or not set(required) <= set(PREFILL_STAGE_SUITES)
    ):
        raise ValueError("Expected distinct known M14 performance suites")
    selected = stage_selection()
    layouts, evidence, sources, native_artifacts = {}, [], {}, {}
    runtime = sampling = None
    kernel_cases = []
    for path in paths:
        summary = json.loads(path.read_text())
        if (
            not summary.get("prefill_stage_performance")
            or summary.get("prefill_stage_policy") != PREFILL_STAGE_POLICY
            or summary.get("prefill_stage_selection") != selected
        ):
            raise ValueError("Expected the complete fixed M14 selection")
        current = summary["runtime_identity"]
        if not current or (runtime is not None and runtime != current):
            raise ValueError("Stage runtime identities differ or are missing")
        runtime = current
        for suite, (layout, _) in PREFILL_STAGE_SUITES.items():
            if suite not in required:
                continue
            record = summary["groups"].get(suite)
            if record is None or record.get("status") == "not_run":
                continue
            if layout in layouts:
                raise ValueError("Duplicate prefill layout evidence")
            if not reusable(record, record.get("identity")):
                raise ValueError("Failed, missing or damaged prefill evidence: " + suite)
            directory = Path(record["directory"])

            def read(relative):
                file = directory / relative
                if str(file) not in record["evidence"]:
                    raise ValueError("Unbound evidence: " + str(file))
                return json.loads(file.read_text())

            snapshot = read("sources.json")
            if not snapshot:
                raise ValueError("Missing source identity")
            for source, fingerprint in snapshot.items():
                if (
                    not isinstance(fingerprint, str)
                    or len(fingerprint) != 64
                    or any(c not in "0123456789abcdef" for c in fingerprint)
                ):
                    raise ValueError("Invalid source fingerprint")
                if source in sources and sources[source] != fingerprint:
                    raise ValueError("Conflicting shared source identity")
                sources[source] = fingerprint
            reports = []
            for dtype in ("fp16", "bf16"):
                prefix = f"public-profile/{dtype}"
                report = read(prefix + "/profile.json")
                proof = read(f"public-profile/{dtype}.dispatch.json")
                if not proof["artifacts"] or not matching_files(proof["artifacts"]):
                    raise ValueError("Native dispatch artifacts changed")
                native = {
                    p: fingerprint
                    for p, fingerprint in proof["artifacts"].items()
                    if not Path(p).is_relative_to(directory)
                }
                if not native or (dtype in native_artifacts and native_artifacts[dtype] != native):
                    raise ValueError("Stage native package identities differ or are missing")
                native_artifacts[dtype] = native
                accuracy = read(prefix + "/accuracy.json")
                if [row["spec"] for row in accuracy] != selected[suite] or report["dtype"] != dtype:
                    raise ValueError("Accuracy/dispatch inputs differ from stage selection")
                current_sampling = (report["cpu_affinity"], report["visible_devices"])
                if (
                    len(current_sampling[0]) != 1
                    or current_sampling[1] != runtime["visible_devices"]
                    or (sampling is not None and sampling != current_sampling)
                ):
                    raise ValueError("Stage sampling affinity/device differs")
                sampling = current_sampling
                reports.append(
                    recompute_report(
                        layout,
                        selected[suite],
                        report,
                        read(prefix + "/wall-samples.json"),
                        read(prefix + "/kernel-samples.json"),
                        proof,
                    )
                )
                for i, row in enumerate(report["results"]):
                    times = row["kernel_us"]
                    kernel_cases.append(
                        dict(
                            layout=layout,
                            dtype=dtype,
                            index=i,
                            **times,
                            ratio=times["fa3"] / times["native_cann"],
                        )
                    )
            average = (
                summarize_varlen(reports, selected[suite])
                if layout == "tnd"
                else summarize(reports, len(selected[suite]))
            )
            if average != read("public-profile/average.json"):
                raise ValueError("Saved average differs from raw timings")
            layouts[layout] = average
            evidence.append(
                dict(
                    summary=str(path),
                    summary_sha256=sha256(path),
                    suite=suite,
                    identity=record["identity"],
                    directory=str(directory),
                )
            )
    if set(layouts) != {PREFILL_STAGE_SUITES[s][0] for s in required}:
        raise ValueError("All requested M14 groups are required")
    if suites is None:
        result = summarize_stage(layouts)
    else:
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
            regressed_segments=sum(r["regressed_segments"] for r in layouts.values()),
            layouts={
                name: {k: v for k, v in r.items() if k not in {"cases", "excluded"}}
                for name, r in layouts.items()
            },
            cases=cases,
            excluded=excluded,
        )
    segments = defaultdict(list)
    for case in result["cases"]:
        ratios = case["ratios"]
        if isinstance(ratios, list):
            ratios = dict(zip(("varlen_r1", "varlen_r4"), ratios))
        for name, value in ratios.items():
            segments[name].append(value)
    return dict(
        result,
        policy=PREFILL_STAGE_POLICY,
        selection=selected,
        segment_means={k: statistics.fmean(v) for k, v in segments.items()},
        kernel_cases=kernel_cases,
        kernel_regressed_cases=sum(c["ratio"] > 1 for c in kernel_cases),
        evidence=evidence,
        native_artifacts=native_artifacts,
        runtime_identity=runtime,
        source_identity=dict(files=len(sources), sha256=digest(sources)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=PREFILL_STAGE_SUITES,
        help="Explicit regression groups; omit to require the complete M14 stage",
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
