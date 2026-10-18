"""M14 raw sample completeness, fixed selection, provenance and aggregate gate."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.mla.support import mla_validation as driver
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_prefill_stage_cases import (
    PREFILL_STAGE_POLICY,
    PREFILL_STAGE_SUITES,
    stage_selection,
)
from tests.mla.support.mla_prefill_stage_performance import (
    cache_identity,
    load_evidence,
    recompute_report,
)
from tests.mla.support.mla_public_performance import (
    SEGMENTS,
    FUNC_SEGMENTS,
    summarize,
    summarize_varlen,
)
from tests.mla.support.mla_validation_policy import check_public_performance
from tests.mla.unit.tools.test_mla_validation_policy import reports, varlen_reports

PA_SUITES = ["prefill-paged-bsnd-profile", "prefill-paged-tnd-profile"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def stage_evidence(root, *, slow=0.6, extra=False):
    native = root / "native.so"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_bytes(b"fixture native package")
    groups = {}
    for suite, (layout, _) in PREFILL_STAGE_SUITES.items():
        specs = stage_selection()[suite]
        ratio = slow if layout == "paged-tnd" else 0.6
        if layout == "tnd":
            _, data = varlen_reports((ratio,) * len(specs))
            for report in data:
                report["results"] = [
                    dict(deepcopy(report["results"][0]), case=i) for i in range(len(specs))
                ]
        else:
            data = reports((ratio,) * len(specs))
        files = {"sources.json": {"shared.py": "a" * 64}}
        for report in data:
            report.update(
                representation="prefill",
                window=True,
                profiler_iterations=10,
                cpu_affinity=[0],
                visible_devices="6",
            )
            wall, kernels, dispatch = [], [], []
            for i, (row, spec) in enumerate(zip(report["results"], specs)):
                fd = not (extra and layout == "bsnd" and i == 0)
                row.update(
                    native_template=3,
                    native_fd=fd,
                    e2e_applicable=not fd or spec["splits"] == 0,
                    kernel_us=dict(fa3=2.0, native_cann=1.0),
                )
                dispatch.append(dict(case=i, native_template=3, native_fd=fd))
                if layout == "tnd":
                    row["spec"] = spec
                    names = [(name, r) for name in ("fa3", "native_cann") for r in (1, 4)]
                else:
                    row.update(cache_identity(layout, spec), generate=dict(median_e2e_us=20.0))
                    names = {name for pair in SEGMENTS.values() for name in pair} | {"generate"}
                    if row["lengths"] is None:
                        names.update(name for pair in FUNC_SEGMENTS.values() for name in pair)
                    names = [(name, None) for name in names]
                for name, r in names:
                    value = (
                        (100 if name == "native_cann" else 100 * ratio)
                        if r
                        else row[name]["median_e2e_us"]
                    )
                    wall.extend(
                        dict(
                            case=i,
                            name=name,
                            round=round_,
                            iteration=iteration,
                            e2e_us=value,
                            **({"repeats": r} if r else {}),
                        )
                        for round_ in range(5)
                        for iteration in range(20)
                    )
                kernels.extend(
                    dict(case=i, name=name, round=round_, iteration=iteration, kernel_us=value)
                    for name, value in (("fa3", 2.0), ("native_cann", 1.0))
                    for round_ in range(5)
                    for iteration in range(10)
                )
            prefix = f"public-profile/{report['dtype']}"
            files.update(
                {
                    prefix + "/profile.json": report,
                    prefix + "/wall-samples.json": wall,
                    prefix + "/kernel-samples.json": kernels,
                    prefix + "/accuracy.json": [dict(spec=s) for s in specs],
                    prefix + ".dispatch.json": dict(
                        dtype=report["dtype"],
                        representation="prefill",
                        specs=specs,
                        cases=dispatch,
                        artifacts={str(native): sha256(native)},
                    ),
                }
            )
        files["public-profile/average.json"] = (
            summarize_varlen(data, specs) if layout == "tnd" else summarize(data, len(specs))
        )
        directory = root / suite
        for name, value in files.items():
            save(directory / name, value)
        groups[suite] = dict(
            status="passed",
            identity=suite,
            directory=str(directory),
            evidence={str(directory / p): sha256(directory / p) for p in files},
        )
    path = root / "summary.json"
    save(
        path,
        dict(
            prefill_stage_performance=True,
            prefill_stage_policy=PREFILL_STAGE_POLICY,
            prefill_stage_selection=stage_selection(),
            groups=groups,
            runtime_identity=dict(extension_sha256="c" * 64, visible_devices="6"),
        ),
    )
    return path


def test_fixed_selection_and_driver_gate(tmp_path, monkeypatch):
    selected = stage_selection()
    assert sum(map(len, selected.values())) * 2 == 60
    for suite, specs in selected.items():
        assert suite in driver.PERFORMANCE_SUITES
        assert {s["mask"] for s in specs} == {"none", "causal", "window"}
        assert {s["splits"] for s in specs} == {0, 1, 2}
    monkeypatch.setenv("MLA_PREFILL_STAGE_PERFORMANCE", "1")
    args = SimpleNamespace(device="6", golden="cache", artifacts=tmp_path)
    assert "MLA_PREFILL_STAGE_PERFORMANCE" not in driver.environment(args, tmp_path, "e2e")
    args.prefill_stage_performance = True
    assert (
        driver.environment(args, tmp_path, "prefill-bsnd-profile")["MLA_PREFILL_STAGE_PERFORMANCE"]
        == "1"
    )
    check_public_performance(dict(passed=False, mean_ratio=1.2))
    args.suite = ["e2e"]
    with pytest.raises(ValueError, match="M14 performance suites"):
        driver.run(args)
    args.suite = ["prefill-bsnd-profile"]
    args.split_stage_performance = True
    with pytest.raises(ValueError, match="M14 performance suites"):
        driver.run(args)
    monkeypatch.delenv("MLA_PREFILL_STAGE_PERFORMANCE")
    with pytest.raises(AssertionError):
        check_public_performance(dict(passed=False, mean_ratio=1.2))


def test_aggregate_weights_configs_and_retains_fd_and_kernel_regressions(tmp_path):
    path = stage_evidence(tmp_path, slow=1.4, extra=True)
    result = load_evidence([path])
    assert result["passed"] and result["mean_ratio"] == pytest.approx((26 * 0.6 + 6 * 1.4) / 32)
    assert result["measured_cases"] == 60 and result["applicable_cases"] == 32
    assert result["excluded_cases"] == 28 and result["regressed_cases"] == 6
    assert all(c["num_splits"] != 0 for c in result["excluded"] if "num_splits" in c)
    assert result["kernel_regressed_cases"] == 60
    assert result["layouts"]["paged-tnd"]["passed"] is False


def test_raw_medians_replace_saved_rounding(tmp_path):
    path = stage_evidence(tmp_path)
    summary = json.loads(path.read_text())
    suite = "prefill-bsnd-profile"
    root = Path(summary["groups"][suite]["directory"]) / "public-profile"

    def read(name):
        return json.loads((root / name).read_text())

    report = read("fp16/profile.json")
    report["results"][3]["fa3"]["median_e2e_us"] += 1e-11
    result = recompute_report(
        "bsnd",
        stage_selection()[suite],
        report,
        read("fp16/wall-samples.json"),
        read("fp16/kernel-samples.json"),
        read("fp16.dispatch.json"),
    )
    assert result["results"][3]["fa3"]["median_e2e_us"] == 60.0


def test_selected_groups_keep_raw_evidence_and_regressions(tmp_path):
    path = stage_evidence(tmp_path, slow=1.2)
    summary = json.loads(path.read_text())
    summary["groups"] = {s: summary["groups"][s] for s in PA_SUITES}
    save(path, summary)
    result = load_evidence([path], PA_SUITES)
    assert result["passed"] and result["mean_ratio"] == pytest.approx(0.9)
    assert result["scope"] == "selected-groups" and result["selected_suites"] == PA_SUITES
    assert result["measured_cases"] == 24 and result["applicable_cases"] == 12
    assert result["excluded_cases"] == 12 and result["regressed_cases"] == 6
    assert result["kernel_regressed_cases"] == 24
    with pytest.raises(ValueError, match="required"):
        load_evidence([path])
    for suites in ([], [PA_SUITES[0]] * 2, ["unknown"]):
        with pytest.raises(ValueError, match="distinct known"):
            load_evidence([path], suites)


@pytest.mark.parametrize("slow,code", [(0.6, 0), (3.0, 1)])
@pytest.mark.parametrize("subset", [False, True])
def test_cli_success_and_failure(tmp_path, slow, code, subset):
    path = stage_evidence(tmp_path, slow=slow)
    output = tmp_path / "stage.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.mla.support.mla_prefill_stage_performance",
            "--evidence",
            str(path),
            "--output",
            str(output),
        ]
        + (["--suite", *PA_SUITES] if subset else []),
        capture_output=True,
    )
    assert result.returncode == code, result.stderr.decode()
    assert json.loads(output.read_text())["passed"] == (code == 0)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "failed",
        "corrupt",
        "unbound",
        "selection",
        "policy",
        "runtime",
        "sources",
        "dtype",
        "affinity",
        "spec",
        "shape",
        "page",
        "dispatch",
        "native",
        "native-identity",
        "accuracy",
        "sample-missing",
        "sample-duplicate",
        "sample-zero",
        "sample-nan",
        "sample-round",
        "median",
        "kernel",
        "average",
        "protocol",
        "empty",
    ],
)
@pytest.mark.parametrize("subset", [False, True])
def test_incomplete_or_changed_evidence_rejected(tmp_path, fault, subset):
    path = stage_evidence(tmp_path)
    summary = json.loads(path.read_text())
    suite = "prefill-paged-tnd-profile"
    record = summary["groups"][suite]
    directory = Path(record["directory"])
    relative = "public-profile/fp16/profile.json"
    if fault == "missing":
        del summary["groups"][suite]
    elif fault == "duplicate":
        with pytest.raises(ValueError, match="Duplicate"):
            load_evidence([path, path], PA_SUITES if subset else None)
        return
    elif fault == "failed":
        record["status"] = "failed"
    elif fault == "corrupt":
        (directory / relative).write_text("corrupted")
    elif fault == "unbound":
        record["evidence"].pop(str(directory / relative))
    elif fault == "selection":
        summary["prefill_stage_selection"][suite].pop()
    elif fault == "policy":
        summary["prefill_stage_performance"] = False
    elif fault == "runtime":
        summary["runtime_identity"] = None
    elif fault == "native":
        (tmp_path / "native.so").write_bytes(b"changed")
    else:
        if fault == "sources":
            relative = "sources.json"
        elif fault in {"spec", "dispatch", "native-identity"}:
            relative = "public-profile/fp16.dispatch.json"
        elif fault == "accuracy":
            relative = "public-profile/fp16/accuracy.json"
        elif fault.startswith("sample-"):
            relative = "public-profile/fp16/wall-samples.json"
        elif fault == "kernel":
            relative = "public-profile/fp16/kernel-samples.json"
        elif fault == "average":
            relative = "public-profile/average.json"
        file = directory / relative
        value = json.loads(file.read_text())
        if fault == "sources":
            value["shared.py"] = "b" * 64
        elif fault == "dtype":
            value["dtype"] = "bf16"
        elif fault == "affinity":
            value["cpu_affinity"] = [1]
        elif fault == "spec":
            value["specs"][0]["page"] = 128
        elif fault == "shape":
            value["results"][0]["shape"][0] += 1
        elif fault == "page":
            value["results"][0]["stride_profile"]["page_size"] = 128
        elif fault == "dispatch":
            value["cases"][0]["native_template"] = 5
        elif fault == "native-identity":
            other = tmp_path / "other-native.so"
            other.write_bytes(b"another valid package")
            value["artifacts"] = {str(other): sha256(other)}
        elif fault == "accuracy":
            value[0]["spec"]["splits"] = 0
        elif fault == "sample-missing":
            value.pop()
        elif fault == "sample-duplicate":
            value.append(deepcopy(value[0]))
        elif fault == "sample-zero":
            value[0]["e2e_us"] = 0
        elif fault == "sample-nan":
            value[0]["e2e_us"] = float("nan")
        elif fault == "sample-round":
            value[0]["round"] = 5
        elif fault == "median":
            value["results"][0]["fa3"]["median_e2e_us"] += 1
        elif fault == "kernel":
            value.pop()
        elif fault == "average":
            value["mean_ratio"] += 0.1
        elif fault == "protocol":
            value["profiler_iterations"] = 9
        elif fault == "empty":
            value["results"] = []
        save(file, value)
        record["evidence"][str(file)] = sha256(file)
    save(path, summary)
    with pytest.raises(ValueError):
        load_evidence([path], PA_SUITES if subset else None)


def test_different_runtime_across_summaries_rejected(tmp_path):
    first = stage_evidence(tmp_path / "one")
    second = stage_evidence(tmp_path / "two")
    a, b = (json.loads(p.read_text()) for p in (first, second))
    suite = "prefill-tnd-profile"
    a["groups"] = {suite: a["groups"][suite]}
    b["groups"].pop(suite)
    b["runtime_identity"]["extension_sha256"] = "d" * 64
    save(first, a)
    save(second, b)
    with pytest.raises(ValueError, match="runtime identities"):
        load_evidence([first, second])
