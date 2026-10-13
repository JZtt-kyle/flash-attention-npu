"""Budgeting, deterministic coverage and aggregate performance failure paths."""

from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest

from tests.mla_public_performance import FUNC_SEGMENTS, SEGMENTS, summarize
from tests.mla_validation_policy import public_cases, select_nodeids


def test_limit_is_deterministic_stratified_and_full_mode_preserves_all():
    nodes = [
        f"m.py::test_{f}[{d}-{i}]" for f in range(4) for d in ("fp16", "bf16") for i in range(50)
    ]
    chosen = select_nodeids(nodes)
    assert len(chosen) == 100
    assert set(chosen) == set(select_nodeids(list(reversed(nodes))))
    assert all(
        any(f"test_{f}[{d}-" in n for n in chosen) for f in range(4) for d in ("fp16", "bf16")
    )
    assert select_nodeids(nodes, 0) == nodes
    assert select_nodeids(nodes[:3]) == nodes[:3]
    with pytest.raises(ValueError):
        select_nodeids(["same", "same"])
    with pytest.raises(ValueError):
        select_nodeids(nodes, -1)


def test_public_input_cap_is_shared_by_dtypes_and_independent_of_results():
    specs = [dict(shape=[1, i, 8, 257]) for i in range(120)]
    selected = public_cases(specs)
    assert len(selected) * 2 == 100
    assert sorted(selected, key=str) == sorted(public_cases(specs[::-1]), key=str)
    assert public_cases(specs[:48]) == specs[:48]


def reports(ratios=(0.6, 1.2)):
    result = []
    for dtype in ("fp16", "bf16"):
        rows = []
        for index, ratio in enumerate(ratios):
            row = dict(
                shape=[1, index + 1, 8, 257],
                scale=None,
                lengths=None,
                stride_profile="contiguous",
                native_template=5,
                native_fd=False,
                e2e_applicable=True,
            )
            for name, (ours, source) in {**SEGMENTS, **FUNC_SEGMENTS}.items():
                row[ours] = dict(median_e2e_us=ratio * 100)
                row[source] = dict(median_e2e_us=100)
                row[name] = ratio
            rows.append(row)
        result.append(
            dict(
                dtype=dtype,
                causal=True,
                results=rows,
                warmup=5,
                rounds=5,
                wall_iterations=20,
                metadata_lifecycle=dict(complete_segments=[1, 4]),
            )
        )
    return result


def test_mean_accepts_individual_regression_but_rejects_overall_regression():
    result = summarize(reports(), 2)
    assert result["passed"] and result["mean_ratio"] == pytest.approx(0.9)
    assert result["regressed_cases"] == 2 and result["regressed_segments"] == 16
    assert not summarize(reports((0.9, 1.3)), 2)["passed"]
    assert summarize(reports((1.0,)), 1)["passed"]


@pytest.mark.parametrize(
    "fault",
    ["missing", "dtype", "nan", "zero", "ratio", "matrix", "protocol", "dispatch", "empty", "cap"],
)
def test_invalid_or_incomplete_performance_evidence_fails(fault):
    data = deepcopy(reports())
    row = data[0]["results"][0]
    count = 2
    if fault == "missing":
        del row["implicit_4"]
    elif fault == "dtype":
        data[1]["dtype"] = "fp16"
    elif fault in {"nan", "zero"}:
        row["fa3"]["median_e2e_us"] = float("nan") if fault == "nan" else 0
    elif fault == "ratio":
        row["e2e_ratio"] = 0.1
    elif fault == "matrix":
        data[1]["results"].pop()
    elif fault == "protocol":
        data[0]["rounds"] = 1
    elif fault == "dispatch":
        row["native_fd"] = True
    elif fault == "empty":
        for report in data:
            for item in report["results"]:
                item.update(native_fd=True, e2e_applicable=False)
    elif fault == "cap":
        count = 51
    with pytest.raises((ValueError, KeyError)):
        summarize(data, count)


def test_plugin_collect_and_execution_use_same_limit(tmp_path):
    module = tmp_path / "test_budget.py"
    module.write_text(
        "import pytest\n@pytest.mark.parametrize('i', range(11))\ndef test_case(i): assert i >= 0\n"
    )
    snapshots = []
    for name, options in (("collect", ["--collect-only"]), ("run", []), ("full", [])):
        output = tmp_path / name
        output.mkdir()
        env = dict(
            os.environ,
            MLA_VALIDATION_RESULT=str(output),
            MLA_VALIDATION_NPU="0",
            MLA_VALIDATION_MAX_CASES="0" if name == "full" else "3",
            TORCH_DEVICE_BACKEND_AUTOLOAD="0",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        )
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "-c",
                "/dev/null",
                "-p",
                "tests.mla_validation_plugin",
                str(module),
                *options,
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        selected = json.loads((output / "cases.json").read_text())
        assert len(selected) == (11 if name == "full" else 3)
        snapshots.append(selected)
        manifest = json.loads((output / "selection.json").read_text())
        assert len(manifest["candidates"]) == 11
    assert snapshots[0] == snapshots[1]


def test_average_cli_propagates_failure_and_preserves_diagnostics(tmp_path):
    paths = [tmp_path / f"{dtype}.json" for dtype in ("fp16", "bf16")]
    for path, report in zip(paths, reports((0.9, 1.3))):
        path.write_text(json.dumps(report))
    output = tmp_path / "average.json"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.mla_public_performance",
            *map(str, paths),
            "--cases-per-dtype",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    saved = json.loads(output.read_text())
    assert not saved["passed"] and saved["regressed_cases"] == 2
    assert len(saved["reports"]) == 2
