"""Regression selection, stable identities and artifact placement after the move."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

from tests.mla.catalog import DIAGNOSTIC_E2E, E2E, PERFORMANCE, UNITS, combined, is_e2e
from tests.mla.paths import LEGACY_PATHS, ROOT, original_nodeid, relocated, work_dir
from tests.mla.support.mla_validation_policy import select_e2e_nodeids, select_nodeids


def test_every_moved_source_is_present_and_uniquely_owned():
    assert len(LEGACY_PATHS.values()) == len(set(LEGACY_PATHS.values()))
    assert all((ROOT / path).is_file() for path in LEGACY_PATHS.values())
    tests = {p for p in LEGACY_PATHS.values() if Path(p).name.startswith("test_")}
    owned = (
        combined(DIAGNOSTIC_E2E, DIAGNOSTIC_E2E)
        + combined(PERFORMANCE, PERFORMANCE)
        + combined(UNITS, UNITS)
    )
    assert tests <= set(owned)
    assert len(owned) == len(set(owned))
    assert is_e2e(combined(E2E, E2E))
    assert not is_e2e(combined(UNITS, UNITS))


def test_support_has_no_dependency_on_collected_test_modules():
    violations = []
    for path in (ROOT / "tests/mla/support").rglob("*.py"):
        tree = ast.parse(path.read_text())
        package = path.relative_to(ROOT).parts[:-1]
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else ()
                module = ".".join((*prefix, *([node.module] if node.module else [])))
                names = [module, *(module + "." + alias.name for alias in node.names)]
            if any(name.startswith(("tests.mla.unit.", "tests.mla.e2e.")) for name in names):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, violations


def test_common_tools_include_catalog_and_dependency_guards():
    from tests.mla.support.mla_validation import CPU_SUITES, SUITES

    assert {
        "tests/mla/unit/tools/test_matrix.py",
        "tests/mla/unit/tools/test_organization.py",
        "tests/mla/unit/tools/test_mla_validation_dependencies.py",
    } <= set(SUITES["tools"])
    assert "unit-tools" in CPU_SUITES
    assert set(SUITES["unit-tools"]) == set(UNITS["tools"])


def test_move_preserves_old_seed_and_selection_identity():
    old = [
        f"tests/test_flash_attn_npu_v3_mla_window_public.py::test_accuracy[fp16-{i}]"
        for i in range(200)
    ]
    moved = [relocated(node) for node in old]
    assert [original_nodeid(node) for node in moved] == old
    assert [original_nodeid(node) for node in select_nodeids(moved)] == select_nodeids(old)


def test_e2e_budget_covers_modules_and_dtypes_without_performance():
    nodes = [
        f"tests/mla/e2e/{module}/test_api.py::test_accuracy[{dtype}-{i}]"
        for module in E2E
        for dtype in ("fp16", "bf16")
        for i in range(30)
    ]
    assert select_e2e_nodeids(nodes) == nodes
    chosen = select_e2e_nodeids(nodes, 60)
    assert len(chosen) == 60 == len(set(chosen))
    assert {n.split("/")[3] for n in chosen} == set(E2E)
    for module in E2E:
        for dtype in ("fp16", "bf16"):
            assert any(f"/{module}/" in n and dtype in n for n in chosen)
    assert set(select_e2e_nodeids(list(reversed(nodes)), 60)) == set(chosen)
    assert all(not p.endswith("_profile.py") for p in combined(E2E, E2E))
    modules = list(E2E)
    assert combined(modules + modules, E2E) == combined(modules, E2E)


def test_default_cli_is_one_e2e_group_and_external_artifacts(tmp_path):
    output = subprocess.check_output(
        [sys.executable, "-m", "tests.mla", "--dry-run"],
        cwd=ROOT,
        env=dict(os.environ, MLA_WORK_DIR=str(tmp_path)),
        text=True,
    )
    preview = json.loads(output)
    assert list(preview["suites"]) == ["e2e"]
    assert preview["max_cases_per_suite"] == 0
    assert preview["full_matrix"] is False
    assert Path(preview["artifacts"]).is_relative_to(tmp_path)
    assert not any(
        "/unit/" in arg or arg.endswith("_profile.py") for arg in preview["suites"]["e2e"]
    )
    assert not list(tmp_path.iterdir())  # Dry run must not build, collect or create artifacts.


def test_work_directory_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MLA_WORK_DIR", str(tmp_path))
    assert work_dir() == tmp_path


def test_build_directory_is_reserved_for_headless_logs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.mla",
            "--dry-run",
            "--output",
            str(ROOT / "build/not-allowed"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "reserved for headless logs" in result.stderr
    assert not (ROOT / "build/not-allowed").exists()


def test_native_package_requires_intact_content(monkeypatch, tmp_path):
    import pytest
    from tests.mla.support import native_package as native

    monkeypatch.setenv("MLA_NATIVE_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="Build the native"):
        native.native_vendor()
    vendor = tmp_path / "opp/vendors" / native.VENDOR
    vendor.mkdir(parents=True)
    binary = vendor / "kernel.o"
    binary.write_bytes(b"test kernel")
    recipe = Path(native.__file__)
    record = dict(
        commit=native.COMMIT,
        recipe=native.sha256(recipe),
        inputs={p.name: native.sha256(p) for p in recipe.with_name("native_inputs").glob("*.json")},
        files={str(binary.relative_to(tmp_path)): native.sha256(binary)},
    )
    manifest = tmp_path / "package.json"
    manifest.write_text(json.dumps(record))
    assert native.native_vendor() == vendor
    binary.write_bytes(b"changed kernel")
    with pytest.raises(RuntimeError, match="package changed"):
        native.native_vendor()
    binary.unlink()
    with pytest.raises(RuntimeError, match="package changed"):
        native.native_vendor()
    record["recipe"] = "old"
    manifest.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="recipe changed"):
        native.native_vendor()


def test_daily_e2e_cost_boundary_is_explicit_even_below_budget():
    import pytest

    nodes = [
        "tests/mla/e2e/window_paged/test_api.py::test_representative[fp16]",
        "tests/mla/e2e/window_paged/test_api.py::test_cost_boundary[fp16]",
    ]
    assert select_e2e_nodeids(nodes, 1) == nodes[:1]
    assert select_e2e_nodeids(nodes, 0) == nodes
    with pytest.raises(ValueError):
        select_e2e_nodeids(nodes, -1)
    with pytest.raises(ValueError):
        select_e2e_nodeids(nodes + nodes)
