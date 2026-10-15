"""Scoped resumption must retain imports, fixtures and shared native sources."""

import pytest

from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_validation_dependencies import python_dependencies, suite_sources
from tests.mla.support.mla_validation_selection import COVERAGE, INCREMENTAL_SUITES, PARENTS


def source_tree(tmp_path):
    files = {
        "tests/__init__.py": "",
        "tests/conftest.py": "from tests.test_fixture import helper\n",
        "tests/test_fixture.py": "helper = 1\n",
        "tests/test_selected.py": "from tests import test_helper\nfrom .test_relative import value\n",
        "tests/test_helper.py": "from tests.common import equation\n",
        "tests/test_relative.py": "value = 1\n",
        "tests/common/__init__.py": "",
        "tests/common/equation.py": "",
        "tests/test_unrelated.py": "",
        "tests/mla_driver.py": "",
        "tests/common/unrelated.py": "",
        "csrc/ascend910/shared.hpp": "",
        "scripts/mla_codex_loop.py": "",
    }
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return {name: "first" for name in files}


def test_unrelated_test_and_headless_changes_preserve_numerical_identity(tmp_path):
    baseline = source_tree(tmp_path)
    selection = ["tests/test_selected.py::test_example[fp16]"]
    before = suite_sources(tmp_path, selection, baseline)
    unrelated = {
        "tests/test_unrelated.py",
        "scripts/mla_codex_loop.py",
        "tests/mla_driver.py",
        "tests/common/unrelated.py",
    }
    for path in unrelated:
        changed = dict(baseline, **{path: "changed"})
        assert suite_sources(tmp_path, selection, changed) == before
    for path in set(baseline) - unrelated:
        changed = dict(baseline, **{path: "changed"})
        assert suite_sources(tmp_path, selection, changed) != before, path
    assert "scripts/mla_codex_loop.py" in suite_sources(
        tmp_path, selection, baseline, headless=True
    )


def test_dynamic_unknown_import_falls_back_and_new_static_import_is_tracked(tmp_path):
    baseline = source_tree(tmp_path)
    path = tmp_path / "tests/test_selected.py"
    selection = ["tests/test_selected.py"]
    path.write_text("import importlib\nimportlib.import_module(variable)\n")
    assert suite_sources(tmp_path, selection, baseline) == baseline
    path.write_text('command = ["python", "-m", "tests.test_unrelated"]\n')
    assert "tests/test_unrelated.py" in suite_sources(tmp_path, selection, baseline)


def test_shared_helper_changes_invalidate_only_its_consumers(tmp_path):
    baseline = source_tree(tmp_path)
    helper = "tests/common/equation.py"
    before = suite_sources(tmp_path, ["tests/test_selected.py"], baseline)
    changed = dict(baseline, **{helper: "changed without changing mtime"})
    assert suite_sources(tmp_path, ["tests/test_selected.py"], changed) != before
    unrelated = suite_sources(tmp_path, ["tests/test_unrelated.py"], baseline)
    assert suite_sources(tmp_path, ["tests/test_unrelated.py"], changed) == unrelated


def test_production_build_tracks_its_helpers_without_cpu_or_profile_builders():
    from tests.mla.paths import ROOT

    baseline = {str(path.relative_to(ROOT)): None for path in (ROOT / "tests").rglob("*.py")}
    baseline["setup.py"] = None
    dependencies = python_dependencies(
        ROOT, ["tests/mla/support/mla_validation_build.py"], baseline
    )
    assert {
        "tests/mla/support/mla_validation_build.py",
        "tests/mla/support/mla_dense_build.py",
        "tests/mla/support/mla_metadata_build.py",
        "tests/mla/support/mla_validation_dependencies.py",
        "tests/mla/paths.py",
    } <= set(dependencies)
    assert not any("/test_" in path or "/public_cases/" in path for path in dependencies)
    assert "tests/mla/support/mla_window_build.py" not in dependencies


@pytest.mark.parametrize("owner", ("tests/__init__.py", "tests/common/conftest.py"))
def test_package_and_nested_fixtures_import_dependencies(tmp_path, owner):
    baseline = source_tree(tmp_path)
    (tmp_path / owner).write_text("from tests.test_unrelated import helper\n")
    baseline[owner] = "new"
    selected = suite_sources(tmp_path, ["tests/test_selected.py"], baseline)
    assert "tests/test_unrelated.py" in selected


@pytest.mark.parametrize(
    "statement",
    ("exec(code)", "runpy.run_path(path)", 'importlib.util.spec_from_file_location("name", path)'),
)
def test_dynamic_file_execution_cannot_narrow_sources(tmp_path, statement):
    baseline = source_tree(tmp_path)
    (tmp_path / "tests/test_selected.py").write_text(statement)
    assert suite_sources(tmp_path, ["tests/test_selected.py"], baseline) == baseline


def test_incremental_selection_is_explicit_unique_and_has_coverage_reasons():
    assert set(INCREMENTAL_SUITES) == set(PARENTS)
    for name, nodes in INCREMENTAL_SUITES.items():
        assert nodes and len(nodes) == len(set(nodes))
        assert all("::test_" in node and not node.startswith("-") for node in nodes)
        assert all(entry["reason"] and entry["nodes"] for entry in COVERAGE[name])
        for node in nodes:
            if "[fp16" in node:
                assert node.replace("[fp16", "[bf16") in nodes


def test_real_numerical_closures_track_builders_without_headless_or_unrelated_tests():
    baseline = runner.sources()
    for name in INCREMENTAL_SUITES:
        selected = suite_sources(runner.ROOT, runner.SUITES[name], baseline)
        assert "scripts/mla_codex_loop.py" not in selected
        assert "tests/mla/unit/tools/test_mla_codex_loop.py" not in selected
        assert selected["setup.py"] == baseline["setup.py"]
        assert "tests/mla/support/mla_metadata_build.py" in selected
        assert "tests/mla/support/mla_ref.py" in selected
        assert "tests/mla/support/components/dense.py" in selected
