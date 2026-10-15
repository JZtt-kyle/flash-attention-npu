"""Coverage, marks, exact case identities and opt-in exhaustive diagnostics."""

from itertools import combinations, product
import json
import os
import subprocess
import sys
from types import SimpleNamespace

from tests.mla.catalog import E2E, combined
from tests.mla.matrix import PAGES, compact_calls, covering_rows
from tests.mla.paths import ROOT
from tests.mla.regression_cases import BEHAVIORS, daily_cases


def test_pair_coverage_for_every_dtype_and_constrained_tuple():
    rows = [
        dict(dtype=d, a=a, b=b, c=c)
        for d, a, b, c in product(range(2), range(3), range(4), range(3))
    ]
    selected = [rows[i] for i in covering_rows(rows, contexts=("dtype",))]
    assert len(selected) < len(rows)
    for dtype in range(2):
        for a, b in combinations(("a", "b", "c"), 2):
            assert {(r[a], r[b]) for r in selected if r["dtype"] == dtype} == {
                (r[a], r[b]) for r in rows if r["dtype"] == dtype
            }
    # A hand-written coupled tuple is not turned into invalid Cartesian inputs.
    constrained = [r for r in rows if r["a"] == r["c"]]
    assert all(constrained[i] in rows for i in covering_rows(constrained, contexts=("dtype",)))


def test_representative_pages_public_adapters_and_fault_values():
    entries = ("public", "op", "eager", "cpp", "public_fake", "op_fake", "eager_fake")
    calls = [
        SimpleNamespace(
            params=dict(dtype=dtype, page=page, entry=entry),
            indices=dict(dtype=di, page=pi, entry=ei),
        )
        for di, dtype in enumerate(("fp16", "bf16"))
        for pi, page in enumerate(range(16, 1025, 16))
        for ei, entry in enumerate(entries)
    ]
    chosen = compact_calls(calls, public=True)
    assert len(chosen) == len(PAGES) * 2 * 2
    assert {c.params["page"] for c in chosen} == set(PAGES)
    assert {c.params["entry"] for c in chosen} == {"public", "public_fake"}
    faults = [
        SimpleNamespace(params=dict(fault=f), indices=dict(fault=i))
        for i, f in enumerate(("negative", "overflow", "stride", "stale", "empty"))
    ]
    assert compact_calls(faults) == faults


def test_shape_boundaries_keep_values_and_extreme_corners():
    rows = [dict(dtype=d, m=m, sk=sk) for d, m, sk in product(range(2), range(9), range(15))]
    chosen = [rows[i] for i in covering_rows(rows, contexts=("dtype",), independent=("m", "sk"))]
    assert len(chosen) < len(rows) // 2
    for dtype in range(2):
        subset = [r for r in chosen if r["dtype"] == dtype]
        assert {r["m"] for r in subset} == set(range(9))
        assert {r["sk"] for r in subset} == set(range(15))
        assert set(product((0, 8), (0, 14))) <= {(r["m"], r["sk"]) for r in subset}


def test_daily_catalog_is_explicit_and_does_not_collect_diagnostics():
    assert set(BEHAVIORS) <= set(E2E)
    selections = combined(E2E, E2E)
    for case in daily_cases():
        assert any(node.endswith(f"test_public_behavior[{'-'.join(case)}]") for node in selections)
    assert all(node.startswith("tests/mla/e2e/") for node in selections)
    assert all(
        "::" in node or node == "tests/mla/e2e/interface/test_forward.py" for node in selections
    )
    assert not any("cost_boundary" in node or "profile" in node for node in selections)
    preview = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "tests.mla", "--dry-run", "--e2e", "window_paged"],
            cwd=ROOT,
            text=True,
        )
    )
    assert preview["max_cases_per_suite"] == 0
    assert all(node in preview["suites"]["e2e"] for node in E2E["window_paged"])


def test_pytest_compaction_preserves_fixtures_marks_ids_and_full_mode(tmp_path):
    module = tmp_path / "test_pairs.py"
    module.write_text("""import pytest
@pytest.fixture(scope="module", params=("fp16", "bf16"))
def dtype(request): return request.param
@pytest.mark.parametrize("a", [pytest.param(0, marks=pytest.mark.xfail(strict=True), id="expected-failure"), 1, 2])
@pytest.mark.parametrize("b", range(3))
@pytest.mark.parametrize("c", range(3))
def test_values(dtype, a, b, c):
    assert dtype in ("fp16", "bf16")
    assert a != 0
""")
    snapshots = []
    for mode in ("compact", "full"):
        output = tmp_path / mode
        output.mkdir()
        env = dict(
            os.environ,
            MLA_VALIDATION_RESULT=str(output),
            MLA_VALIDATION_MAX_CASES="0",
            MLA_FULL_MATRIX="1" if mode == "full" else "0",
            MLA_VALIDATION_NPU="0",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-c",
                "/dev/null",
                "-o",
                "addopts=",
                "-p",
                "tests.mla.conftest",
                "-p",
                "tests.mla.support.mla_validation_plugin",
                str(module),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "xfailed" in result.stdout
        snapshots.append(set(json.loads((output / "cases.json").read_text())))
    assert len(snapshots[1]) == 2 * 3 * 3 * 3
    assert snapshots[0] < snapshots[1]
