"""Exercise failure propagation, evidence integrity and resumption without NPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.mla.support import mla_validation as runner
from tests.mla.support import mla_validation_build as builder
from tests.mla.support.mla_validation_build import matching_files


def args(tmp_path, **extra):
    return argparse.Namespace(
        output=tmp_path / "run",
        artifacts=tmp_path / "artifacts",
        suite=["tools"],
        device="6",
        golden="cache",
        dry_run=False,
        collect_only=False,
        resume=False,
        **extra,
    )


def evidence(directory, code=0, skip=False, empty=False):
    directory.mkdir(parents=True, exist_ok=True)
    runner.save(directory / "cases.json", [] if empty else ["test_example.py::test_case"])
    inner = "<skipped/>" if skip else "<failure/>" if code else ""
    (directory / "results.xml").write_text(
        f'<testsuites><testsuite><testcase name="test_case">{inner}</testcase></testsuite></testsuites>'
    )


@pytest.mark.parametrize(
    "code,skip,empty", [(0, False, False), (1, False, False), (0, True, False), (0, False, True)]
)
def test_exit_code_alone_does_not_prove_pass(tmp_path, code, skip, empty):
    evidence(tmp_path, code, skip, empty)
    result = runner.validate_result(tmp_path, code)
    assert (result["status"] == "passed") == (code == 0 and not skip and not empty)


def test_loaded_extension_must_match(tmp_path):
    evidence(tmp_path)
    runner.save(tmp_path / "loaded.json", dict(path="wrong.so", sha256="bad"))
    with pytest.raises(RuntimeError, match="different extension"):
        runner.validate_result(tmp_path, 0, dict(extension="expected.so", extension_sha256="good"))


def test_evidence_missing_changed_failed_or_different_identity_cannot_resume(tmp_path):
    file = tmp_path / "evidence"
    file.write_text("passed")
    record = dict(status="passed", identity="same", evidence={str(file): runner.sha256(file)})
    assert runner.reusable(record, "same")
    assert not runner.reusable(record, "changed")
    assert not runner.reusable(dict(record, status="failed"), "same")
    file.write_text("modified")
    assert not runner.reusable(record, "same")
    file.unlink()
    assert not runner.reusable(record, "same")


def test_dependency_content_change_invalidates_even_with_same_mtime(tmp_path):
    file = tmp_path / "dependency.h"
    file.write_text("old")
    before = file.stat()
    identity = {str(file): runner.sha256(file)}
    file.write_text("new")
    os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not matching_files(identity)


def test_build_reuses_only_intact_outputs_and_dependencies(tmp_path, monkeypatch):
    directory = tmp_path / "artifacts"
    header = tmp_path / "external.h"
    header.write_text("old")
    compiler = {"version": "one"}
    calls = []
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "build_inputs", lambda: dict(compiler))

    def compile_extension(command, **kwargs):
        assert kwargs["env"]["FLASH_ATTN_FORCE_REBUILD"] == "TRUE"
        calls.append(command)
        target = tmp_path / "flash_attn_npu_3"
        target.mkdir(exist_ok=True)
        (target / "extension.so").write_bytes(b"extension")
        deps = tmp_path / "build/lib.test/flash_attn_npu_3"
        deps.mkdir(parents=True, exist_ok=True)
        (deps / "metadata.o.d").write_text(f"metadata.o: {header}\n")
        (deps / "obsolete.o.d").write_text("obsolete.o: /missing/historical/header.h\n")
        kwargs["stdout"].write(
            f"[link-cmd] bisheng {deps / 'metadata.o'} -o {target / 'extension.so'}\n"
        )

    def probe(path, *a, **k):
        path.mkdir(parents=True, exist_ok=True)
        (path / "probe.so").write_bytes(b"probe")
        (path / "build-manifest.json").write_text("{}")

    monkeypatch.setattr(builder.subprocess, "run", compile_extension)
    monkeypatch.setattr(builder, "build_probes", probe)
    monkeypatch.setattr(builder, "build_probe", probe)
    builder.prepare(directory)
    builder.prepare(directory)
    assert len(calls) == 1
    header.write_text("new")
    builder.prepare(directory)
    assert len(calls) == 2
    (directory / "producer/probe.so").write_bytes(b"broken")
    builder.prepare(directory)
    assert len(calls) == 3
    compiler["version"] = "two"
    builder.prepare(directory)
    assert len(calls) == 4
    (directory / "prepare.json").write_text("incomplete")
    builder.prepare(directory)
    assert len(calls) == 5


def test_lock_blocks_a_second_process(tmp_path):
    path = tmp_path / "lock"
    with runner.locked(path):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; from tests.mla.support.mla_validation import locked; "
                f"\nwith locked(Path({str(path)!r})): pass",
            ],
            cwd=runner.ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "Another validation process" in result.stderr


@pytest.mark.parametrize(
    "suite",
    [
        "e2e",
        "tools",
        "window-device-tools",
        "window-paged-tools",
        "window-mixed-tools",
        "window-tnd-tools",
        "window-paged-cpu",
        "window-mixed-cpu",
        "window-tnd-cpu",
        "window-paged-sanitize",
        "window-mixed-sanitize",
        "window-tnd-sanitize",
        "window-bsnd-tools",
        "window-bsnd-cpu",
        "window-bsnd-sanitize",
        "window-cpu",
        "window-sanitize",
        "causal-cpu",
        "causal-sanitize",
        "causal-bsnd-cpu",
        "causal-bsnd-sanitize",
        "used-shared-cpu",
        "used-shared-sanitize",
    ],
)
def test_success_failure_interruption_and_resume(tmp_path, monkeypatch, suite):
    if suite == "e2e":
        monkeypatch.setattr(runner, "CPU_SUITES", runner.CPU_SUITES | {suite})
    config = args(tmp_path)
    config.suite = [suite]
    dependencies = {"compiler": "first", "sanitize": suite == "causal-sanitize"}
    monkeypatch.setattr(runner, "suite_dependencies", lambda _: dict(dependencies))
    mutable_sources = {"source": "first"}
    monkeypatch.setattr(runner, "sources", lambda: dict(mutable_sources))
    script = tmp_path / "child.py"
    script.write_text("""import json, os, pathlib, sys
p = pathlib.Path(os.environ["MLA_VALIDATION_RESULT"])
code = int(pathlib.Path(sys.argv[1]).read_text())
(p / "cases.json").write_text(json.dumps(["example::test_ok"]))
detail = "<failure/>" if code else ""
(p / "results.xml").write_text('<testsuites><testsuite><testcase name="test_ok">' + detail + '</testcase></testsuite></testsuites>')
sys.exit(code)
""")
    exit_file = tmp_path / "exit"
    exit_file.write_text("0")
    monkeypatch.setattr(
        runner, "pytest_command", lambda *a: [sys.executable, str(script), str(exit_file)]
    )
    assert runner.run(config) == 0
    first = json.loads((config.output / "summary.json").read_text())["groups"][suite]
    config.resume = True
    assert runner.run(config) == 0
    assert json.loads((config.output / "summary.json").read_text())["groups"][suite]["status"] == (
        "inherited" if runner.is_e2e(runner.SUITES[suite]) else "passed"
    )
    Path(first["directory"], "cases.json").unlink()
    exit_file.write_text("7")
    assert runner.run(config) == 7
    assert suite not in json.loads((config.output / "state.json").read_text())
    exit_file.write_text("130")
    assert runner.run(config) == 130
    exit_file.write_text("0")
    assert runner.run(config) == 0
    mutable_sources["source"] = "changed"
    assert runner.run(config) == 0
    assert (
        json.loads((config.output / "summary.json").read_text())["groups"][suite]["status"]
        == "passed"
    )
    (config.output / "state.json").write_text("interrupted index write")
    assert runner.run(config) == 0
    assert list(config.output.glob("attempt-*/invalid-state.json"))
    dependencies["compiler"] = "changed"
    assert runner.run(config) == 0
    assert (
        json.loads((config.output / "summary.json").read_text())["groups"][suite]["status"]
        == "passed"
    )
    if suite == "tools":
        config.max_cases = 7
        assert runner.run(config) == 0
        assert (
            json.loads((config.output / "summary.json").read_text())["groups"][suite]["status"]
            == "passed"
        )


def test_environment_cannot_inherit_sampling_or_old_fixture_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "--random-sample=1 -k nothing")
    monkeypatch.setenv("MLA_PAGED_ATTENTION_DIR", "/old/probes")
    config = args(tmp_path)
    env = runner.environment(config, tmp_path, "public")
    assert env["PYTEST_ADDOPTS"] == ""
    assert env["MLA_PAGED_ATTENTION_DIR"] == str(tmp_path / "probes")
    assert "--random-sample=0" in runner.pytest_command("public", tmp_path)
    assert runner.SUITES["public"] == [runner.PUBLIC, "-k", "not public_release_trap"]


def test_dry_run_does_not_write_or_import_npu(tmp_path, monkeypatch, capsys):
    config = args(tmp_path)
    config.dry_run = True
    config.suite = ["public"]
    monkeypatch.setattr(runner, "execute", lambda *a: pytest.fail("dry-run executed a process"))
    assert runner.run(config) == 0
    assert "not public_release_trap" in capsys.readouterr().out
    assert not config.output.exists()


@pytest.mark.parametrize(
    "suite,sanitize",
    [
        ("window-cpu", "0"),
        ("window-sanitize", "1"),
        ("causal-cpu", "0"),
        ("causal-sanitize", "1"),
        ("used-shared-cpu", "0"),
        ("used-shared-sanitize", "1"),
    ],
)
def test_causal_selection_and_isolated_cpu_environment(
    tmp_path, monkeypatch, capsys, suite, sanitize
):
    config = args(tmp_path)
    config.suite = [suite]
    config.dry_run = True
    monkeypatch.setenv("MLA_SCHEDULE_ARTIFACT_DIR", "/old/causal")
    monkeypatch.setenv("MLA_SCHEDULE_SANITIZE", "unexpected")
    monkeypatch.setattr(runner, "execute", lambda *a: pytest.fail("dry-run executed"))
    assert runner.run(config) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["prepare"] == []
    expected = (
        runner.USED_SHARED_CPU
        if suite.startswith("used-shared")
        else [runner.WINDOW_CPU]
        if suite.startswith("window-")
        else [runner.CAUSAL_CPU]
    )
    assert runner.SUITES[suite] == expected
    assert all(path in preview["suites"][suite] for path in expected)
    env = runner.environment(config, tmp_path, suite)
    assert env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert env["MLA_VALIDATION_NPU"] == "0"
    assert env["MLA_SCHEDULE_SANITIZE"] == sanitize
    assert env["MLA_SCHEDULE_ARTIFACT_DIR"] == str(tmp_path / "schedule")


@pytest.mark.parametrize(
    "suite",
    [
        "causal-cpu",
        "window-cpu",
        "window-sanitize",
        "window-bsnd-cpu",
        "window-bsnd-sanitize",
        "window-paged-cpu",
        "window-mixed-cpu",
        "window-tnd-cpu",
        "window-paged-sanitize",
        "window-mixed-sanitize",
        "window-tnd-sanitize",
    ],
)
def test_causal_dependencies_changing_during_execution_fail(tmp_path, monkeypatch, suite):
    config = args(tmp_path)
    config.suite = [suite]
    dependencies = {"source": "original"}
    monkeypatch.setattr(runner, "sources", lambda: {})
    monkeypatch.setattr(runner, "suite_dependencies", lambda _: dict(dependencies))

    def execute(command, env, directory, lock_fds):
        evidence(directory)
        dependencies["source"] = "modified"
        return dict(returncode=0, seconds=0)

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.run(config) == 1
    group = json.loads((config.output / "summary.json").read_text())["groups"][suite]
    assert group["status"] == "failed"
    assert "fixed source changed" in group["error"]
    assert suite not in json.loads((config.output / "state.json").read_text())


@pytest.mark.parametrize("mode", ["ok", "exit", "stderr", "eof", "malformed", "extra"])
@pytest.mark.parametrize("pooled", [False, True])
def test_causal_driver_checks_protocol_and_final_sanitizer_exit(tmp_path, mode, pooled):
    from tests.mla.support.mla_causal_build import CpuDriver
    from tests.mla.support.mla_cpu_driver import CpuDriverPool

    binary = tmp_path / "driver"
    binary.write_text(f"""#!{sys.executable}
import json, sys
mode = {mode!r}
for line in sys.stdin:
    if mode == "eof":
        sys.exit(0)
    print("invalid JSON" if mode == "malformed" else json.dumps(list(map(int, line.split()))), flush=True)
if mode == "stderr":
    print("sanitizer diagnostic", file=sys.stderr)
if mode == "extra":
    print("[]", flush=True)
sys.exit(7 if mode == "exit" else 0)
""")
    binary.chmod(0o755)

    def exercise():
        if pooled:
            with CpuDriverPool() as pool:
                assert pool.evaluate(binary, [[1, 2]]) == [[1, 2]]
                assert pool.evaluate(binary, [[3]]) == [[3]]
            return
        with CpuDriver(binary) as driver:
            assert driver.evaluate([[1, 2], [3]]) == [[1, 2], [3]]

    if mode == "ok":
        exercise()
    else:
        with pytest.raises((RuntimeError, ValueError)):
            exercise()
    records = list(tmp_path.glob("driver*-run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["returncode"] == (7 if mode == "exit" else 0)
    assert record["requests"] == (0 if mode in {"eof", "malformed"} else 2)


def test_cpu_pool_reuses_only_matching_executable_arguments_and_environment(tmp_path):
    from tests.mla.support.mla_cpu_driver import CpuDriverPool

    binary = tmp_path / "driver"
    source = f"""#!{sys.executable}
import json, os, sys
for line in sys.stdin:
    print(json.dumps([int(sys.argv[1]), int(os.environ['MLA_TEST_MODE']),
                      *[int(x) for x in line.split()]]), flush=True)
"""
    binary.write_text(source)
    binary.chmod(0o755)
    with CpuDriverPool() as pool:
        env = {"MLA_TEST_MODE": "1", "PYTEST_CURRENT_TEST": "first"}
        assert pool.evaluate(binary, [[3]], args=[1], env=env) == [[1, 1, 3]]
        env["PYTEST_CURRENT_TEST"] = "second"
        assert pool.evaluate(binary, [[4]], args=[1], env=env) == [[1, 1, 4]]
        assert len(pool.drivers) == 1
        assert pool.evaluate(binary, [[5]], args=[2], env=env) == [[2, 1, 5]]
        env["MLA_TEST_MODE"] = "2"
        assert pool.evaluate(binary, [[6]], args=[2], env=env) == [[2, 2, 6]]
        replacement = tmp_path / "replacement"
        replacement.write_text(source.replace("int(x)", "2 * int(x)"))
        replacement.chmod(0o755)
        replacement.replace(binary)
        assert pool.evaluate(binary, [[7]], args=[2], env=env) == [[2, 2, 14]]
        assert len(pool.drivers) == 4
    records = [json.loads(p.read_text()) for p in tmp_path.glob("driver*-run.json")]
    assert sorted(row["requests"] for row in records) == [1, 1, 1, 2]
    assert all(row["returncode"] == 0 for row in records)


@pytest.mark.parametrize(
    "suite",
    (
        "causal-smoke",
        "causal-attention",
        "causal-profile",
        "window-check",
        "window-attention",
        "window-profile",
        "window-mixed-check",
        "window-tnd-check",
        "window-mixed-public",
        "window-tnd-public",
        "window-bsnd-check",
        "window-bsnd-public",
    ),
)
def test_causal_device_groups_use_new_probe_and_device_identity(tmp_path, capsys, suite):
    config = args(tmp_path)
    config.suite = [suite]
    config.dry_run = True
    assert runner.run(config) == 0
    preview = json.loads(capsys.readouterr().out)
    assert bool(preview["prepare"]) == (suite not in runner.STANDALONE_CHECK_SUITES)
    assert suite not in runner.CPU_SUITES
    selection = preview["suites"][suite]
    kind = "window" if suite in runner.WINDOW_DEVICE_SUITES else "causal"
    assert any(kind in path and path.endswith(".py") for path in selection)
    assert runner.CAUSAL_CPU not in selection
    env = runner.environment(config, tmp_path, suite)
    assert env["MLA_VALIDATION_NPU"] == "1"
    assert env[f"MLA_{kind.upper()}_ATTENTION_DIR"] == str(tmp_path / "probes")
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "6"


def test_shared_regression_preserves_each_layout():
    selected = {
        path
        for suite in ("shared-dense", "shared-nonpaged", "shared-paged")
        for path in runner.SUITES[suite]
        if path.endswith(".py")
    }
    assert selected == {
        runner.relocated(f"tests/test_flash_attn_npu_v3_mla_{name}.py")
        for name in (
            "dense",
            "multiblock",
            "kvblocks",
            "longkv",
            "bsnd_attention",
            "tnd_attention",
            "tnd_bsnd_attention",
            "used_attention",
            "bsnd_stride",
            "tnd_stride",
            "paged_attention",
            "tnd_paged_attention",
        )
    }
    assert "task_ring" in runner.SUITES["shared-paged"][-1]
    assert "zero" in runner.SUITES["shared-paged"][-1]


@pytest.mark.parametrize("suite", ("causal-smoke", "window-check"))
def test_standalone_causal_smoke_never_requires_old_production_artifacts(
    tmp_path, monkeypatch, suite
):
    config = args(tmp_path)
    config.suite = [suite]
    # This is a simulated device run. The outer acceptance process may hold
    # the actual card lock, so exercise real file locks in an isolated tree.
    real_locked = runner.locked
    monkeypatch.setattr(
        runner, "locked", lambda path: real_locked(tmp_path / "locks" / str(path).lstrip("/"))
    )
    monkeypatch.setattr(runner, "sources", lambda: {"source": "fixed"})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "npu-smi test\n")
    checked = []
    monkeypatch.setattr(runner, "validate_causal_builds", lambda d, **kw: checked.append((d, kw)))
    calls = []
    fail = False

    def execute(command, env, directory, lock_fds=()):
        calls.append(command)
        if "runtime" in command:
            runner.save(
                Path(command[-1]),
                dict(extension="test.so", extension_sha256="hash", loaded_libraries={}),
            )
        else:
            evidence(directory, code=int(fail))
            runner.save(directory / "loaded.json", dict(path="test.so", sha256="hash"))
        code = int(fail and "runtime" not in command)
        return dict(returncode=code, seconds=0, status="failed" if code else "passed")

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.run(config) == 0
    assert checked and not (config.artifacts / "prepare.json").exists()
    assert all("prepare" not in command for command in calls)
    summary = json.loads((config.output / "summary.json").read_text())
    assert summary["build"]["status"] == "not_required"
    assert summary["groups"][suite]["status"] == "passed"
    fail = True
    assert runner.run(config) == 1
    assert suite not in json.loads((config.output / "state.json").read_text())


@pytest.mark.parametrize("fault", ("source", "dependency", "library", "loaded", "missing"))
@pytest.mark.parametrize("kind", ("causal", "window"))
def test_causal_probe_identity_change_is_not_accepted(tmp_path, fault, kind):
    paths = {}
    for dtype in ("fp16", "bf16"):
        base = tmp_path / "probes" / dtype
        base.mkdir(parents=True)
        parts = {}
        for name in ("source", "dependency", "library"):
            path = base / name
            path.write_text("before")
            parts[name] = path
        paths[dtype] = parts
        libs = {
            "migrated": dict(path=str(parts["library"]), sha256=runner.sha256(parts["library"]))
        }
        runner.save(base / "loaded.json", libs)
        runner.save(
            base / "build-manifest.json",
            dict(
                **{kind: True},
                dtype=dtype,
                libraries=libs,
                sources={str(parts["source"]): runner.sha256(parts["source"])},
                dependencies={str(parts["dependency"]): runner.sha256(parts["dependency"])},
            ),
        )
    runner.validate_causal_builds(tmp_path, kind=kind)
    base = tmp_path / "probes" / "bf16"
    if fault == "loaded":
        runner.save(base / "loaded.json", {})
    elif fault == "missing":
        paths["bf16"]["library"].unlink()
    else:
        paths["bf16"][fault].write_text("after")
    with pytest.raises(RuntimeError):
        runner.validate_causal_builds(tmp_path, kind=kind)


@pytest.mark.parametrize(
    "suite",
    (
        "causal-profile",
        "shared-profile",
        "window-profile",
        "window-bsnd-profile",
        "window-paged-profile",
        "window-mixed-profile",
        "window-tnd-profile",
    ),
)
def test_performance_always_executes_on_resume(tmp_path, monkeypatch, suite):
    config = args(tmp_path)
    config.suite = [suite]
    monkeypatch.setattr(runner, "CPU_SUITES", runner.CPU_SUITES | {suite})
    monkeypatch.setattr(runner, "sources", lambda: {"source": "fixed"})
    monkeypatch.setattr(runner, "validate_causal_builds", lambda _, **kw: None)
    calls = []

    def execute(command, env, directory, lock_fds=()):
        calls.append(command)
        evidence(directory)
        return dict(returncode=0, seconds=0)

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.run(config) == 0
    config.resume = True
    assert runner.run(config) == 0
    assert len(calls) == 2
    summary = json.loads((config.output / "summary.json").read_text())
    assert summary["groups"][suite]["status"] == "passed"
