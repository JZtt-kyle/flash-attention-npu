"""Exercise failure propagation, evidence integrity and resumption without NPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests import mla_validation as runner
from tests import mla_validation_build as builder
from tests.mla_validation_build import matching_files


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
                "from pathlib import Path; from tests.mla_validation import locked; "
                f"\nwith locked(Path({str(path)!r})): pass",
            ],
            cwd=runner.ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "Another validation process" in result.stderr


def test_success_failure_interruption_and_resume(tmp_path, monkeypatch):
    config = args(tmp_path)
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
    first = json.loads((config.output / "summary.json").read_text())["groups"]["tools"]
    config.resume = True
    assert runner.run(config) == 0
    assert (
        json.loads((config.output / "summary.json").read_text())["groups"]["tools"]["status"]
        == "inherited"
    )
    Path(first["directory"], "cases.json").unlink()
    exit_file.write_text("7")
    assert runner.run(config) == 7
    assert "tools" not in json.loads((config.output / "state.json").read_text())
    exit_file.write_text("130")
    assert runner.run(config) == 130
    exit_file.write_text("0")
    assert runner.run(config) == 0
    mutable_sources["source"] = "changed"
    assert runner.run(config) == 0
    assert (
        json.loads((config.output / "summary.json").read_text())["groups"]["tools"]["status"]
        == "passed"
    )
    (config.output / "state.json").write_text("interrupted index write")
    assert runner.run(config) == 0
    assert list(config.output.glob("attempt-*/invalid-state.json"))


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
