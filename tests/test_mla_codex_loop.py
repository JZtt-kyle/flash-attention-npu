"""Exercise the headless loop with real temporary Git repos and a fake CLI."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

_spec = importlib.util.spec_from_file_location(
    "mla_codex_loop", Path(__file__).resolve().parents[1] / "scripts/mla_codex_loop.py"
)
loop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loop)


SESSION = "12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def setup_loop(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for key, value in (
        ("user.name", "Loop test"),
        ("user.email", "loop@example.invalid"),
        ("commit.gpgsign", "false"),
    ):
        loop.git(repo, "config", key, value)
    (repo / ".gitignore").write_text("build/\n")
    (repo / "work").write_text("before\n")
    loop.git(repo, "add", ".")
    loop.git(repo, "commit", "-qm", "baseline")
    base = repo / "build/mla-codex-loop"
    directory = base / "run"
    directory.mkdir(parents=True)
    pending = dict(
        kind="development",
        head_before=loop.git(repo, "rev-parse", "HEAD"),
        branch=loop.git(repo, "branch", "--show-current"),
        session_id=None,
        attempts=0,
        run_dir="run",
        model=None,
        profile=None,
    )
    state = dict(version=3, repo=str(repo), development_runs=0, reviewed_runs=0, pending=pending)
    path = base / "state.json"
    loop.save_state(path, state)
    cli = tmp_path / "codex"
    cli.write_text("""#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
mode = os.environ["MLA_LOOP_TEST_MODE"]
print(json.dumps({"type":"thread.started","thread_id":"12345678-1234-1234-1234-123456789abc"}), flush=True)
answer = pathlib.Path(args[args.index("--output-last-message") + 1])
answer.write_text("source precision conflict; validation remains incomplete\\n")
if mode == "commit":
    pathlib.Path("work").write_text("after\\n")
    subprocess.run(["git", "add", "work"], check=True)
    subprocess.run(["git", "commit", "-qm", "step complete"], check=True)
if mode == "failure":
    print(json.dumps({"type":"turn.failed"}), flush=True)
    sys.exit(7)
if mode != "incomplete":
    print(json.dumps({"type":"turn.completed"}), flush=True)
""")
    cli.chmod(0o755)
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "blocked")
    with (base / "loop.lock").open("a") as lock:
        args = argparse.Namespace(
            repo=repo,
            model=None,
            profile=None,
            retry_blocked=False,
            lock_fd=lock.fileno(),
            max_dev_runs=1,
            dry_run=False,
            review_docs=False,
            interval=0,
            resume=None,
        )
        yield args, str(cli), state, path


def result_files(path):
    directory = path.parent / "run"
    return directory, json.loads((directory / "result.json").read_text())


def test_success_counts_once_and_does_not_start_next_step(setup_loop, monkeypatch):
    args, cli, state, path = setup_loop
    before = state["pending"]["head_before"]
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "commit")
    loop.run_loop(args, cli, path, state)
    assert state["pending"] is None and state["development_runs"] == 1
    assert loop.git(args.repo, "rev-list", "--count", f"{before}..HEAD") == "1"
    directory, result = result_files(path)
    assert result["accepted"] and result["status"] == "completed"
    assert len(list(directory.glob("attempt-*"))) == 1


def test_blocked_prints_reason_preserves_state_and_refuses_automatic_retry(setup_loop, capsys):
    args, cli, state, path = setup_loop
    with pytest.raises(RuntimeError, match="blocked"):
        loop.run_loop(args, cli, path, state)
    assert "source precision conflict" in capsys.readouterr().out
    directory, result = result_files(path)
    assert result["status"] == "blocked" and result["turn_completed"] and result["exit_code"] == 0
    assert not result["accepted"] and state["development_runs"] == 0
    assert (directory / "last-message.txt").read_text().startswith("source precision conflict")
    snapshot = path.read_bytes()
    with pytest.raises(RuntimeError, match="未调用模型"):
        loop.run_loop(args, cli, path, state)
    assert "source precision conflict" in capsys.readouterr().out
    assert path.read_bytes() == snapshot and len(list(directory.glob("attempt-*"))) == 1


def test_explicit_retry_uses_original_session(setup_loop, monkeypatch):
    args, cli, state, path = setup_loop
    with pytest.raises(RuntimeError):
        loop.run_loop(args, cli, path, state)
    args.retry_blocked = True
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "commit")
    loop.run_loop(args, cli, path, state)
    directory, result = result_files(path)
    command = json.loads((directory / result["attempt"] / "command.json").read_text())
    assert "resume" in command and SESSION in command
    assert "上次执行中断" not in command[-1]
    assert state["development_runs"] == 1 and state["pending"] is None


@pytest.mark.parametrize("mode", ("failure", "incomplete"))
def test_failed_or_incomplete_turn_remains_retryable(setup_loop, monkeypatch, capsys, mode):
    args, cli, state, path = setup_loop
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", mode)
    with pytest.raises(RuntimeError):
        loop.run_loop(args, cli, path, state)
    assert "source precision conflict" in capsys.readouterr().out
    directory, result = result_files(path)
    assert result["status"] == "interrupted" and not loop.blocked_result(directory)
    assert state["development_runs"] == 0
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "commit")
    loop.run_loop(args, cli, path, state)
    assert state["development_runs"] == 1


def test_legacy_no_commit_result_blocks_without_rewriting_attempt(setup_loop):
    args, cli, state, path = setup_loop
    directory = path.parent / "run"
    state["pending"].update(attempts=2, session_id=SESSION)
    legacy = dict(
        exit_code=0,
        accepted=False,
        error="RuntimeError: 本轮开发没有产生提交；保留原会话，重启后继续本轮",
    )
    loop.save_state(directory / "result.json", legacy)
    with pytest.raises(RuntimeError, match="未调用模型"):
        loop.run_session(args, cli, state, path)
    assert not list(directory.glob("attempt-*"))
    assert json.loads((directory / "result.json").read_text()) == legacy


def test_accepted_receipt_recovers_count_without_calling_cli(setup_loop, monkeypatch):
    args, cli, state, path = setup_loop
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "commit")
    loop.run_session(args, cli, state, path)
    assert state["pending"] is not None and state["development_runs"] == 0
    loop.run_loop(args, "/missing-cli", path, state)
    assert state["pending"] is None and state["development_runs"] == 1


def test_new_head_allows_old_session_to_verify_existing_commit(setup_loop, monkeypatch):
    args, cli, state, path = setup_loop
    with pytest.raises(RuntimeError):
        loop.run_session(args, cli, state, path)
    (args.repo / "work").write_text("externally completed\n")
    loop.git(args.repo, "add", "work")
    loop.git(args.repo, "commit", "-qm", "external completion")
    head = loop.git(args.repo, "rev-parse", "HEAD")
    loop.run_loop(args, cli, path, state)
    assert loop.git(args.repo, "rev-parse", "HEAD") == head
    assert state["pending"] is None and state["development_runs"] == 1


def test_branch_drift_is_rejected_before_cli(setup_loop):
    args, cli, state, path = setup_loop
    loop.git(args.repo, "checkout", "-qb", "other")
    with pytest.raises(RuntimeError, match="分支已改变"):
        loop.run_loop(args, "/missing-cli", path, state)
    assert state["pending"]["attempts"] == 0


def test_retirement_preserves_files_and_does_not_count_success(setup_loop):
    args, cli, state, path = setup_loop
    with pytest.raises(RuntimeError):
        loop.run_session(args, cli, state, path)
    directory, result = result_files(path)
    args.retire_pending = "completed separately in the interactive handoff"
    loop.retire_pending(args, state, path)
    assert state["pending"] is None and state["development_runs"] == 0
    assert not json.loads((directory / "retired.json").read_text())["accepted"]
    assert json.loads((directory / "result.json").read_text()) == result
    assert (directory / result["attempt"] / "last-message.txt").exists()
