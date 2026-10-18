"""Exercise the headless loop with real temporary Git repos and a fake CLI."""

from tests.mla.paths import ROOT

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys

import pytest

_spec = importlib.util.spec_from_file_location("mla_codex_loop", ROOT / "scripts/mla_codex_loop.py")
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
        started_clean=True,
    )
    state = dict(version=3, repo=str(repo), development_runs=0, reviewed_runs=0, pending=pending)
    path = base / "state.json"
    loop.save_state(path, state)
    cli = tmp_path / "codex"
    cli.write_text("""#!/usr/bin/env python3
import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
mode = os.environ["MLA_LOOP_TEST_MODE"]
kind = "documentation" if "专门进行一次 MLA 文档修正优化" in args[-1] else "development"
if mode == "cycle":
    mode = os.environ.get("MLA_LOOP_DOC_MODE", "no-change") if kind == "documentation" else "commit"
with pathlib.Path(os.environ["MLA_LOOP_CALLS"]).open("a") as log:
    log.write(json.dumps({"kind": kind, "args": args}) + "\\n")
print(json.dumps({"type":"thread.started","thread_id":"12345678-1234-1234-1234-123456789abc"}), flush=True)
answer = pathlib.Path(args[args.index("--output-last-message") + 1])
answer.write_text("source precision conflict; validation remains incomplete\\n")
if mode == "commit":
    target = pathlib.Path("docs.md" if kind == "documentation" else "work")
    with target.open("a") as output:
        output.write("after\\n")
    subprocess.run(["git", "add", str(target)], check=True)
    subprocess.run(["git", "commit", "-qm", "step complete"], check=True)
if mode == "failure":
    print(json.dumps({"type":"turn.failed"}), flush=True)
    sys.exit(7)
if mode != "incomplete":
    print(json.dumps({"type":"turn.completed"}), flush=True)
""")
    cli.chmod(0o755)
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "blocked")
    monkeypatch.setenv("MLA_LOOP_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.delenv("MLA_LOOP_DOC_MODE", raising=False)
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


def calls(args):
    return [
        json.loads(line) for line in (args.repo.parent / "calls.jsonl").read_text().splitlines()
    ]


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


@pytest.mark.parametrize("leftover", ("work", "untracked.py"))
def test_clean_round_cannot_leave_uncommitted_work(setup_loop, monkeypatch, leftover):
    args, cli, state, path = setup_loop
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "commit")
    # Simulate a partial commit: the CLI commits work but leaves another edit.
    target = args.repo / leftover
    original = loop.validate_commit

    def leave_edit(repo, pending):
        target.write_text("unfinished\n")
        return original(repo, pending)

    monkeypatch.setattr(loop, "validate_commit", leave_edit)
    with pytest.raises(RuntimeError, match="仍有未提交改动"):
        loop.run_loop(args, cli, path, state)
    assert state["development_runs"] == 0 and state["pending"] is not None
    _, result = result_files(path)
    assert not result["accepted"]
    assert target.read_text() == "unfinished\n"


def test_preexisting_work_and_legacy_rounds_remain_recoverable(setup_loop, monkeypatch):
    args, cli, state, path = setup_loop
    state["pending"].pop("started_clean")
    existing = args.repo / "user-work.py"
    existing.write_text("preserve me\n")
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "commit")
    loop.run_loop(args, cli, path, state)
    assert state["development_runs"] == 1 and state["pending"] is None
    assert existing.read_text() == "preserve me\n"


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


@pytest.mark.parametrize("limit", (4, 5, 6, 10))
def test_every_five_successful_developments_reviews_even_at_limit(setup_loop, monkeypatch, limit):
    args, cli, state, path = setup_loop
    args.max_dev_runs = limit
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "cycle")
    loop.run_loop(args, cli, path, state)
    expected = []
    for number in range(1, limit + 1):
        expected.append("development")
        if number % 5 == 0:
            expected.append("documentation")
    assert [call["kind"] for call in calls(args)] == expected
    assert state["development_runs"] == limit
    assert state["reviewed_runs"] == limit // 5 * 5
    assert state["pending"] is None


def test_review_counter_accumulates_across_invocations(setup_loop, monkeypatch):
    args, cli, state, path = setup_loop
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "cycle")
    args.max_dev_runs = 3
    loop.run_loop(args, cli, path, state)
    restored = loop.prepare_state(args, loop.read_state(path, args.repo), path)
    args.max_dev_runs = 2
    loop.run_loop(args, cli, path, restored)
    assert [call["kind"] for call in calls(args)] == ["development"] * 5 + ["documentation"]
    assert restored["development_runs"] == restored["reviewed_runs"] == 5


@pytest.mark.parametrize("version,dev,reviewed", ((1, 5, 0), (2, 77, 45), (3, 77, 45)))
def test_old_overdue_state_reviews_once_before_new_development(
    setup_loop, monkeypatch, version, dev, reviewed
):
    args, cli, state, path = setup_loop
    state.update(version=version, development_runs=dev, reviewed_runs=reviewed, pending=None)
    loop.save_state(path, state)
    restored = loop.prepare_state(args, loop.read_state(path, args.repo), path)
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "cycle")
    loop.run_loop(args, cli, path, restored)
    assert [call["kind"] for call in calls(args)] == ["documentation", "development"]
    assert restored["reviewed_runs"] == dev and restored["development_runs"] == dev + 1


def test_overdue_pending_development_finishes_before_review(setup_loop, monkeypatch):
    args, cli, state, path = setup_loop
    state.update(development_runs=77, reviewed_runs=45)
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "cycle")
    loop.run_loop(args, cli, path, state)
    assert [call["kind"] for call in calls(args)] == ["development", "documentation"]
    assert state["development_runs"] == state["reviewed_runs"] == 78


@pytest.mark.parametrize("mode", ("no-change", "commit"))
def test_manual_review_runs_once_and_resets_interval(setup_loop, monkeypatch, mode):
    args, cli, state, path = setup_loop
    before = state["pending"]["head_before"]
    state.update(development_runs=3, pending=None)
    args.review_docs, args.max_dev_runs = True, 0
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "cycle")
    monkeypatch.setenv("MLA_LOOP_DOC_MODE", mode)
    loop.run_loop(args, cli, path, state)
    assert [call["kind"] for call in calls(args)] == ["documentation"]
    assert state["development_runs"] == state["reviewed_runs"] == 3
    count = "1" if mode == "commit" else "0"
    assert loop.git(args.repo, "rev-list", "--count", f"{before}..HEAD") == count
    args.review_docs, args.max_dev_runs = False, 4
    loop.run_loop(args, cli, path, state)
    assert [call["kind"] for call in calls(args)] == ["documentation"] + ["development"] * 4
    assert state["reviewed_runs"] == 3


def test_manual_review_cannot_replace_pending_development(setup_loop):
    args, cli, state, path = setup_loop
    args.review_docs = True
    snapshot = path.read_bytes()
    with pytest.raises(RuntimeError, match="还有未完成开发轮次"):
        loop.run_loop(args, "/missing-cli", path, state)
    assert path.read_bytes() == snapshot


@pytest.mark.parametrize("mode", ("failure", "incomplete"))
def test_failed_automatic_review_resumes_same_session_then_continues(setup_loop, monkeypatch, mode):
    args, cli, state, path = setup_loop
    state.update(development_runs=5, pending=None)
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "cycle")
    monkeypatch.setenv("MLA_LOOP_DOC_MODE", mode)
    with pytest.raises(RuntimeError):
        loop.run_loop(args, cli, path, state)
    assert state["development_runs"] == 5 and state["reviewed_runs"] == 0
    assert state["pending"]["automatic_review"]
    directory = path.parent / state["pending"]["run_dir"]
    assert not json.loads((directory / "result.json").read_text())["accepted"]
    restored = loop.prepare_state(args, loop.read_state(path, args.repo), path)
    monkeypatch.delenv("MLA_LOOP_DOC_MODE")
    loop.run_loop(args, cli, path, restored)
    history = calls(args)
    assert [call["kind"] for call in history] == ["documentation", "documentation", "development"]
    assert "resume" in history[1]["args"] and SESSION in history[1]["args"]
    assert restored["development_runs"] == 6 and restored["reviewed_runs"] == 5
    assert len(list(directory.glob("attempt-*"))) == 2


@pytest.mark.parametrize("automatic", (None, False))
def test_legacy_or_manual_documentation_resume_exits_without_development(
    setup_loop, monkeypatch, automatic
):
    args, cli, state, path = setup_loop
    state["pending"].update(kind="documentation", session_id=SESSION)
    if automatic is not None:
        state["pending"]["automatic_review"] = automatic
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "cycle")
    loop.run_loop(args, cli, path, state)
    history = calls(args)
    assert [call["kind"] for call in history] == ["documentation"]
    assert "resume" in history[0]["args"]
    assert state["development_runs"] == 0


def test_automatic_review_receipt_recovers_once_and_manual_override_exits(setup_loop, monkeypatch):
    args, cli, state, path = setup_loop
    state.update(development_runs=5)
    state["pending"].update(kind="documentation", automatic_review=True)
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "cycle")
    loop.run_session(args, cli, state, path)
    assert state["reviewed_runs"] == 0
    restored = loop.prepare_state(args, loop.read_state(path, args.repo), path)
    args.review_docs = True
    loop.run_loop(args, "/missing-cli", path, restored)
    assert restored["reviewed_runs"] == restored["development_runs"] == 5
    assert restored["pending"] is None and len(calls(args)) == 1


@pytest.mark.parametrize("dev,reviewed", ((0, 0), (77, 45)))
def test_dry_run_previews_reviews_without_writing_state(setup_loop, capsys, dev, reviewed):
    args, cli, state, path = setup_loop
    state.update(development_runs=dev, reviewed_runs=reviewed, pending=None)
    args.dry_run, args.max_dev_runs = True, 0
    snapshot = path.read_bytes()
    loop.run_loop(args, cli, path, state)
    prompts = [shlex.split(line)[-1] for line in capsys.readouterr().out.splitlines()]
    expected = ([loop.DOC_PROMPT] if dev else []) + [loop.DEV_PROMPT] * 5 + [loop.DOC_PROMPT]
    assert prompts == expected
    assert path.read_bytes() == snapshot
    assert not (args.repo.parent / "calls.jsonl").exists()


@pytest.mark.parametrize("dev,due", ((4, False), (5, True)))
def test_check_reports_review_schedule_without_writing_state(
    setup_loop, monkeypatch, capsys, dev, due
):
    args, cli, state, path = setup_loop
    state.update(development_runs=dev, pending=None)
    loop.save_state(path, state)
    for name in ("AGENTS.md", "docs/mla/status.md", "docs/mla/plan.md"):
        target = args.repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n")
    snapshot = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["mla_codex_loop.py", "--repo", str(args.repo), "--check"])
    monkeypatch.setattr(loop.shutil, "which", lambda name: cli)
    monkeypatch.setattr(loop, "check_cli", lambda executable: {"version": "fake"})
    assert loop.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["automatic_doc_review"] and report["doc_review_interval"] == 5
    assert report["doc_review_due"] is due and report["reviewed_runs"] == 0
    assert path.read_bytes() == snapshot
    assert not (args.repo.parent / "calls.jsonl").exists()


def test_invalid_review_origin_is_rejected(setup_loop):
    args, cli, state, path = setup_loop
    state["pending"]["automatic_review"] = "false"
    loop.save_state(path, state)
    with pytest.raises(ValueError, match="未完成轮次格式错误"):
        loop.read_state(path, args.repo)


def write_next_step(args, state, value):
    status = args.repo / "docs/mla/status.md"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(value)
    loop.git(args.repo, "add", "docs/mla/status.md")
    loop.git(args.repo, "commit", "-qm", "handoff")
    if state["pending"]:
        state["pending"]["head_before"] = loop.git(args.repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("dry_run", (False, True))
def test_completed_scope_exits_without_cli_or_state_changes(setup_loop, capsys, dry_run):
    args, _, state, path = setup_loop
    state["pending"] = None
    args.dry_run, args.max_dev_runs = dry_run, 0
    write_next_step(args, state, "<!-- mla-loop-next: none -->\n")
    loop.save_state(path, state)
    snapshot = path.read_bytes()
    loop.run_loop(args, "/missing-cli", path, state)
    assert "无下一开发步" in capsys.readouterr().out
    assert path.read_bytes() == snapshot and state["development_runs"] == 0
    assert not (args.repo.parent / "calls.jsonl").exists()


@pytest.mark.parametrize("pending", (False, True))
def test_completion_still_finishes_pending_round_and_due_review(setup_loop, monkeypatch, pending):
    args, cli, state, path = setup_loop
    if not pending:
        state["pending"] = None
    state["development_runs"] = 5
    args.max_dev_runs = 0
    write_next_step(args, state, "<!-- mla-loop-next: none -->\n")
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "cycle")
    loop.run_loop(args, cli, path, state)
    expected = (["development"] if pending else []) + ["documentation"]
    assert [call["kind"] for call in calls(args)] == expected
    assert state["pending"] is None
    assert state["development_runs"] == state["reviewed_runs"] == 5 + int(pending)


def test_scope_completion_is_reread_after_final_commit(setup_loop, monkeypatch):
    args, cli, state, path = setup_loop
    args.max_dev_runs = 0
    write_next_step(args, state, "<!-- mla-loop-next: M21 -->\n")
    # The fake developer closes the scope in its sole commit, as the real prompt requires.
    executable = loop.Path(cli)
    code = executable.read_text().replace(
        'subprocess.run(["git", "add", str(target)], check=True)',
        'pathlib.Path("docs/mla/status.md").write_text("<!-- mla-loop-next: none -->\\n")\n'
        '    subprocess.run(["git", "add", str(target), "docs/mla/status.md"], check=True)',
    )
    executable.write_text(code)
    monkeypatch.setenv("MLA_LOOP_TEST_MODE", "commit")
    loop.run_loop(args, cli, path, state)
    assert state["development_runs"] == 1 and state["pending"] is None
    assert len(calls(args)) == 1


@pytest.mark.parametrize(
    "content",
    (
        "<!-- mla-loop-next: -->\n",
        "<!-- mla-loop-next: M18 -->\n<!-- mla-loop-next: none -->\n",
        "<!-- mla-loop-next: none --> trailing\n",
    ),
)
def test_invalid_scope_marker_rejected_before_cli(setup_loop, content):
    args, _, state, path = setup_loop
    write_next_step(args, state, content)
    snapshot = path.read_bytes()
    with pytest.raises(RuntimeError, match="标记格式错误或重复"):
        loop.run_loop(args, "/missing-cli", path, state)
    assert path.read_bytes() == snapshot
    assert not (args.repo.parent / "calls.jsonl").exists()


@pytest.mark.parametrize("step,expected", (("M18", "ready"), ("none", "complete")))
def test_check_reports_scope_without_starting_model(
    setup_loop, monkeypatch, capsys, step, expected
):
    args, cli, state, path = setup_loop
    state["pending"] = None
    write_next_step(
        args, state, f"<!-- mla-loop-next: {step} -->\n下一步由 `mla-loop-next` 表示。\n"
    )
    for name in ("AGENTS.md", "docs/mla/plan.md"):
        (args.repo / name).write_text("fixture\n")
    loop.save_state(path, state)
    snapshot = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["mla_codex_loop.py", "--repo", str(args.repo), "--check"])
    monkeypatch.setattr(loop.shutil, "which", lambda name: cli)
    monkeypatch.setattr(loop, "check_cli", lambda executable: {"version": "fake"})
    assert loop.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == expected and report["next_step"] == step
    assert path.read_bytes() == snapshot
    assert not (args.repo.parent / "calls.jsonl").exists()
