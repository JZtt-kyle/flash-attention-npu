#!/usr/bin/env python3
"""按 status/plan 串行完成单提交开发；中断后恢复同一个 Codex 会话。

用法（Linux / Python 3.9+，先完成 codex login）：
    python3 mla_codex_loop.py --check
    python3 mla_codex_loop.py --max-dev-runs 1
    python3 mla_codex_loop.py --dry-run --max-dev-runs 5
    python3 mla_codex_loop.py --resume <旧版日志中的会话 ID>
    python3 mla_codex_loop.py --retry-blocked
    python3 mla_codex_loop.py --review-docs
    python3 mla_codex_loop.py --retire-pending '说明旧轮次为何被当前工作替代'

新轮次使用 codex exec；未完成轮次使用 codex exec resume <明确的 ID>。
沿用登录和模型配置，--approve-for-me 使用自动审批及 workspace-write 沙箱。
需要支持该参数的 Codex CLI（已核对 0.163.0）。
计数、未完成轮次、会话 ID 和日志保存在目标仓库 build/mla-codex-loop/。
网络错误或中断后重新运行本脚本即可恢复，失败轮不会计数或自动另开对话。
正常结束但没有提交的开发轮记为 blocked；先解决原因，再用 --retry-blocked 明确恢复。
--resume 用于接管旧版遗留会话；多个旧会话未完成时必须明确指定 ID。
恢复的是对话历史和磁盘工作，不会还原被终止进程的内存或自动续跑测试进程。
开发轮必须恰好新增一个线性提交；文档轮允许没有提交。
从干净工作区开始的轮次必须干净结束，不能把漏提交的源码留给下一轮。
本脚本的成功检查不替代 AGENTS.md 要求的测试和验收。
文档随每步更新；自上次文档轮起每成功完成五轮开发，自动专门修正优化文档再继续。
开发次数上限不包含文档轮；到达上限时先完成已到期文档轮再退出。
旧计数已超期时补做一次；未完成轮次优先恢复，失败不推进计数。
--review-docs 显式运行一次文档核验后退出；--check 不启动模型或修改状态。
--retire-pending 仅归档待办轮次并退出，不计成功、不删除源码/日志或开启开发。
status.md 可用 <!-- mla-loop-next: M18 --> 标记唯一下一步；完成时改为 none。
none 时先处理待恢复轮次及已到期文档轮，然后正常退出，不创建空开发轮。
旧交接没有该标记时保留按次数运行的行为；标记格式错误或重复时停止。

Codex 非交互接口：https://learn.chatgpt.com/docs/non-interactive-mode
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid


DOC_REVIEW_INTERVAL = 5


LEGACY_PROMPTS = {
    "development": "继续MLA迁移开发",
    "documentation": (
        "MLA经过多轮开发后，文档可能已经腐化，与实际不符，结构不清晰。"
        "请你对文档进行核验，优化当前的文档"
    ),
}
DEV_PROMPT = (
    "继续 MLA 迁移开发。先读 AGENTS.md、docs/mla/status.md 和 plan.md 的唯一下一步，"
    "核对当前分支、工作树、已有进程和闭环证据；当前文档优先于旧会话计划。"
    "用户已确认的范围无需重复确认。默认一轮交付一个完整可调用行为，将同一布局的 host、AICPU、Kernel、"
    "metadata、公开 real/fake 入口及必要验收合为一次提交；稳定的 mask/split 随布局接入，不按实现层或开关拆轮。"
    "仅新增数值算法或共享协议有独立未覆盖风险时，才在 plan 写明风险和完成条件后拆出组件步。"
    "优先复用源算法、现有组件和驱动，不为任务收敛全面重构已验收路径。完成实现、必要验证、状态更新后"
    "提交一次并立即停止。接手/收尾核对 pytest 用例清单、JUnit、实际扩展身份及原始日志；"
    "不重复手写全树审计；长测前先核对廉价夹具和代表用例。"
    "测试使用 python3 -m pytest -q tests/test_flash_attn_npu_v3_mla.py，"
    "只保留公开功能 E2E，参照仓库普通测试风格；不恢复已退役的单元、性能或私有验收驱动。"
    "只继承依赖未变且完整有效的 E2E 正确性证据，不按固定数量填充或截断，"
    "回归失败使用 -k 或精确节点定位。"
    "复用数学 reference 缓存；开发自检、代表检查和提交验收不重复计算相同 reference。"
    "build/ 只保留 mla-codex-loop 无头日志和状态；其他产物放仓库外 MLA_WORK_DIR。不要把旧无 mask 测试"
    "当成新增能力验收。文档随本步更新，阶段结束或契约改变时按影响范围核对。"
    "循环会在每五轮成功开发后另开专门的文档修正优化轮。"
    "若 status 有 mla-loop-next 标记，随本步更新其唯一下一步；只有已授权范围全部验收完成才设为 none。"
    "没有已授权的下一步时停止，不自行扩大范围或制造空提交。"
    "验收无法完成时如实更新交接并停止，不能提交为已闭环。"
)
DOC_PROMPT = (
    "专门进行一次 MLA 文档修正优化，处理开发累积的腐化、冗余和错误。"
    "先读 AGENTS.md、docs/mla/status.md 和 plan.md，再核对 design.md 与相关代码、固定来源及验收证据。"
    "修正过期或矛盾的能力描述、步骤状态、命令、路径和链接；合并重复内容，压缩已完成步骤的历史细节，"
    "保持 design/plan/status 的契约、计划、当前交接三种职责及唯一下一步，历史从 Git 查询。"
    "保留 status 的 mla-loop-next 标记，并核对它与正文的下一步一致。"
    "用户最新约定优先；证据缺失、失败和未运行如实标记，不把文档陈述当作实现或验收依据。"
    "保留既有未提交工作，只修改本轮文档；不推进功能，不探测设备、重建扩展或重跑无关设备矩阵。"
    "完成一致性、链接和适用质量检查后至多提交一次并停止；无需修改时说明核对范围与结果。"
)
PROMPTS = {"development": DEV_PROMPT, "documentation": DOC_PROMPT}


class Stopped(BaseException):
    def __init__(self, signum):
        self.signum = signum


def stop(signum, _frame):
    raise Stopped(signum)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path, state):
    """在同目录原子替换 JSON，并同步文件及目录；调用者持有循环锁。"""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        temp_path = Path(f.name)
        try:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    try:
        temp_path.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp_path.unlink(missing_ok=True)


def read_state(path, repo):
    if not path.exists():
        state = {"repo": str(repo), "development_runs": 0, "reviewed_runs": 0}
    else:
        state = read_json(path)
    if not isinstance(state, dict):
        raise ValueError(f"状态文件格式错误：{path}")
    dev, reviewed = state.get("development_runs"), state.get("reviewed_runs")
    if (
        state.get("repo") != str(repo)
        or type(dev) is not int
        or type(reviewed) is not int
        or reviewed < 0
        or reviewed > dev
        or state.get("version", 1) not in (1, 2, 3)
    ):
        raise ValueError(f"状态文件计数、版本或仓库不匹配：{path}")
    pending = state.setdefault("pending", None)
    if pending is not None:
        if (
            not isinstance(pending, dict)
            or pending.get("kind") not in PROMPTS
            or not isinstance(pending.get("run_dir"), str)
            or Path(pending["run_dir"]).name != pending["run_dir"]
            or pending["run_dir"] in ("", ".", "..")
            or not isinstance(pending.get("head_before"), str)
            or not re.fullmatch(r"[0-9a-f]{40,64}", pending["head_before"])
            or type(pending.get("attempts")) is not int
            or pending["attempts"] < 0
            or type(pending.get("automatic_review", False)) is not bool
            or type(pending.get("started_clean", False)) is not bool
        ):
            raise ValueError(f"未完成轮次格式错误：{path}")
        if pending.get("session_id"):
            uuid.UUID(pending["session_id"])
    return state


def legacy_runs(base):
    """只读旧版日志；成功轮的提交号可定位随后中断轮的起点。"""
    runs, previous_head = [], None
    for directory in sorted(base.glob("*-*")):
        if not directory.is_dir() or not (directory / "command.json").exists():
            continue
        result_path = directory / "result.json"
        result = read_json(result_path) if result_path.exists() else {}
        if result.get("exit_code") == 0:
            previous_head = result.get("head_after")
            continue
        log = directory / "session.log"
        with log.open(encoding="utf-8", errors="replace") as f:
            header = f.read(8192)
        ids = re.findall(r"^session id: ([0-9a-f-]{36})\s*$", header, re.M)
        prompt = (directory / "prompt.txt").read_text(encoding="utf-8").strip()
        kind = next((key for key, value in LEGACY_PROMPTS.items() if value == prompt), None)
        if not ids or not kind:
            raise RuntimeError(f"旧轮次缺少可恢复的会话 ID 或提示词，请检查 {directory}")
        runs.append(
            {
                "run_dir": directory.name,
                "kind": kind,
                "head_before": result.get("head_before") or previous_head,
                "session_id": str(uuid.UUID(ids[0])),
                "attempts": 0,
            }
        )
    return runs


def rollout_head(repo, session_id):
    """首次旧轮次没有 result.json 时，只读该会话自己的元数据获取起点。"""
    codex_dir = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    for path in (codex_dir / "sessions").glob(f"**/*{session_id}*.jsonl"):
        with path.open(encoding="utf-8") as f:
            metadata = json.loads(f.readline()).get("payload", {})
        if metadata.get("id") == session_id and metadata.get("cwd") == str(repo):
            return metadata.get("git", {}).get("commit_hash")
    return None


def recover_id(directory):
    ids = set()
    for path in directory.glob("attempt-*/events.jsonl"):
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # 进程可能在写最后一行时被终止。
                if isinstance(event, dict) and event.get("type") == "thread.started":
                    ids.add(str(uuid.UUID(event["thread_id"])))
    if len(ids) > 1:
        raise RuntimeError(f"同一轮出现多个会话 ID，请检查 {directory}")
    return next(iter(ids), None)


def prepare_state(args, state, path):
    """选择恢复目标；本函数不写磁盘，dry-run 也可以调用。"""
    pending = state["pending"]
    if pending:
        recovered = recover_id(path.parent / pending["run_dir"])
        saved = pending.get("session_id")
        if recovered and saved and recovered != saved:
            raise RuntimeError("状态与事件日志的会话 ID 不一致")
        pending["session_id"] = saved or recovered
        if args.resume:
            if pending["session_id"] and args.resume != pending["session_id"]:
                raise RuntimeError(f"已有未完成会话 {pending['session_id']}，不能替换为另一会话")
            pending["session_id"] = args.resume
    elif args.resume or state.get("version", 1) == 1:
        runs = legacy_runs(path.parent)
        if args.resume:
            runs = [run for run in runs if run["session_id"] == args.resume]
            if not runs:
                raise RuntimeError("指定 ID 不在旧版未完成轮次中，不能重复计数已完成会话")
        if len(runs) > 1:
            choices = "\n".join(f"  {r['session_id']}  {r['run_dir']}" for r in runs)
            raise RuntimeError(f"发现多个旧版未完成会话，请用 --resume ID 指定：\n{choices}")
        if runs:
            pending = runs[0]
            pending["head_before"] = (
                rollout_head(args.repo, pending["session_id"]) or pending["head_before"]
            )
            if not pending["head_before"]:
                raise RuntimeError("无法确定旧轮次开始时的提交号，请先核对旧会话记录")
            state["pending"] = pending
    state["version"] = 3
    return state


def validate_start(repo, pending):
    """在付出模型开销前拒绝漂移的恢复基线；允许已提交但尚未收尾。"""
    if not pending:
        return
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if pending.get("branch") and pending["branch"] != branch:
        raise RuntimeError("分支已改变，请切回原分支或核对后用 --retire-pending 原因 归档旧轮次")
    before, after = pending["head_before"], git(repo, "rev-parse", "HEAD")
    if before != after and git(repo, "rev-list", "--parents", "-n", "1", "HEAD").split() != [
        after,
        before,
    ]:
        raise RuntimeError(
            f"待恢复轮次的基线 {before[:12]} 已偏离当前 HEAD {after[:12]}；"
            "请核对已有工作，再用 --retire-pending 原因 归档旧轮次。不会自动调用旧会话。"
        )


def retire_pending(args, state, path):
    pending = state["pending"]
    if not pending:
        raise RuntimeError("没有待归档的未完成轮次")
    reason = args.retire_pending.strip()
    if not reason:
        raise ValueError("归档原因不能为空")
    directory = path.parent / pending["run_dir"]
    if (directory / "result.json").exists() and read_json(directory / "result.json").get(
        "accepted"
    ):
        raise RuntimeError("该轮已有成功回执，应先恢复补计数，不能归档为未完成")
    record = directory / "retired.json"
    if record.exists():
        previous = read_json(record)
        if previous["state"]["pending"] != pending:
            raise RuntimeError("已有归档与待办状态不一致")
    else:
        save_state(
            record,
            {
                "reason": reason,
                "head_when_retired": git(args.repo, "rev-parse", "HEAD"),
                "state": state,
                "accepted": False,
            },
        )
    state["pending"] = None
    save_state(path, state)
    print(f"已归档未完成轮次：{record}；计数不变，源码及会话日志保留。", flush=True)


def command(args, codex, pending, attempt):
    cmd = [codex, "exec", "--approve-for-me", "--cd", str(args.repo), "--color", "never"]
    for name in ("model", "profile"):
        value = getattr(args, name) or pending.get(name)
        if value:
            cmd += [f"--{name}", value]
    session_id = pending.get("session_id")
    if session_id:
        cmd.append("resume")
    cmd += ["--json", "--output-last-message", str(attempt / "last-message.txt")]
    prompt = PROMPTS[pending["kind"]]
    if session_id:
        prompt = (
            f"请恢复本轮尚未完成的任务：{prompt}\n"
            f"本轮开始时的 Git 提交是 {pending['head_before']}。"
            "请先核对上次最终说明、当前文档、代码、进程及已有验证结果。"
            "上次可能是验收阻塞或进程中断，按实际原因继续；已确认的新契约优先于旧会话结论。"
            "若本轮已经完成并提交，请核对该提交并给出最终总结，不再开启下一步或重复提交。"
            "仍遵循 AGENTS.md 的验收要求；无法确认完成的测试不能记为通过。"
        )
        cmd.append(session_id)
    return cmd + [prompt]


def validate_commit(repo, pending):
    validate_start(repo, pending)
    before, after = pending["head_before"], git(repo, "rev-parse", "HEAD")
    if after == before:
        if pending["kind"] == "development":
            raise RuntimeError(
                "本轮开发正常结束但没有产生提交，已记录为 blocked；先处理最终说明中的原因，再用 --retry-blocked 恢复"
            )
    elif git(repo, "rev-list", "--parents", "-n", "1", "HEAD").split() != [after, before]:
        raise RuntimeError("相对本轮最初起点并非只新增一个线性提交，请核对后再恢复")
    if pending.get("started_clean") and git(
        repo, "status", "--porcelain", "--untracked-files=normal"
    ):
        raise RuntimeError("本轮从干净工作区开始，但结束时仍有未提交改动；保留本轮，不开始下一轮")
    return after


def completed_receipt(args, pending, directory):
    path = directory / "result.json"
    result = read_json(path) if path.exists() else {}
    if not result.get("accepted"):
        return False
    if (
        result.get("session_id") != pending.get("session_id")
        or result.get("head_before") != pending["head_before"]
        or result.get("head_after") != validate_commit(args.repo, pending)
    ):
        raise RuntimeError("完成记录与当前会话/提交不一致，不能自动补计数")
    print("发现已完成但尚未计数的轮次，只补记一次计数。", flush=True)
    return True


def blocked_result(directory):
    path = directory / "result.json"
    result = read_json(path) if path.exists() else {}
    # Preserve old attempts and recognize the former no-commit error without
    # misclassifying transport errors or incomplete turns as task blockers.
    if result.get("accepted"):
        return None
    if result.get("status") == "blocked" or (
        result.get("exit_code") == 0
        and result.get("error", "").startswith("RuntimeError: 本轮开发没有产生提交")
    ):
        return result
    return None


def show_answer(directory, attempt):
    answer = attempt / "last-message.txt"
    if answer.exists():
        text = answer.read_text(encoding="utf-8")
        (directory / "last-message.txt").write_text(text, encoding="utf-8")
        print(text, flush=True)


def check_blocked(args, pending, directory):
    result = blocked_result(directory)
    if (
        result
        and not args.retry_blocked
        and git(args.repo, "rev-parse", "HEAD") == pending["head_before"]
    ):
        attempt = result.get("attempt")
        if isinstance(attempt, str) and Path(attempt).name == attempt:
            show_answer(directory, directory / attempt)
        raise RuntimeError(
            "待恢复轮次被验收或任务条件阻塞，未调用模型。"
            "处理最终说明中的原因后，用 --retry-blocked 恢复同一会话；源码和日志保留。"
        )


def terminate(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def run_session(args, codex, state, path):
    pending = state["pending"]
    directory = path.parent / pending["run_dir"]
    if completed_receipt(args, pending, directory):
        return
    check_blocked(args, pending, directory)
    if pending["attempts"] and not pending.get("session_id"):
        raise RuntimeError("上次启动后未取得会话 ID；请查日志并用 --resume ID 绑定，避免另开对话")
    attempt = Path(
        tempfile.mkdtemp(prefix=f"attempt-{pending['attempts'] + 1:04d}-", dir=directory)
    )
    cmd = command(args, codex, pending, attempt)
    (attempt / "prompt.txt").write_text(cmd[-1] + "\n", encoding="utf-8")
    save_state(attempt / "command.json", cmd)
    pending["attempts"] += 1
    save_state(path, state)  # 在启动模型前持久化本轮起点和恢复信息。
    label = f"恢复 {pending['session_id']}" if pending.get("session_id") else "新建对话"
    print(f"[{pending['kind']}] {label}\n过程日志：{directory / 'session.log'}", flush=True)
    completed, child = False, None
    result = {
        "head_before": pending["head_before"],
        "accepted": False,
        "attempt": attempt.name,
        "status": "interrupted",
    }
    try:
        with (
            (directory / "session.log").open("ab", buffering=0) as log,
            (attempt / "events.jsonl").open("w", encoding="utf-8") as events,
        ):
            log.write(f"\n--- {attempt.name}: {label} ---\n".encode())
            child = subprocess.Popen(
                cmd,
                cwd=args.repo,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
                pass_fds=(args.lock_fd,),
            )
            try:
                for line in child.stdout:
                    events.write(line)
                    events.flush()
                    log.write(line.encode("utf-8"))
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "thread.started":
                        session_id = str(uuid.UUID(event["thread_id"]))
                        if pending.get("session_id") and pending["session_id"] != session_id:
                            raise RuntimeError("CLI 返回了不同会话，已停止，避免误续另一个对话")
                        pending["session_id"] = session_id
                        save_state(path, state)
                        (directory / "session-id.txt").write_text(session_id + "\n")
                        print(f"会话 ID：{session_id}", flush=True)
                    elif event.get("type") == "turn.completed":
                        completed = True
                    elif event.get("type") == "turn.failed":
                        completed = False
                result["exit_code"] = child.wait()
            except BaseException:
                terminate(child)
                raise
            finally:
                child.stdout.close()
        # Always surface the explanation, even when the turn failed or the
        # agent deliberately stopped without a commit after failed acceptance.
        show_answer(directory, attempt)
        result["turn_completed"] = completed
        if result["exit_code"]:
            raise RuntimeError(f"Codex 退出码 {result['exit_code']}；重启脚本将恢复本轮会话")
        if not pending.get("session_id") or not completed:
            raise RuntimeError("没有收到会话 ID 或 turn.completed，保留本轮以便恢复")
        result["head_after"] = git(args.repo, "rev-parse", "HEAD")
        if pending["kind"] == "development" and result["head_after"] == pending["head_before"]:
            result["status"] = "blocked"
        else:
            result["status"] = "failed"
        result["head_after"] = validate_commit(args.repo, pending)
        result["status"] = "completed"
        result["accepted"] = True
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["exit_code"] = child.poll() if child else None
        raise
    finally:
        result["session_id"] = pending.get("session_id")
        save_state(attempt / "result.json", result)
        # 先持久化验收结果，随后 run_loop 原子更新计数并清除 pending。
        save_state(directory / "result.json", result)
        save_state(path, state)


def doc_review_due(state):
    return state["development_runs"] - state["reviewed_runs"] >= DOC_REVIEW_INTERVAL


def next_step(repo):
    """只读交接中的结束信号；不从模型最终回答猜测任务是否完成。"""
    status = repo / "docs/mla/status.md"
    if not status.exists():
        return None
    content = status.read_text(encoding="utf-8")
    markers = re.findall(r"<!--\s*mla-loop-next\b", content)
    if not markers:
        return None  # Compatible with handoffs written before this marker existed.
    matches = re.findall(r"^<!-- mla-loop-next: ([A-Za-z][A-Za-z0-9_-]*) -->$", content, re.M)
    if len(matches) != 1 or len(markers) != 1:
        raise RuntimeError("status.md 的 mla-loop-next 标记格式错误或重复，请核对唯一下一步")
    return matches[0]


def run_loop(args, codex, path, state):
    completed = 0
    limit = args.max_dev_runs or (5 if args.dry_run else 0)
    while True:
        pending = state["pending"]
        validate_start(args.repo, pending)
        step = next_step(args.repo)
        if pending and args.review_docs and pending["kind"] != "documentation":
            raise RuntimeError("还有未完成开发轮次，不能插入文档核验")
        if pending:
            kind = pending["kind"]
        elif args.review_docs or doc_review_due(state):
            kind = "documentation"
        elif step == "none":
            print("已授权范围已完成，无下一开发步；保留计数和日志，正常退出。", flush=True)
            return
        elif limit and completed >= limit:
            return
        else:
            kind = "development"
        if not pending:
            pending = {
                "kind": kind,
                "head_before": git(args.repo, "rev-parse", "HEAD"),
                "branch": git(args.repo, "rev-parse", "--abbrev-ref", "HEAD"),
                "session_id": None,
                "attempts": 0,
                "model": args.model,
                "profile": args.profile,
                "automatic_review": kind == "documentation" and not args.review_docs,
                "started_clean": not git(
                    args.repo, "status", "--porcelain", "--untracked-files=normal"
                ),
            }
            if not args.dry_run:
                directory = Path(
                    tempfile.mkdtemp(
                        prefix=time.strftime("%Y%m%d-%H%M%S-") + kind + "-", dir=path.parent
                    )
                )
                pending["run_dir"] = directory.name
                (directory / "prompt.txt").write_text(PROMPTS[kind] + "\n", encoding="utf-8")
                state["pending"] = pending
                save_state(path, state)
        if args.dry_run:
            print(shlex.join(command(args, codex, pending, path.parent / "PREVIEW")))
        else:
            run_session(args, codex, state, path)
        if kind == "development":
            state["development_runs"] += 1
            completed += 1
        else:
            state["reviewed_runs"] = state["development_runs"]
        state["pending"] = None
        if not args.dry_run:
            save_state(path, state)
            print(
                f"累计开发 {state['development_runs']} 次，最近文档核验覆盖到第 {state['reviewed_runs']} 次。",
                flush=True,
            )
        if kind == "documentation" and (
            args.review_docs or not pending.get("automatic_review", False)
        ):
            return
        if limit and completed >= limit and not doc_review_due(state):
            return
        if not args.dry_run:
            time.sleep(args.interval)


def check_cli(codex):
    help_text = subprocess.check_output([codex, "exec", "--help"], text=True)
    if "--approve-for-me" not in help_text:
        raise RuntimeError("当前 Codex CLI 不支持 --approve-for-me，请更新 CLI 后重试")
    resume_help = subprocess.check_output([codex, "exec", "resume", "--help"], text=True)
    for flag in ("--json", "--output-last-message"):
        if flag not in help_text or flag not in resume_help:
            raise RuntimeError(f"当前 Codex CLI 的 exec/resume 缺少 {flag}")
    auth = subprocess.run([codex, "login", "status"], capture_output=True, text=True)
    if auth.returncode:
        raise RuntimeError("Codex 未登录，请先运行 codex login")
    version = subprocess.check_output([codex, "--version"], text=True).strip()
    return {"version": version, "authenticated": True}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="目标仓库，默认当前脚本所属仓库",
    )
    parser.add_argument(
        "--max-dev-runs",
        type=int,
        default=0,
        help="本次完成的开发次数（含恢复轮）；到期文档轮完成后退出；0 表示持续运行",
    )
    parser.add_argument("--interval", type=float, default=5, help="轮次间隔秒数，默认 5")
    parser.add_argument("--model", help="覆盖模型，默认沿用配置或未完成轮的参数")
    parser.add_argument("--profile", help="指定 Codex profile")
    parser.add_argument(
        "--resume",
        type=lambda value: str(uuid.UUID(value)),
        help="指定要恢复的旧版或缺少 ID 的未完成会话",
    )
    parser.add_argument(
        "--retry-blocked", action="store_true", help="处理阻塞原因后，明确允许恢复同一未完成会话"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览命令，不调用 Codex、不写状态；默认预览 5 次开发",
    )
    modes.add_argument(
        "--check", action="store_true", help="检查状态、锁、CLI 和登录后退出，不调用模型"
    )
    modes.add_argument(
        "--retire-pending", metavar="原因", help="归档已被其他工作替代的未完成轮次后退出"
    )
    parser.add_argument(
        "--review-docs", action="store_true", help="显式核验文档一次后退出，不计开发轮次"
    )
    args = parser.parse_args()
    if args.max_dev_runs < 0 or not math.isfinite(args.interval) or args.interval < 0:
        parser.error("次数和间隔必须为非负有限数值")
    if args.review_docs and args.max_dev_runs:
        parser.error("--review-docs 与 --max-dev-runs 不能同时使用")
    if args.retire_pending is not None and (args.resume or args.review_docs or args.retry_blocked):
        parser.error("归档不能同时恢复会话或核验文档")
    args.repo = Path(git(args.repo.resolve(), "rev-parse", "--show-toplevel")).resolve()
    path = args.repo / "build" / "mla-codex-loop" / "state.json"
    if args.dry_run:
        run_loop(args, "codex", path, prepare_state(args, read_state(path, args.repo), path))
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "loop.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("该仓库已有 MLA Codex 循环在运行") from None
        state = prepare_state(args, read_state(path, args.repo), path)
        if args.retire_pending is not None:
            retire_pending(args, state, path)
            return 0
        validate_start(args.repo, state["pending"])
        for name in ("AGENTS.md", "docs/mla/status.md", "docs/mla/plan.md"):
            if not (args.repo / name).is_file():
                raise RuntimeError(f"缺少开发交接文件：{name}")
        codex = shutil.which("codex")
        if not codex:
            raise RuntimeError("PATH 中找不到 codex；请安装 Codex CLI 并完成 codex login")
        cli = check_cli(codex)
        step = next_step(args.repo)
        if args.check:
            print(
                json.dumps(
                    dict(
                        status=(
                            "blocked"
                            if state["pending"]
                            and blocked_result(path.parent / state["pending"]["run_dir"])
                            else "complete"
                            if not state["pending"] and step == "none" and not doc_review_due(state)
                            else "ready"
                        ),
                        next_step=step,
                        repo=str(args.repo),
                        head=git(args.repo, "rev-parse", "HEAD"),
                        pending=state["pending"],
                        development_runs=state["development_runs"],
                        reviewed_runs=state["reviewed_runs"],
                        automatic_doc_review=True,
                        doc_review_interval=DOC_REVIEW_INTERVAL,
                        doc_review_due=doc_review_due(state),
                        working_tree_clean=not git(
                            args.repo, "status", "--porcelain", "--untracked-files=normal"
                        ),
                        cli=cli,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        save_state(path, state)
        # 父脚本被 SIGKILL 时，仍存活的 CLI 保持锁，防止同时恢复同一会话。
        args.lock_fd = lock.fileno()
        run_loop(args, codex, path, state)
    return 0


if __name__ == "__main__":
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    try:
        sys.exit(main())
    except Stopped as exc:
        print("\n已停止并保留未完成轮次；重启脚本会恢复同一个会话。", file=sys.stderr)
        sys.exit(128 + exc.signum)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
