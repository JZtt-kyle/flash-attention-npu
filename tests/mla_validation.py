"""Reproducible MLA checks: python -m tests.mla_validation --help.

Daily suites select at most 100 deterministic cases; --full-matrix opts into all cases.
Builds, golden tensors and successful test evidence have separate identities.
Performance remains in tests.mla_paged_profile and must always be measured.
"""

import argparse
from collections import Counter
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from tests.mla_dense_build import ROOT, sha256
from tests.mla_validation_build import matching_files
from tests.mla_validation_dependencies import suite_sources
from tests.mla_validation_selection import INCREMENTAL_SUITES
from tests.mla_validation_policy import DEFAULT_MAX_CASES, POLICY_VERSION


PUBLIC = "tests/test_flash_attn_npu_v3_mla_paged_tnd_public.py"
CAUSAL_CPU = "tests/test_flash_attn_npu_v3_mla_causal_schedule.py"
USED_SHARED_CPU = [
    f"tests/test_flash_attn_npu_v3_mla_{name}.py"
    for name in (
        "schedule",
        "bsnd_schedule",
        "tnd_schedule",
        "tnd_bsnd_schedule",
        "used_schedule",
        "paged_schedule",
        "used_metadata",
    )
]
SUITES = {
    "used-shared-cpu": USED_SHARED_CPU,
    "used-shared-sanitize": USED_SHARED_CPU,
    "causal-bsnd-cpu": [
        "tests/test_flash_attn_npu_v3_mla_causal_metadata.py",
        "tests/test_flash_attn_npu_v3_mla_used_metadata_consumer.py",
    ],
    "causal-bsnd-sanitize": [
        "tests/test_flash_attn_npu_v3_mla_causal_metadata.py",
        "tests/test_flash_attn_npu_v3_mla_used_metadata_consumer.py",
    ],
    "causal-bsnd-public": [
        "tests/test_flash_attn_npu_v3_mla_causal_public.py",
        "-k",
        "not release_trap",
    ],
    "causal-bsnd-check": [
        "tests/test_flash_attn_npu_v3_mla_causal_public.py",
        "-k",
        "representative",
    ],
    "causal-bsnd-trap": ["tests/test_flash_attn_npu_v3_mla_causal_public.py", "-k", "release_trap"],
    "causal-bsnd-profile": ["tests/test_mla_causal_bsnd_profile.py", "-k", "not native_preflight"],
    "causal-bsnd-native-check": ["tests/test_mla_causal_bsnd_profile.py", "-k", "native_preflight"],
    "causal-bsnd-legacy": [
        "tests/test_flash_attn_npu_v3_mla_bsnd_used.py",
        "tests/test_flash_attn_npu_v3_mla_kvcache.py",
        "-k",
        "descriptor_rejections or tensor_rejections or explicit_metadata_rejected or word_overflow or fullgraph or cross_stream_allocator_lifetime or metadata_rejects_unsupported or metadata_length_tensor_boundary",
    ],
    "public": [PUBLIC, "-k", "not public_release_trap"],
    "trap": [PUBLIC, "-k", "public_release_trap"],
    "device": [
        "tests/test_flash_attn_npu_v3_mla_paged_tnd_production.py",
        "-k",
        "not release_trap",
    ],
    "bsnd": ["tests/test_flash_attn_npu_v3_mla_paged_public.py", "-k", "not public_release_trap"],
    "metadata": ["tests/test_flash_attn_npu_v3_mla_paged_metadata_public.py"],
    "ordinary": [
        "tests/test_flash_attn_npu_v3_mla_contract.py",
        "tests/test_flash_attn_npu_v3_metadata_signature.py",
        "tests/test_flash_attn_npu_v3_mla_metadata_dispatch.py",
        "tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_func_metadata_bsnd",
        "tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_varlen_func_metadata_tnd[data_type0-1-1-1-512-1024-128-True]",
        "tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_kvcache_metadata_bsnd[data_type0-1-1-1-1024-1024-128-128-False]",
        "tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_kvcache_metadata_flash_decode[data_type1-8-1-64-2-False-0.0-0]",
        "tests/test_torch_compile_fa3.py::test_fa3_910_scheduler_metadata_torch_compile_correctness",
        "tests/test_torch_compile_fa3.py::test_fa3_910_varlen_torch_compile_correctness",
        "tests/test_torch_compile_fa3.py::test_fa3_910_fixed_torch_compile_correctness",
    ],
    "nonpaged": [
        "tests/test_flash_attn_npu_v3_mla_mixed_metadata_public.py",
        "tests/test_flash_attn_npu_v3_mla_kvcache.py",
        "-k",
        "not public_source_accuracy and not public_rounding and not reuse_scales_streams_and_allocator and not shared_metadata_host_threads and not public_runtime_error",
        *[
            f"tests/test_flash_attn_npu_v3_mla_tnd_paged_attention.py::{name}"
            for name in (
                "test_strides",
                "test_empty_cache",
                "test_stale_lengths_and_regeneration",
                "test_corruption_precedes_attention",
            )
        ],
    ],
    "cpu": [
        "tests/test_flash_attn_npu_v3_mla_build.py::test_generated_mla_launch",
        "tests/test_flash_attn_npu_v3_mla_build.py::test_device_dependency_staleness",
    ],
    "tools": [
        "tests/test_mla_validation.py",
        "tests/test_mla_source_compare.py",
        "tests/test_mla_codex_loop.py",
        "tests/test_mla_reference_cache.py",
        "tests/test_mla_validation_dependencies.py",
        "tests/test_mla_paged_reference_cache.py",
        "tests/test_mla_native_dispatch.py",
        "tests/test_mla_validation_policy.py",
        "tests/common/test_golden_cache.py",
    ],
    "reference": ["tests/test_flash_attn_npu_v3_mla_reference.py"],
    "causal-smoke": [
        "tests/test_flash_attn_npu_v3_mla_causal_attention.py",
        "-k",
        "cost_and_mask_smoke or amla_and_masked_max_changes",
    ],
    "causal-attention": ["tests/test_flash_attn_npu_v3_mla_causal_attention.py"],
    "causal-profile": ["tests/test_mla_causal_profile.py"],
    "shared-profile": ["tests/test_mla_shared_profile.py"],
    "shared-dense": [
        f"tests/test_flash_attn_npu_v3_mla_{name}.py"
        for name in ("dense", "multiblock", "kvblocks", "longkv")
    ],
    "shared-nonpaged": [
        *[
            f"tests/test_flash_attn_npu_v3_mla_{name}.py"
            for name in (
                "bsnd_attention",
                "tnd_attention",
                "tnd_bsnd_attention",
                "used_attention",
                "bsnd_stride",
                "tnd_stride",
            )
        ],
        "-k",
        "not runtime and not trap",
    ],
    "shared-paged": [
        *[
            f"tests/test_flash_attn_npu_v3_mla_{name}.py"
            for name in ("paged_attention", "tnd_paged_attention")
        ],
        "-k",
        "task_ring or strides or empty or zero or scale_halves or reuse_with_new or stream_and_allocator",
    ],
    "causal-cpu": [CAUSAL_CPU],
    "causal-sanitize": [CAUSAL_CPU],
}
SUITES.update(INCREMENTAL_SUITES)
PERFORMANCE_SUITES = {
    "causal-profile",
    "shared-profile",
    "causal-bsnd-profile",
    "causal-bsnd-native-check",
}
CAUSAL_DEVICE_SUITES = {
    "causal-smoke",
    "causal-attention",
    "causal-profile",
    "causal-check",
    "causal-bsnd-public",
    "causal-bsnd-check",
    "causal-bsnd-profile",
}
CPU_SUITES = {
    "used-shared-cpu",
    "used-shared-sanitize",
    "cpu",
    "tools",
    "reference",
    "causal-cpu",
    "causal-sanitize",
    "causal-bsnd-cpu",
    "causal-bsnd-sanitize",
}


def suite_dependencies(suite):
    if suite not in {
        "causal-cpu",
        "causal-sanitize",
        "causal-bsnd-cpu",
        "causal-bsnd-sanitize",
        "used-shared-cpu",
        "used-shared-sanitize",
    }:
        return {}
    from tests.mla_causal_build import toolchain_identity

    return toolchain_identity(suite.endswith("sanitize"))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sources():
    # Conservative closure, deliberately excluding docs and evidence directories.
    # Shared helpers, setup flags, fixtures and untracked source changes count.
    paths = [ROOT / "setup.py", ROOT / "pyproject.toml", ROOT / "scripts/mla_codex_loop.py"]
    for directory in ("tests", "csrc/ascend910", "csrc/catlass/include", "flash_attn_npu_3"):
        paths.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.suffix in {".py", ".cpp", ".h", ".hpp", ".aicpu", ".asc"}
        )
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another validation process holds {path}") from exc
        yield stream.fileno()


def execute(command, env, directory, lock_fds=()):
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = dict(command=command, cwd=str(ROOT), status="running", pid=os.getpid())
    record["environment"] = {
        key: value
        for key, value in env.items()
        if key.startswith(("MLA_", "GOLDEN_CACHE_", "FLASH_ATTN_"))
        or key
        in {
            "ASCEND_HOME_PATH",
            "ASCEND_TOOLKIT_HOME",
            "ASCEND_OPP_PATH",
            "ASCEND_AICPU_PATH",
            "ASCEND_RT_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "TORCH_DEVICE_BACKEND_AUTOLOAD",
            "PYTEST_ADDOPTS",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "CI_TORCH_SEED",
        }
    }
    save(directory / "command.json", record)
    with (directory / "output.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=lock_fds,
        )
        record["child_pid"] = process.pid
        save(directory / "command.json", record)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 130
    record.update(
        returncode=code,
        seconds=time.monotonic() - started,
        status="passed" if code == 0 else "interrupted" if code == 130 else "failed",
    )
    save(directory / "command.json", record)
    return record


def pytest_command(suite, directory, collect=False):
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-x",
        "-q",
        *SUITES[suite],
        "-p",
        "tests.mla_validation_plugin",
        "-o",
        "addopts=",
        "--random-sample=0",
    ]
    if collect:
        return command + ["--collect-only"]
    return command + [f"--basetemp={directory / 'tmp'}", f"--junitxml={directory / 'results.xml'}"]


def environment(args, directory, suite):
    env = dict(os.environ)
    # Inherited pytest/plugin or old step-directory settings must not silently
    # change coverage or redirect fixtures to unrelated artifacts.
    for key in list(env):
        if key.startswith(("MLA_", "GOLDEN_CACHE_", "PYTEST_")) and key != "MLA_CANN_SOURCE_ROOT":
            env.pop(key)
    env.update(
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        ASCEND_RT_VISIBLE_DEVICES=args.device,
        FLASH_ATTN_BUILD_VERSION="v3",
        FLASH_ATTN_BUILD_NPU="910",
        FLASH_ATTN_SKIP_SUBMODULE_INIT="1",
        PYTEST_ADDOPTS="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        MLA_VALIDATION_RESULT=str(directory),
        MLA_VALIDATION_NPU="0" if suite in CPU_SUITES else "1",
        GOLDEN_CACHE_MODE="off" if args.golden == "off" else "cache",
        GOLDEN_CACHE_REFRESH="1" if args.golden == "refresh" else "0",
        GOLDEN_CACHE_DIR=str(args.artifacts / "golden"),
        GOLDEN_CACHE_STATS_FILE=str(directory / "golden.tsv"),
        MLA_PAGED_ATTENTION_DIR=str(directory / "probes"),
        MLA_PAGED_ATTENTION_REUSE="1",
        MLA_PAGED_METADATA_LIBRARY=str(args.artifacts / "producer/metadata.so"),
    )
    for variable, child in {
        "MLA_PAGED_METADATA_DEVICE_DIR": "producer",
        "MLA_PAGED_SCHEDULE_DIR": "source",
        "MLA_PAGED_PUBLIC_DIR": "public",
        "MLA_PAGED_CONSUMER_CPU_DIR": "consumer",
        "MLA_PAGED_PRODUCTION_DIR": "production",
        "MLA_TND_PAGED_KVCACHE_DIR": "kvcache",
    }.items():
        env[variable] = str(directory / child)
    env["MLA_VALIDATION_MAX_CASES"] = str(getattr(args, "max_cases", DEFAULT_MAX_CASES))
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0" if suite in CPU_SUITES else "1"
    if suite in CAUSAL_DEVICE_SUITES:
        env["MLA_CAUSAL_ATTENTION_DIR"] = str(directory / "probes")
    if suite in {
        "causal-cpu",
        "causal-sanitize",
        "causal-bsnd-cpu",
        "causal-bsnd-sanitize",
        "used-shared-cpu",
        "used-shared-sanitize",
    }:
        env["MLA_SCHEDULE_ARTIFACT_DIR"] = str(directory / "schedule")
        env["MLA_SCHEDULE_SANITIZE"] = "1" if suite.endswith("sanitize") else "0"
        if suite == "causal-bsnd-sanitize":
            env["MLA_CONSUMER_SANITIZE"] = "1"
    return env


def validate_result(directory, returncode, runtime_info=None):
    cases = json.loads((directory / "cases.json").read_text())
    root = ET.parse(directory / "results.xml").getroot()
    tests = list(root.iter("testcase"))
    failed = sum(any(t.find(kind) is not None for kind in ("failure", "error")) for t in tests)
    skipped = sum(t.find("skipped") is not None for t in tests)
    passed = len(tests) - failed - skipped
    if runtime_info:
        loaded = json.loads((directory / "loaded.json").read_text())
        if loaded != dict(path=runtime_info["extension"], sha256=runtime_info["extension_sha256"]):
            raise RuntimeError("pytest loaded a different extension")
    # Exit zero, xfail/skip or an empty selection are not acceptance evidence.
    ok = (
        returncode == 0
        and bool(cases)
        and len(cases) == len(set(cases)) == len(tests)
        and not failed
        and not skipped
    )
    return dict(
        status="passed" if ok else "failed",
        passed=passed,
        failed=failed,
        skipped=skipped,
        collected=len(cases),
        cases_sha256=digest(cases),
    )


def validate_causal_builds(directory):
    """Check both fresh probes and their actual loaded identities after execution."""
    for dtype in ("fp16", "bf16"):
        base = directory / "probes" / dtype
        manifest = json.loads((base / "build-manifest.json").read_text())
        if not manifest.get("causal") or manifest.get("dtype") != dtype:
            raise RuntimeError("Expected a causal probe build for " + dtype)
        sources = {str(ROOT / p): h for p, h in manifest["sources"].items()}
        libraries = {r["path"]: r["sha256"] for r in manifest["libraries"].values()}
        loaded = json.loads((base / "loaded.json").read_text())
        if (
            not manifest["dependencies"]
            or not libraries
            or loaded != manifest["libraries"]
            or not all(matching_files(x) for x in (sources, manifest["dependencies"], libraries))
        ):
            raise RuntimeError(
                "Causal probe sources, dependencies, outputs or loaded identities changed"
            )


def reusable(record, identity):
    if not isinstance(record, dict):
        return False
    if record.get("status") != "passed" or record.get("identity") != identity:
        return False
    try:
        evidence = record["evidence"]
        return bool(evidence) and all(
            sha256(Path(path)) == value for path, value in evidence.items()
        )
    except (KeyError, OSError):
        return False


def cache_stats(path):
    if not path.exists():
        return {}
    return dict(Counter(line.split("\t")[0] for line in path.read_text().splitlines()))


def run(args):
    args.max_cases = getattr(args, "max_cases", DEFAULT_MAX_CASES)
    if not 0 <= args.max_cases <= DEFAULT_MAX_CASES:
        raise ValueError("--max-cases must be 1..100; use --full-matrix for all cases")
    args.output = args.output.resolve()
    args.artifacts = args.artifacts.resolve()
    args.suite = list(dict.fromkeys(args.suite))
    if args.resume and args.golden == "refresh":
        raise ValueError("--resume cannot be combined with --golden refresh")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    max_cases_per_suite=args.max_cases,
                    selection_policy=POLICY_VERSION,
                    prepare=[
                        sys.executable,
                        "-m",
                        "tests.mla_validation_build",
                        "prepare",
                        str(args.artifacts),
                    ]
                    if set(args.suite) - CPU_SUITES
                    and not args.collect_only
                    and not set(args.suite) <= {"causal-smoke", "causal-check"}
                    else [],
                    suites={
                        name: pytest_command(
                            name, args.output / "attempt-N" / name, args.collect_only
                        )
                        for name in args.suite
                    },
                    golden=args.golden,
                    device=args.device,
                    artifacts=str(args.artifacts),
                    resume=args.resume,
                ),
                indent=2,
            )
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        lock_fds = [stack.enter_context(locked(args.output / ".lock"))]
        needs_npu = bool(set(args.suite) - CPU_SUITES) and not args.collect_only
        # The representative component probe builds itself from source. It
        # does not call the production extension or PA producer. Full device
        # acceptance still prepares production and checks its actual runtime.
        needs_build = needs_npu and not set(args.suite) <= {"causal-smoke", "causal-check"}
        if needs_npu:
            lock_fds += [
                stack.enter_context(
                    locked(Path("/tmp") / f"flash-attn-mla-device-{args.device}.lock")
                ),
                stack.enter_context(locked(args.artifacts / ".lock")),
            ]
        attempt = args.output / f"attempt-{time.time_ns()}"
        attempt.mkdir()
        state_path = args.output / "state.json"
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if not isinstance(state, dict):
                raise ValueError("Expected a group index")
        except ValueError:
            # Preserve the damaged index for diagnosis; it cannot authorize a
            # skipped test. Existing attempt directories remain untouched.
            shutil.copyfile(state_path, attempt / "invalid-state.json")
            state = {}
        summary = dict(
            max_cases_per_suite=args.max_cases,
            selection_policy=POLICY_VERSION,
            attempt=str(attempt),
            groups={name: dict(status="not_run") for name in args.suite},
        )
        summary_path = args.output / "summary.json"
        save(summary_path, summary)
        save(
            attempt / "baseline.json",
            dict(
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            ),
        )
        runtime_info = None
        baseline = sources()
        save(attempt / "sources.json", baseline)
        if needs_npu:
            env = environment(args, attempt, "public")
            # Do not initialize the runtime in the compiler subprocess.
            build_env = dict(env, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
            for name, command, step_env in (
                (
                    "build",
                    [
                        sys.executable,
                        "-m",
                        "tests.mla_validation_build",
                        "prepare",
                        str(args.artifacts),
                    ],
                    build_env,
                ),
                (
                    "runtime",
                    [
                        sys.executable,
                        "-m",
                        "tests.mla_validation_build",
                        "runtime",
                        str(attempt / "runtime.json"),
                    ],
                    env,
                ),
            ):
                if name == "build" and not needs_build:
                    continue
                print(f"{name}: running; log {attempt / name / 'output.log'}", flush=True)
                result = execute(command, step_env, attempt / name, lock_fds)
                if sources() != baseline:
                    result.update(
                        status="failed", returncode=1, error="Source changed during preparation"
                    )
                summary[name] = result
                save(summary_path, summary)
                print(f"{name}: {result['status']} ({result['seconds']:.1f}s)", flush=True)
                if result["returncode"]:
                    return result["returncode"]
            runtime_info = json.loads((attempt / "runtime.json").read_text())
            driver_info = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
            (attempt / "npu-smi.log").write_text(driver_info)
            runtime_info["driver_header"] = next(
                line.strip() for line in driver_info.splitlines() if "npu-smi" in line
            )
            runtime_info["host_kernel"] = os.uname().release
            if needs_build:
                shutil.copyfile(args.artifacts / "prepare.json", attempt / "build-manifest.json")
            else:
                summary["build"] = dict(
                    status="not_required",
                    scope="Fresh independent causal smoke probes; no production execution",
                )
            summary["runtime_identity"] = runtime_info
        packages = {
            n: importlib.metadata.version(n) for n in ("pytest", "torch", "torch_npu", "numpy")
        }
        for suite in args.suite:
            directory = attempt / suite
            directory.mkdir()
            env = environment(args, directory, suite)
            dependencies = suite_dependencies(suite)
            save(directory / "suite-dependencies.json", dependencies)
            group_sources = suite_sources(ROOT, SUITES[suite], baseline, headless=suite == "tools")
            save(directory / "sources.json", group_sources)
            # Omit only output locations from the identity. Runtime and all
            # inherited environment values participate without exposing secrets.
            semantic_env = {
                k: v for k, v in env.items() if not k.startswith(("MLA_", "GOLDEN_CACHE_"))
            }
            identity = digest(
                dict(
                    source_schema=2,
                    sources=group_sources,
                    packages=packages,
                    runtime=runtime_info if suite not in CPU_SUITES else sys.version,
                    environment=semantic_env,
                    selection=SUITES[suite],
                    case_limit=args.max_cases,
                    selection_policy=POLICY_VERSION,
                    dependencies=dependencies,
                    golden=args.golden,
                    build=sha256(args.artifacts / "prepare.json") if needs_build else None,
                )
            )
            if (
                args.resume
                and not args.collect_only
                and suite not in PERFORMANCE_SUITES
                and reusable(state.get(suite, {}), identity)
            ):
                summary["groups"][suite] = dict(
                    status="inherited",
                    from_directory=state[suite]["directory"],
                    passed=state[suite]["passed"],
                    seconds=0,
                    identity=identity,
                )
                save(summary_path, summary)
                print(f"{suite}: inherited {state[suite]['passed']} cases", flush=True)
                continue
            # Invalidate before launching, so interruption cannot resurrect an
            # older successful run after a newer failure.
            if not args.collect_only:
                state.pop(suite, None)
                save(state_path, state)
            command = pytest_command(suite, directory, args.collect_only)
            if needs_build and suite not in CPU_SUITES:
                # Probe libraries are shared; numerical outputs and loaded
                # manifests belong to this attempt, not the mutable cache.
                for relative in ("probes/fp16", "probes/bf16", "producer"):
                    target = directory / relative
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(
                        args.artifacts / relative / "build-manifest.json",
                        target / "build-manifest.json",
                    )
            print(f"{suite}: running; log {directory / 'output.log'}", flush=True)
            result = execute(command, env, directory, lock_fds)
            result.update(
                directory=str(directory),
                identity=identity,
                golden=cache_stats(directory / "golden.tsv"),
            )
            if not args.collect_only:
                try:
                    result.update(
                        validate_result(
                            directory,
                            result["returncode"],
                            runtime_info if suite not in CPU_SUITES else None,
                        )
                    )
                except (OSError, ValueError, ET.ParseError, RuntimeError) as exc:
                    result.update(status="failed", error=str(exc))
            elif result["returncode"] == 0:
                cases = json.loads((directory / "cases.json").read_text())
                result.update(status="collected" if cases else "failed", collected=len(cases))
            if result["returncode"] == 130:
                result["status"] = "interrupted"
            if (
                suite_sources(ROOT, SUITES[suite], sources(), headless=suite == "tools")
                != group_sources
            ):
                result.update(status="failed", error="Source changed during validation")
            if suite_dependencies(suite) != dependencies:
                result.update(
                    status="failed", error="CPU toolchain or fixed source changed during validation"
                )
            if runtime_info and not matching_files(runtime_info["loaded_libraries"]):
                result.update(status="failed", error="Loaded libraries changed during validation")
            if needs_build:
                prepared = json.loads((args.artifacts / "prepare.json").read_text())
                if not matching_files(prepared["outputs"]) or not matching_files(
                    prepared["dependencies"]
                ):
                    result.update(
                        status="failed",
                        error="Build artifacts or dependencies changed during validation",
                    )
            if (
                suite in CAUSAL_DEVICE_SUITES
                and not args.collect_only
                and result["returncode"] == 0
            ):
                try:
                    validate_causal_builds(directory)
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    result.update(status="failed", error=str(exc))
            summary["groups"][suite] = result
            if result["status"] == "passed":
                result["evidence"] = {
                    str(p): sha256(p)
                    for p in directory.rglob("*")
                    if p.is_file() and "tmp" not in p.relative_to(directory).parts
                }
                state[suite] = result
                save(state_path, state)
            save(summary_path, summary)
            save(attempt / "summary.json", summary)
            print(f"{suite}: {result['status']} ({result['seconds']:.1f}s)", flush=True)
            if result["status"] in {"failed", "interrupted"}:
                return result["returncode"] or 1
        save(attempt / "summary.json", summary)
    return 0


def case_limit(value):
    count = int(value)
    if not 1 <= count <= DEFAULT_MAX_CASES:
        raise argparse.ArgumentTypeError("expected 1..100; use --full-matrix for all cases")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Evidence directory; attempts are never overwritten",
    )
    parser.add_argument("--suite", nargs="+", choices=SUITES, required=True)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument(
        "--max-cases",
        type=case_limit,
        default=DEFAULT_MAX_CASES,
        help="Maximum deterministic pytest cases per suite (default: 100)",
    )
    limits.add_argument(
        "--full-matrix",
        dest="max_cases",
        action="store_const",
        const=0,
        help="Explicitly run all correctness cases; performance keeps its fixed cap",
    )
    parser.add_argument("--artifacts", type=Path, default=ROOT / "build/mla-validation-artifacts")
    parser.add_argument("--device", default="6", choices=[str(i) for i in range(8)])
    parser.add_argument("--golden", choices=("off", "cache", "refresh"), default="cache")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without building, importing torch or writing files",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Record exact nodeids without building or executing tests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Inherit successful groups only when identities and evidence match",
    )
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
