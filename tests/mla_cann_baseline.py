"""Dense MLA CANN accuracy and profiling, independent of FA3 kernels.

Run from the repository root with ``python -m tests.mla_cann_baseline --help``.
The D/P/FD01 cases and mappings follow docs/mla/design.md. No CANN kernel
code is copied. Only BSND, no mask, native automatic splitting, and O/LSE are
covered here. This is a test adapter, never a production fallback.

NPU event spans include dispatch gaps and possible adapter copies; they are
NOT isolated kernel timings. Installed-package measurements do not establish
equivalence to the pinned CANN source. Optional --profile-dir records isolated
kernel durations and Slice conversion costs in a separate profiler capture.
Native build and actual device dispatch evidence must accompany comparisons.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from tests.common.compare import assert_fa_close
from tests.common.mla_ref import fa3_to_cann_inputs, mla_reference_pair
from tests.mla_cann_profile import profile_calls


PINNED_CANN = "ca18aee610729ca5270a44e2d84c751bc687fbf0"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class Case:
    name: str
    representation: str
    batch: int
    seqlen_q: int
    seqlen_k: int
    heads_q: int
    heads_k: int
    split_rope: bool = True
    scale: float | None = None


CASES = (
    Case("D01-default", "compressed", 2, 1, 257, 8, 1),
    Case("D01-scale", "compressed", 2, 1, 257, 8, 1, scale=0.23),
    Case("D01-zero", "compressed", 2, 1, 257, 8, 1, scale=0.0),
    Case("D02", "compressed", 1, 7, 65, 16, 1),
    Case("P01-split", "prefill", 2, 17, 31, 8, 2),
    Case("P01-combined", "prefill", 2, 17, 31, 8, 2, split_rope=False),
    Case("P02-mha-split", "prefill", 1, 33, 129, 8, 8),
    Case("P02-mha-combined", "prefill", 1, 33, 129, 8, 8, split_rope=False),
    Case("P02-mqa-split", "prefill", 1, 33, 129, 8, 1),
    Case("P02-mqa-combined", "prefill", 1, 33, 129, 8, 1, split_rope=False),
    # Dense, unmasked FD01 inputs use CANN's native automatic split policy.
    Case("FD01-D-q1", "compressed", 1, 1, 8193, 16, 1),
    Case("FD01-D-q7", "compressed", 1, 7, 8193, 16, 1),
    Case("FD01-P-q1-split", "prefill", 1, 1, 8193, 8, 1),
    Case("FD01-P-q1-combined", "prefill", 1, 1, 8193, 8, 1, split_rope=False),
    Case("FD01-P-q7-split", "prefill", 1, 7, 8193, 8, 1),
    Case("FD01-P-q7-combined", "prefill", 1, 7, 8193, 8, 1, split_rope=False),
    Case("P-long-split", "prefill", 1, 257, 1025, 8, 2),
    Case("P-long-combined", "prefill", 1, 257, 1025, 8, 2, split_rope=False),
)


def make_inputs(case, dtype, seed=20260924):
    """Generate on CPU so repeated runs and split/combined use identical data."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    compressed = case.representation == "compressed"
    if case.representation not in ("compressed", "prefill"):
        raise ValueError(f"Unknown representation: {case.representation}")
    dq, dv = (64, 512) if compressed else (192, 128)

    def rand(sequence, heads, dim):
        return torch.randn(case.batch, sequence, heads, dim, generator=generator).to(dtype)

    return dict(
        q=rand(case.seqlen_q, case.heads_q, dq),
        k=rand(case.seqlen_k, case.heads_k, dq),
        v=rand(case.seqlen_k, case.heads_k, dv),
        qv=rand(case.seqlen_q, case.heads_q, dv) if compressed else None,
    )


def require_910b(device):
    import torch_npu  # noqa: F401 -- register NPU operators before execution

    torch.npu.set_device(device)
    name = torch.npu.get_device_name(device)
    if "910B" not in name.upper():
        raise RuntimeError(f"This baseline requires a 910B device, got {name}")
    return name


class CannCall:
    """Prepared logical inputs; allocate outputs on every invocation.

    Explicit output shapes work around torch_npu 2.7.1's default allocation
    of Dq rather than Dv for combined 192/128 prefill. Split prefill keeps its
    192-wide input row stride. Any copies by torch_npu/CANN occur inside the
    timed call. H2D input transfers and one-time creation of views are excluded.
    """

    def __init__(self, inputs, *, scale=None, split_rope=True, kv_lengths=None, causal=False):
        q, k, v = (inputs[name] for name in ("q", "k", "v"))
        if q.device.type != "npu" or q.dtype not in DTYPES.values():
            raise ValueError("CANN baseline requires FP16/BF16 NPU inputs")
        if min(q.shape[:3]) <= 0 or k.shape[1] <= 0:
            raise ValueError("Empty sequences are outside this dense baseline")
        mapped = fa3_to_cann_inputs(**inputs, softmax_scale=scale, split_rope=split_rope)
        self.query, self.key, self.value = (mapped.pop(name) for name in ("query", "key", "value"))
        self.kwargs = dict(
            **{name: mapped[name] for name in ("query_rope", "key_rope")},
            softmax_scale=mapped["scale"],
            num_query_heads=q.shape[2],
            num_key_value_heads=k.shape[2],
            input_layout="BSND",
            sparse_mode=0,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            return_softmax_lse=True,
        )
        if causal:
            self.kwargs.update(
                sparse_mode=3,
                atten_mask=torch.ones((2048, 2048), dtype=torch.bool, device=q.device).triu(1),
            )
        self.out_shape = (*q.shape[:3], v.shape[-1])
        self.lse_shape = (q.shape[0], q.shape[2], q.shape[1], 1)
        self.operator = torch.ops.npu.npu_fused_infer_attention_score_v2.out
        if kv_lengths is not None:
            if len(kv_lengths) != q.shape[0] or any(not 0 <= x <= k.shape[1] for x in kv_lengths):
                raise ValueError("CANN BSND lengths must fit each allocated cache")
            # The pinned public torch_npu API takes a host integer list. Its
            # conversion/submission remains inside each timed operator call.
            self.kwargs["actual_seq_kvlen"] = list(kv_lengths)

    def __call__(self):
        out = torch.empty(self.out_shape, device=self.query.device, dtype=self.query.dtype)
        lse = torch.empty(self.lse_shape, device=self.query.device, dtype=torch.float32)
        out, lse = self.operator(self.query, self.key, self.value, **self.kwargs, out=[out, lse])
        if tuple(out.shape) != self.out_shape or tuple(lse.shape) != self.lse_shape:
            raise AssertionError(f"Unexpected CANN output shapes: {out.shape}, {lse.shape}")
        if out.dtype != self.query.dtype or lse.dtype != torch.float32:
            raise AssertionError("CANN must return input-dtype O and FP32 LSE")
        return out, lse.squeeze(-1)

    def description(self):
        tensors = dict(query=self.query, key=self.key, value=self.value)
        tensors.update({name: self.kwargs[name] for name in ("query_rope", "key_rope")})
        return {
            "tensors": {
                name: None
                if tensor is None
                else dict(
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                    storage_offset=tensor.storage_offset(),
                    dtype=str(tensor.dtype),
                )
                for name, tensor in tensors.items()
            },
            "content_key_is_value": self.key.is_set_to(self.value),
            "scale": self.kwargs["softmax_scale"],
            "layout": "BSND",
            "mask": "right-down causal" if self.kwargs["sparse_mode"] == 3 else "none",
            "sparse_mode": self.kwargs["sparse_mode"],
            "split": "CANN native automatic policy",
            "metadata": "not exposed by FIA",
            "actual_seq_kvlen": self.kwargs.get("actual_seq_kvlen"),
            "output_allocation": "every call via explicit out overload",
        }


def check_case(case, dtype, device, seed=20260924):
    cpu = make_inputs(case, dtype, seed)
    refs = mla_reference_pair(**cpu, softmax_scale=case.scale)
    npu = {name: None if tensor is None else tensor.to(device) for name, tensor in cpu.items()}
    call = CannCall(npu, scale=case.scale, split_rope=case.split_rope)
    out, lse = call()
    # Copying to CPU waits for the computation; never time either reference.
    results = {}
    for name, actual, ref, pt in (
        ("out", out.cpu(), refs[0], refs[2]),
        ("lse", lse.cpu(), refs[1], refs[3]),
    ):
        assert_fa_close(actual, ref, pt, name=name)
        ulp = (ref + 0.3 - 0.3 - ref).abs().max().item()
        pt_error = (pt - ref).abs().max().item()
        results[name] = dict(
            max_error=(actual - ref).abs().max().item(),
            pytorch_max_error=pt_error,
            tolerance=max(2 * pt_error + 2 * ulp, 1e-5),
        )
    return call, results


def measure(call, *, warmup, iterations, rounds):
    """Record event-span means and individual synchronized host call latencies."""
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    samples = []
    for _ in range(rounds):
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        event_total_us = start.elapsed_time(end) * 1000
        wall_us = []
        for _ in range(iterations):
            begin = time.perf_counter_ns()
            call()
            torch.npu.synchronize()
            wall_us.append((time.perf_counter_ns() - begin) / 1000)
        samples.append(
            dict(
                event_total_us=event_total_us,
                event_call_us=event_total_us / iterations,
                synchronized_wall_us=wall_us,
            )
        )
    return dict(
        samples=samples,
        median_event_call_us=statistics.median(row["event_call_us"] for row in samples),
        median_synchronized_wall_us=statistics.median(
            value for row in samples for value in row["synchronized_wall_us"]
        ),
    )


def file_info(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def git_output(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def check_vendor_configuration(vendor):
    """Keep a requested custom-source run from silently using the installed OPP."""
    vendor = Path(vendor).resolve()
    configured = [
        path for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if path
    ]
    if len(configured) != 1 or Path(configured[0]).resolve() != vendor:
        raise RuntimeError("ASCEND_CUSTOM_OPP_PATH must contain exactly the requested CANN vendor")
    if not vendor.is_dir():
        raise RuntimeError(f"CANN vendor directory does not exist: {vendor}")
    # A JIT-only custom package can load new tiling with an installed, incompatible
    # device binary. Require both dense floating-point binaries before any NPU call.
    vendor_kernel_files(vendor)
    return vendor


def vendor_kernel_files(vendor):
    kernel_root = Path(vendor) / "op_impl/ai_core/tbe/kernel"
    if not kernel_root.resolve().is_relative_to(Path(vendor).resolve()):
        raise RuntimeError("Custom FIA kernel directory escapes the vendor directory")
    configs = sorted(kernel_root.glob("config/ascend910b/**/fused_infer_attention_score.json"))
    if len(configs) != 1:
        raise RuntimeError("Custom CANN requires a unique compiled 910B FIA kernel configuration")
    bins = json.loads(configs[0].read_text())["binList"]
    artifacts = [file_info(configs[0])]
    for dtype in ("float16", "bfloat16"):
        matching = [
            item
            for item in bins
            if item["inputs"][0]["dtype"] == dtype and item["outputs"][0]["dtype"] == dtype
        ]
        if len(matching) != 1:
            raise RuntimeError(f"Custom CANN requires one dense {dtype} FIA binary")
        metadata = (kernel_root / matching[0]["binInfo"]["jsonFilePath"]).resolve()
        if not metadata.is_relative_to(kernel_root.resolve()):
            raise RuntimeError("Custom FIA kernel metadata escapes the vendor directory")
        info = json.loads(metadata.read_text())
        binary = (metadata.parent / (info["binFileName"] + info["binFileSuffix"])).resolve()
        if not binary.is_relative_to(kernel_root.resolve()) or not binary.is_file():
            raise RuntimeError(f"Custom CANN device binary is missing or outside vendor: {dtype}")
        binary_info = file_info(binary)
        if info.get("sha256") != binary_info["sha256"] or not info.get("kernelList"):
            raise RuntimeError(f"Custom CANN device binary does not match its metadata: {dtype}")
        artifacts.extend((file_info(metadata), binary_info))
    return artifacts


def verify_loaded_vendor(vendor, source, loaded_paths):
    """Check package loading and source bytes, not device dispatch or timing.

    Native build logs and device dispatch evidence are still needed to connect these
    loaded host libraries and packaged headers to an executed device binary.
    """
    vendor = check_vendor_configuration(vendor)
    if git_output(source, "rev-parse", "HEAD") != PINNED_CANN:
        raise RuntimeError("Custom vendor verification requires the pinned CANN checkout")
    libraries = []
    for name in ("libcust_opapi.so", "libcust_opmaster_rt2.0.so"):
        candidates = sorted({path.resolve() for path in vendor.rglob(name) if path.is_file()})
        if (
            len(candidates) != 1
            or not candidates[0].is_relative_to(vendor)
            or str(candidates[0]) not in loaded_paths
        ):
            raise RuntimeError(f"Requested custom CANN library was not uniquely loaded: {name}")
        libraries.append(file_info(candidates[0]))
    headers = []
    for name in (
        "fia_kernel_nonquant_mla.h",
        "fia_block_cube_nonquant_mla.h",
        "fia_block_vec_nonquant_mla.h",
        "fia_kernel_nonquant.h",
    ):
        relative = f"attention/fused_infer_attention_score/op_kernel/arch22/{name}"
        pinned = subprocess.check_output(
            ["git", "-C", str(source), "show", f"{PINNED_CANN}:{relative}"]
        )
        candidates = sorted(vendor.glob(f"**/arch22/{name}"))
        if len(candidates) != 1 or candidates[0].read_bytes() != pinned:
            raise RuntimeError(f"Custom CANN header does not match the pinned source: {name}")
        headers.append(file_info(candidates[0]))
    return dict(
        path=str(vendor),
        loaded_libraries=libraries,
        pinned_headers=headers,
        packaged_kernels=vendor_kernel_files(vendor),
    )


def environment(cann_source=None, expected_vendor=None):
    """Capture loaded libraries AFTER running FIA, not just environment paths."""
    import torch_npu

    root = Path(__file__).resolve().parents[1]
    opp = Path(os.environ["ASCEND_OPP_PATH"]).resolve()
    toolkit = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    paths = {
        row.split()[-1] for row in Path("/proc/self/maps").read_text().splitlines() if "/" in row
    }
    selected = sorted(
        path
        for path in paths
        if any(
            name in Path(path).name
            for name in (
                "libopapi",
                "libophost_transformer",
                "liboptiling",
                "libcust_opapi",
                "libcust_opmaster",
                "libcust_opsproto",
            )
        )
    )
    version_paths = (opp / "version.info", toolkit / "share/info/ops_transformer/version.info")
    result = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,
        executable=sys.executable,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch=dict(version=torch.__version__, path=torch.__file__),
        torch_npu=dict(version=torch_npu.__version__, path=torch_npu.__file__),
        device=torch.npu.get_device_name(torch.npu.current_device()),
        device_index=torch.npu.current_device(),
        environment={
            name: os.environ.get(name)
            for name in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_OPP_PATH",
                "ASCEND_CUSTOM_OPP_PATH",
                "ASCEND_HOME_PATH",
                "LD_LIBRARY_PATH",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONPATH",
                "ASCEND_GLOBAL_LOG_LEVEL",
                "ASCEND_SLOG_PRINT_TO_STDOUT",
            )
        },
        repository_head=git_output(root, "rev-parse", "HEAD"),
        catlass_head=git_output(root / "csrc/catlass", "rev-parse", "HEAD"),
        harness=[
            file_info(root / path)
            for path in (
                "tests/mla_cann_baseline.py",
                "tests/mla_cann_profile.py",
                "tests/common/mla_ref.py",
                "tests/common/compare.py",
            )
        ],
        loaded_libraries=[file_info(path) for path in selected],
        installed_versions={
            str(path): path.read_text() for path in version_paths if path.is_file()
        },
        operator_schema=str(torch.ops.npu.npu_fused_infer_attention_score_v2.out._schema),
        pinned_cann_commit=PINNED_CANN,
        pinned_source_binary_correspondence="unverified; installed-package diagnostics only",
    )
    if cann_source is not None:
        source = Path(cann_source).resolve()
        relative = Path("attention/fused_infer_attention_score/op_kernel/arch22")
        installed = opp / "built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/arch32"
        result["source_checkout"] = dict(
            path=str(source),
            commit=git_output(source, "rev-parse", "HEAD"),
            scoped_status=git_output(
                source,
                "status",
                "--short",
                "--",
                "attention/fused_infer_attention_score",
                "attention/common",
            ),
            files=[
                dict(
                    source=file_info(source / relative / name),
                    installed=file_info(installed / name) if (installed / name).is_file() else None,
                )
                for name in (
                    "fia_kernel_nonquant_mla.h",
                    "fia_block_cube_nonquant_mla.h",
                    "fia_block_vec_nonquant_mla.h",
                    "fia_kernel_nonquant.h",
                )
            ],
        )
    if expected_vendor is not None:
        if cann_source is None:
            raise ValueError("Custom vendor verification requires a CANN source checkout")
        result["custom_vendor"] = verify_loaded_vendor(expected_vendor, cann_source, paths)
        result["pinned_source_binary_correspondence"] = (
            "custom host libraries loaded and four packaged headers match the pinned commit; "
            "packaged device binaries match their compiler metadata; "
            "native build and executed device binary provenance require separate evidence"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON evidence path (must not exist)"
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cases", nargs="+", choices=[case.name for case in CASES])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--profile-dir", type=Path, help="New directory for raw profiler evidence")
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--profile-rounds", type=int, default=3)
    parser.add_argument(
        "--cann-source", type=Path, help="Optional read-only checkout for provenance"
    )
    parser.add_argument(
        "--expected-vendor",
        type=Path,
        help="Require this custom OPP's host libraries and pinned headers; needs --cann-source",
    )
    args = parser.parse_args()
    if (
        min(args.warmup, args.iterations, args.rounds, args.profile_iterations, args.profile_rounds)
        <= 0
    ):
        parser.error("warmup, iterations and rounds must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve raw evidence")
    if args.profile_dir is not None and args.profile_dir.exists():
        parser.error("profile directory already exists; choose a new path")
    if args.profile_dir is not None and os.environ.get("ASCEND_GLOBAL_LOG_LEVEL", "3") != "3":
        parser.error("profiling requires ASCEND_GLOBAL_LOG_LEVEL=3 to exclude debug logging")
    if args.expected_vendor is not None:
        if args.cann_source is None:
            parser.error("--expected-vendor requires --cann-source")
        check_vendor_configuration(args.expected_vendor)
    require_910b(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="running",
        scope="Dense BSND CANN baseline; pinned-source attribution requires build and dispatch evidence",
        timing_contract="event spans include dispatch gaps/copies; unprofiled host samples include output allocation, dispatch and synchronization; excludes H2D and one-time input views; profiler durations exclude dispatch gaps and report FIA and Slice kernels separately; kernel sums are device work, not wall latency",
        settings={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        results=[],
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        calls = []
        for case in CASES:
            if args.cases and case.name not in args.cases:
                continue
            for dtype_name in args.dtypes:
                row = dict(case=asdict(case), dtype=dtype_name, status="running")
                report["results"].append(row)
                save()
                call, accuracy = check_case(case, DTYPES[dtype_name], args.device, args.seed)
                row.update(accuracy=accuracy, inputs=call.description())
                if args.profile_dir is not None:
                    calls.append(call)
                row["timing"] = measure(
                    call, warmup=args.warmup, iterations=args.iterations, rounds=args.rounds
                )
                row["status"] = "passed"
                save()
                print(
                    f"PASS {case.name} {dtype_name}: O/LSE; event span {row['timing']['median_event_call_us']:.3f} us/call",
                    flush=True,
                )
        if args.profile_dir is not None:
            csv_path, profiles = profile_calls(
                calls,
                args.profile_dir,
                iterations=args.profile_iterations,
                rounds=args.profile_rounds,
            )
            report["profiler"] = dict(
                kernel_details=file_info(csv_path),
                level="Level0, CPU+NPU, no shapes/stacks/memory/AI-core metrics",
                attribution="serialized synchronized calls; chronological Slice* then FIA; validated counts; CSV Step Id is not used",
            )
            for row, profile in zip(report["results"], profiles):
                row["profile"] = profile
        report["environment"] = environment(args.cann_source, args.expected_vendor)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        # Preserve partial results even if execution or provenance collection fails.
        save()


if __name__ == "__main__":
    main()
