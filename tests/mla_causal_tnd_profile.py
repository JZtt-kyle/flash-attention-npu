"""Fixed causal varlen public E2E, with native packing/restoration inside timing."""

import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.common.mla_ref import mla_reference_pair
from tests.mla_cann_baseline import check_vendor_configuration, environment, file_info
from tests.mla_source_compare import assert_mla_source_close
from tests.mla_varlen_used_profile import CannTndUsedCall
from tests.test_flash_attn_npu_v3_mla_causal_tnd_public import (
    PERFORMANCE_CASES,
    prepare,
    call,
    effective,
)


class CausalTndCall(CannTndUsedCall):
    def __init__(self, case):
        args = case["args"]
        inputs = {name: args[name] for name in case["raw"]}
        super().__init__(
            inputs,
            case["spec"],
            args["softmax_scale"] if args["softmax_scale"] is not None else 576**-0.5,
        )
        self.native.kwargs.update(
            sparse_mode=3,
            atten_mask=torch.ones((2048, 2048), dtype=torch.bool, device=inputs["q"].device).triu(
                1
            ),
        )


def identity(case):
    return dict(
        spec=case["spec"],
        inputs={
            name: dict(
                shape=list(value.shape),
                stride=list(value.stride()),
                offset=value.storage_offset(),
                storage_bytes=value.untyped_storage().size(),
                sha256=hashlib.sha256(
                    case["raw"][name].contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            )
            for name, value in case["args"].items()
            if name in case["raw"]
        },
    )


def accuracy(case, actual, source):
    uq, uk = effective(case["spec"])
    raw = case["raw"]
    actual, source = ([t.cpu() for t in pair] for pair in (actual, source))
    qs = ks = 0
    reports = []
    for b, (q, k) in enumerate(zip(uq, uk)):
        if q:
            inputs = {
                name: t[qs : qs + q].unsqueeze(0).contiguous()
                if name in ("q", "qv")
                else t[ks : ks + k].unsqueeze(0).contiguous()
                for name, t in raw.items()
            }
            refs = mla_reference_pair(
                **inputs, softmax_scale=case["spec"].get("scale"), causal=True
            )
            if k == 0:
                refs[1].fill_(-torch.inf)
                refs[3].fill_(-torch.inf)
            got = [actual[0][qs : qs + q].unsqueeze(0), actual[1][:, qs : qs + q].unsqueeze(0)]
            src = [source[0][qs : qs + q].unsqueeze(0), source[1][:, qs : qs + q].unsqueeze(0)]
            masked = max(0, q - k)
            assert (src[0][:, :masked] == 0).all()
            assert src[1][:, :, :masked].isposinf().all()
            src[1][:, :, :masked] = -torch.inf
            comparison = {}
            for i, name in enumerate(("out", "lse")):
                if torch.equal(got[i], src[i]):
                    comparison[name] = assert_mla_source_close(
                        got[i], src[i], refs[i], refs[i + 2], name=name
                    )
                else:
                    # Native automatic FD can change arithmetic order. Keep
                    # its real dispatch and check both outputs against math.
                    assert_fa_close(got[i], refs[i], refs[i + 2], name=name)
                    assert_fa_close(src[i], refs[i], refs[i + 2], name="native-" + name)
                    comparison[name] = dict(source="different", math="passed")
            reports.append(comparison)
        qs += case["spec"]["q"][b]
        ks += case["spec"]["kv"][b]
    return reports


def sample_kernels(call_one, count, output, names, production_name, source_name):
    """Original 5 x 10 alternating profiler protocol; retain every task."""
    labels = []
    p = torch_npu.profiler
    with p.profile(
        activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
        schedule=p.schedule(wait=0, warmup=1, active=count * 5 * 10 * 2, repeat=1),
        on_trace_ready=p.tensorboard_trace_handler(str(output / "profiler")),
        experimental_config=p._ExperimentalConfig(
            profiler_level=p.ProfilerLevel.Level0,
            export_type=p.ExportType.Text,
            data_simplification=False,
        ),
    ) as prof:
        result = call_one(0, names[0])
        torch.npu.synchronize()
        prof.step()
        for index in range(count):
            for round_id in range(5):
                for iteration in range(10):
                    for name in names if round_id % 2 == 0 else names[::-1]:
                        result = call_one(index, name)
                        torch.npu.synchronize()
                        labels.append(
                            dict(case=index, name=name, round=round_id, iteration=iteration)
                        )
                        prof.step()
    del result
    csvs = list(output.rglob("kernel_details.csv"))
    raws = list(output.rglob("op_summary_*.csv"))
    assert len(csvs) == len(raws) == 1
    with csvs[0].open() as f:
        details = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Start Time(us)"].strip()))
    with raws[0].open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
    assert len(rows) == len(details)
    for row, detail in zip(rows, details):
        assert row["Op Name"] == detail["Name"] and row["Device_id"] == detail["Device_id"]
        assert Decimal(row["Task Start Time(us)"].strip()) == Decimal(
            detail["Start Time(us)"].strip()
        )
        assert float(row["Task Duration(us)"]) == float(detail["Duration(us)"])
    # Every timed submission ends in its attention kernel. Native gather and
    # restore tasks can surround it; use the following producer as the boundary.
    pos = 0
    patterns = {}
    for label in labels:
        if label["name"] == names[0]:
            group = rows[pos : pos + 2]
            assert len(group) == 2 and "ComputeMlaCausalTndMetadata" in group[0]["Op Name"]
            assert production_name in group[1]["Op Name"]
            label["generation_us"] = float(group[0]["Task Duration(us)"])
            kernel = group[1]
        else:
            index = label["case"]
            if index not in patterns:
                end = next(i for i in range(pos, len(rows)) if "ComputeMla" in rows[i]["Op Name"])
                patterns[index] = [row["Op Name"] for row in rows[pos:end]]
            pattern = patterns[index]
            group = rows[pos : pos + len(pattern)]
            assert [row["Op Name"] for row in group] == pattern
            kernels = [r for r in group if source_name in r["Op Name"]]
            # CANN conversions share an API prefix; the actual attention task
            # has the precise suffix, while the standalone source uses its name.
            if source_name == "FusedInferAttentionScore":
                kernels = [r for r in group if r["Op Name"].endswith("_FusedInferAttentionScore")]
            assert len(kernels) == 1, group
            kernel = kernels[0]
        pos += len(group)
        label.update(
            tasks=group,
            kernel_us=float(kernel["Task Duration(us)"]),
            conversion_us=sum(
                float(r["Task Duration(us)"])
                for r in group
                if r is not kernel and "ComputeMla" not in r["Op Name"]
            ),
        )
    assert pos == len(rows)
    (output / "kernel-samples.json").write_text(json.dumps(labels, indent=2) + "\n")
    return labels, dict(kernel_csv=str(csvs[0]), raw_csv=str(raws[0]))


def profile(dtype_name, output, vendor, dispatch=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vendor = check_vendor_configuration(vendor)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    torch.npu.set_device(0)
    cases = [prepare(dtype, s) for s in PERFORMANCE_CASES]
    natives = [CausalTndCall(c) for c in cases]
    identities = [identity(c) for c in cases]
    if dispatch is None:
        checks = []
        for index, (case, native) in enumerate(zip(cases, natives)):
            actual = call(case["args"])
            torch.npu.synchronize()
            os.write(1, f"MLA_STRIDE_NATIVE_BEGIN {dtype_name} {index}\n".encode())
            source = native()
            torch.npu.synchronize()
            os.write(1, f"MLA_STRIDE_NATIVE_END {dtype_name} {index}\n".encode())
            checks.append(accuracy(case, actual, source))
        (output / "accuracy.json").write_text(
            json.dumps(dict(inputs=identities, comparisons=checks), indent=2) + "\n"
        )
        env = environment(Path(__file__).resolve().parents[2] / "ops-transformer", vendor)
        env["fa3_extension"] = file_info(api.flash_attn_npu_3.__file__)
        (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
        return
    proof = json.loads(Path(dispatch).read_text())
    assert proof["inputs"] == identities and proof["dtype"] == dtype_name
    for path, digest in proof["artifacts"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    environment_paths = [Path(p) for p in proof["artifacts"] if Path(p).name == "environment.json"]
    assert len(environment_paths) == 1
    expected_environment = json.loads(environment_paths[0].read_text())
    assert expected_environment["fa3_extension"] == file_info(api.flash_attn_npu_3.__file__)
    (output / "environment.json").write_text(json.dumps(expected_environment, indent=2) + "\n")
    torch.set_num_threads(1)

    def run(index, name, repeats=1):
        function = natives[index] if name == "cann" else lambda: call(cases[index]["args"])
        return [function() for _ in range(repeats)]

    names = ("fa3", "cann")
    (output / "npu-before.log").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
    for index in range(len(cases)):
        for _ in range(5):
            for name in names:
                result = run(index, name, 4)
                torch.npu.synchronize()
    wall = []
    for index in range(len(cases)):
        for round_id in range(5):
            for iteration in range(20):
                for repeats in (1, 4):
                    for name in names if round_id % 2 == 0 else names[::-1]:
                        start = time.perf_counter_ns()
                        result = run(index, name, repeats)
                        torch.npu.synchronize()
                        wall.append(
                            dict(
                                case=index,
                                round=round_id,
                                iteration=iteration,
                                repeats=repeats,
                                name=name,
                                e2e_us=(time.perf_counter_ns() - start) / 1000,
                            )
                        )
    del result
    (output / "wall.json").write_text(json.dumps(wall, indent=2) + "\n")
    samples, paths = sample_kernels(
        run, len(cases), output, names, "mla_tnd_used_", "FusedInferAttentionScore"
    )
    results = []
    for index, case in enumerate(cases):
        row = dict(
            case=index,
            **identities[index],
            **{
                key: proof["cases"][index][key]
                for key in ("native_fd", "native_template", "native_tiling_key", "e2e_applicable")
            },
        )
        for r in (1, 4):
            times = {
                name: statistics.median(
                    v["e2e_us"]
                    for v in wall
                    if v["case"] == index and v["name"] == name and v["repeats"] == r
                )
                for name in names
            }
            row[f"r{r}"] = dict(**times, ratio=times["fa3"] / times["cann"])
        row["kernel_us"] = {
            name: statistics.median(
                v["kernel_us"] for v in samples if v["case"] == index and v["name"] == name
            )
            for name in names
        }
        row["conversion_kernel_us"] = statistics.median(
            v["conversion_us"] for v in samples if v["case"] == index and v["name"] == "cann"
        )
        results.append(row)
    report = dict(
        dtype=dtype_name,
        causal=True,
        warmup=5,
        rounds=5,
        wall_iterations=20,
        profiler_iterations=10,
        repeats=[1, 4],
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        results=results,
        dispatch=str(dispatch),
        **paths,
    )
    (output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "npu-after.log").write_text(subprocess.check_output(["npu-smi", "info"], text=True))


if __name__ == "__main__":
    import sys

    profile(*sys.argv[1:])
