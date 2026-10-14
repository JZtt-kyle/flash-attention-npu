"""BSND/TND PA causal public E2E, with native layout conversions inside each call."""

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla_cann_baseline import check_vendor_configuration, environment, file_info
from tests.mla_causal_tnd_profile import sample_kernels
from tests.mla_paged_profile import CannPagedCall
from tests.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests import test_flash_attn_npu_v3_mla_causal_paged_public as bsnd
from tests import test_flash_attn_npu_v3_mla_causal_paged_tnd_public as tnd


class CausalPagedCall(CannPagedCall):
    def __init__(self, case):
        super().__init__(
            case["args"], case["config"]["lengths"], case["scale"], case["config"]["q"]
        )
        self.kwargs.update(sparse_mode=3, next_tokens=0, atten_mask=case["mask"])


def identity(case, spec=None):
    tensors = dict(case["host"], page_table=case["table_host"])
    if spec is None:
        spec = {k: v for k, v in case["config"].items() if k != "causal"}
    return dict(
        spec=json.loads(json.dumps(spec)),
        inputs={
            name: dict(
                shape=list(t.shape),
                stride=list(t.stride()),
                offset=t.storage_offset(),
                sha256=hashlib.sha256(
                    t.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            )
            for name, t in tensors.items()
        },
    )


def profile(
    dtype_name,
    output,
    vendor,
    probe_dir,
    producer_manifest,
    dispatch=None,
    *,
    layout="bsnd",
    suite=None,
    native_class=CausalPagedCall,
    mask="causal",
    metadata_name=None,
):
    suite = suite or {"bsnd": bsnd, "tnd": tnd}[layout]
    component, call, metadata = suite.component, suite.call, suite.metadata
    make_case, compare, forward_args = suite.make_case, suite.compare, suite.forward_args
    masked_rows, PERFORMANCE_CASES = suite.masked_rows, suite.PERFORMANCE_CASES
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vendor = check_vendor_configuration(vendor)
    torch.npu.set_device(0)
    torch.set_num_threads(8)
    probes = component.load_attention(probe_dir)
    manifest = json.loads(Path(producer_manifest).read_text())
    assert (
        hashlib.sha256(Path(manifest["library"]).read_bytes()).hexdigest()
        == manifest["library_sha256"]
    )
    with load_probe(manifest["library"], output / "producer") as producer:
        component.configure_producer(producer)
        cases = [make_case(probes, producer, s) for s in PERFORMANCE_CASES]
    for case in cases:
        case["args"] = forward_args(case)
    natives = [native_class(c) for c in cases]
    # Physical capacity is filled by make_case; keep the frozen input spec as
    # the public matrix identity and retain materialized descriptors separately.
    identities = [identity(c, spec) for c, spec in zip(cases, PERFORMANCE_CASES)]
    if dispatch is None:
        checks = []
        for index, (case, native) in enumerate(zip(cases, natives)):
            actual = call(case["args"])[:2]
            compare(probes, case, [actual])
            os.write(1, f"MLA_STRIDE_NATIVE_BEGIN {dtype_name} {index}\n".encode())
            source = native()
            torch.npu.synchronize()
            os.write(1, f"MLA_STRIDE_NATIVE_END {dtype_name} {index}\n".encode())
            values = component.valid_outputs(case, [t.cpu() for t in source])
            masked = masked_rows(case)
            assert (values[0][masked] == 0).all() and values[1][:, masked].isposinf().all()
            values[1][:, masked] = -torch.inf
            refs = component.references(case)
            native_checks = {}
            for i, name in enumerate(("out", "lse")):
                assert_fa_close(values[i], refs[i], refs[i + 2], name="native-" + name)
                native_checks[name] = "passed"
            checks.append(native_checks)
            torch.save(
                dict(
                    spec=case["config"],
                    inputs=case["host"],
                    pages=case["table_host"],
                    actual=[t.cpu() for t in actual],
                    native=values,
                    references=refs,
                ),
                output / f"accuracy-{index}.pt",
            )
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

    records = [metadata(c["args"]) for c in cases]
    torch.npu.synchronize()

    def run(index, name, repeats=1, entry="implicit"):
        if name == "cann":
            return [natives[index]() for _ in range(repeats)]
        args = cases[index]["args"]
        record = (
            metadata(args) if entry == "generate" else records[index] if entry == "reuse" else None
        )
        return [call(dict(args, scheduler_metadata=record)) for _ in range(repeats)]

    names = ("fa3", "cann")
    (output / "npu-before.log").write_text(subprocess.check_output(["npu-smi", "info"], text=True))
    entries = ("implicit", "generate", "reuse")
    for index in range(len(cases)):
        for _ in range(5):
            for entry in entries:
                for name in names:
                    result = run(index, name, 4, entry)
                    torch.npu.synchronize()
    wall = []
    for index in range(len(cases)):
        for round_id in range(5):
            for iteration in range(20):
                for entry in entries:
                    for repeats in (1, 4):
                        for name in names if round_id % 2 == 0 else names[::-1]:
                            start = time.perf_counter_ns()
                            result = run(index, name, repeats, entry)
                            torch.npu.synchronize()
                            wall.append(
                                dict(
                                    case=index,
                                    round=round_id,
                                    iteration=iteration,
                                    repeats=repeats,
                                    entry=entry,
                                    name=name,
                                    e2e_us=(time.perf_counter_ns() - start) / 1000,
                                )
                            )
    del result
    (output / "wall.json").write_text(json.dumps(wall, indent=2) + "\n")
    generation = []
    for index, case in enumerate(cases):
        for _ in range(5):
            _record = metadata(case["args"])
            torch.npu.synchronize()
        for round_id in range(5):
            for iteration in range(20):
                start = time.perf_counter_ns()
                _record = metadata(case["args"])
                torch.npu.synchronize()
                generation.append(
                    dict(
                        case=index,
                        round=round_id,
                        iteration=iteration,
                        e2e_us=(time.perf_counter_ns() - start) / 1000,
                    )
                )
    (output / "metadata-wall.json").write_text(json.dumps(generation, indent=2) + "\n")
    samples, paths = sample_kernels(
        run,
        len(cases),
        output,
        names,
        "mla_paged_tnd_" if layout == "tnd" else "mla_paged_bsnd_",
        "FusedInferAttentionScore",
        metadata_name
        or (
            "ComputeMlaCausalPagedTndMetadata"
            if layout == "tnd"
            else "ComputeMlaCausalPagedBsndMetadata"
        ),
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
        if mask == "window":
            row["native_call"] = natives[index].description()
        row["entries"] = {}
        for entry in entries:
            row["entries"][entry] = {}
            for r in (1, 4):
                times = {
                    name: statistics.median(
                        v["e2e_us"]
                        for v in wall
                        if v["case"] == index
                        and v["name"] == name
                        and v["repeats"] == r
                        and v["entry"] == entry
                    )
                    for name in names
                }
                row["entries"][entry][f"r{r}"] = dict(**times, ratio=times["fa3"] / times["cann"])
        row["metadata_us"] = statistics.median(
            v["e2e_us"] for v in generation if v["case"] == index
        )
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
        causal=mask == "causal",
        window=mask == "window",
        layout=layout,
        warmup=5,
        rounds=5,
        wall_iterations=20,
        profiler_iterations=10,
        repeats=[1, 4],
        entries=list(entries),
        metadata_lifecycle=dict(
            complete_segments=[1, 4],
            generate="one public generation plus R forwards",
            reuse="one prior record plus R forwards",
            implicit="R forwards each internally generating its record",
        ),
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
