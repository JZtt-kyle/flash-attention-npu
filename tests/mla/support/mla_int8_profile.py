"""Fixed INT8 component samples: NZ oracle, direct ND target; no public E2E."""

import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import torch

from tests.mla.support.mla_int8_cases import SOURCE_CASES, make_inputs, strided_inputs
from tests.mla.support.mla_int8_component import ComponentCall
from tests.mla.support.mla_split_kv_attention_profile import sample_components


# Geometry covers both source FD services, direct paths, mask/empty/length
# boundaries. Two ND row-stride cases account for adaptation in the timed kernel.
PERFORMANCE_CASES = tuple((i, "contiguous") for i in range(len(SOURCE_CASES))) + (
    (3, "row-gap"),
    (12, "row-gap"),
)


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    calls, records = [], []
    for index, view in PERFORMANCE_CASES:
        spec = SOURCE_CASES[index]
        inputs = make_inputs(spec) if view == "contiguous" else strided_inputs(spec, view)
        pair = {
            name: ComponentCall(spec, inputs, oracle=name == "oracle")
            for name in ("oracle", "migrated")
        }
        results = {name: call().cpu() for name, call in pair.items()}
        assert torch.isfinite(results["oracle"]).all() and torch.isfinite(results["migrated"]).all()
        assert torch.equal(results["oracle"], results["migrated"])
        for call in pair.values():
            assert torch.all(call.workspace[call.workspace_size :].cpu() == 123)
        calls.append(pair)
        records.append(
            dict(
                spec=spec,
                view=view,
                source_pack_seconds=pair["oracle"].native.pack_seconds,
                key=pair["oracle"].fixture["key"],
                cores=pair["oracle"].fixture["cores"],
                fd_balance=pair["oracle"].params.tndSplitCoreParams.FdBalanceFlag,
                source_elementwise=True,
            )
        )
    (output / "cases.json").write_text(json.dumps(records, indent=2) + "\n")
    return calls, records


def profile(output):
    output = Path(output)
    calls, records = prepare(output)
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    (output / "npu-before.log").write_text(subprocess.check_output(["npu-smi", "info"], text=True))

    def call(index, name):
        calls[index][name]()
        torch.npu.synchronize()

    try:
        for i in range(len(calls)):
            for _ in range(5):
                for name in ("oracle", "migrated"):
                    call(i, name)
        results, paths = sample_components(call, PERFORMANCE_CASES, output, "int8_component")
        assert len(results) == len(PERFORMANCE_CASES)
        for row in results:
            assert all(math.isfinite(row[k]) and row[k] > 0 for k in ("kernel_ratio", "wall_ratio"))

        def segment(indices):
            return dict(
                count=len(indices),
                kernel_ratio=statistics.mean(results[i]["kernel_ratio"] for i in indices),
                wall_ratio=statistics.mean(results[i]["wall_ratio"] for i in indices),
                kernel_regressions=sum(results[i]["kernel_ratio"] > 1 for i in indices),
                wall_regressions=sum(results[i]["wall_ratio"] > 1 for i in indices),
            )

        report = dict(
            scope="Preallocated one-kernel INT8 component: source NZ versus target ND on identical native partitions. O-only endpoints; LSE is separately verified by unit tests. Source NZ packing is preparation, recorded per case; target ND conversion remains inside its kernel. No production metadata or public E2E claim.",
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            results=results,
            **paths,
            average=segment(range(len(results))),
            native_fd=segment(
                [i for i, pair in enumerate(calls) if pair["oracle"].fixture["mode"] & 2]
            ),
            direct=segment(
                [i for i, pair in enumerate(calls) if not pair["oracle"].fixture["mode"] & 2]
            ),
        )
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "npu-after.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        return report
    finally:
        os.sched_setaffinity(0, affinity)
