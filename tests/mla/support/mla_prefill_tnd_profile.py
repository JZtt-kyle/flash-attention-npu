"""Public varlen R=1/4 timing with fixed native TND packing inside the boundary."""

import argparse
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch
from tests.mla.support.public_cases.prefill_tnd import make_case, metadata, invoke, public, compare
from tests.mla.support.mla_prefill_bsnd_profile import sample_kernels
from tests.mla.support.mla_varlen_used_profile import CannTndUsedCall
from tests.mla.support.mla_ref import normalize_fa3_window
from tests.mla.support.mla_prefill_tnd_cases import PERFORMANCE_CASES
from tests.mla.support.mla_native_dispatch import sha256, e2e_applicable
from tests.common.compare import assert_fa_close


class PrefillTndCall(CannTndUsedCall):
    def __init__(self, c):
        # Reuse established allocation-gap packing. These fixed timing inputs
        # retain all Q tokens, so no post-attention scatter is needed.
        assert c["qlens"] == c["q"]
        super().__init__(
            c["t"], dict(q=c["q"], kv=c["k"], uq=None, uk=c["klens"]), c["options"]["softmax_scale"]
        )
        o = c["options"]
        left, right = normalize_fa3_window(
            c["shape"][1], c["shape"][3], causal=o["causal"], window_size=o["window_size"]
        )
        kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
        self.kwargs = dict(
            actual_seq_qlen=list(itertools.accumulate(c["qlens"])),
            actual_seq_kvlen=list(itertools.accumulate(c["klens"])),
            num_query_heads=c["shape"][2],
            num_key_value_heads=c["shape"][4],
            softmax_scale=o["softmax_scale"],
            input_layout="TND",
            return_softmax_lse=True,
            sparse_mode=3 if kind == 1 else 4 if kind == 2 else 0,
            pre_tokens=2**31 - 1 if left < 0 else left,
            next_tokens=2**31 - 1 if right < 0 else right,
        )
        if kind:
            self.kwargs["atten_mask"] = torch.ones(
                (2048, 2048), dtype=torch.bool, device="npu"
            ).triu(1)

    def __call__(self):
        p = self.pack()
        q = p["q"]
        out = torch.empty((*q.shape[:2], 128), dtype=q.dtype, device=q.device)
        lse = torch.empty((*q.shape[:2], 1), dtype=torch.float32, device=q.device)
        out, lse = torch.ops.npu.npu_fused_infer_attention_score_v2.out(
            q, p["k"], p["v"], **self.kwargs, out=[out, lse]
        )
        return out, lse.squeeze(-1).T

    def description(self):
        return {k: v for k, v in self.kwargs.items() if k != "atten_mask"}


def run(dtype, directory, dispatch=None, check=False):
    directory.mkdir(parents=True, exist_ok=True)
    specs = PERFORMANCE_CASES[:1] if check else PERFORMANCE_CASES
    cases = [make_case(dtype, **{k: v for k, v in s.items() if k != "window"}) for s in specs]
    natives = [PrefillTndCall(c) for c in cases]
    records = [metadata(c) for c in cases]
    if dispatch is None:
        checks = []
        for i, (c, native, record) in enumerate(zip(cases, natives, records)):
            refs = compare(c, public(c), record)
            print(f"MLA_STRIDE_NATIVE_BEGIN {dtype} {i}", flush=True)
            source = native()
            torch.npu.synchronize()
            print(f"MLA_STRIDE_NATIVE_END {dtype} {i}", flush=True)
            for j, name in enumerate(("out", "lse")):
                value = source[j].cpu()
                if j == 1:
                    value[value.isposinf()] = -torch.inf
                assert_fa_close(
                    value,
                    refs[name + "_high"],
                    refs[name + "_low"],
                    name="native-prefill-tnd-" + name,
                )
            checks.append(
                dict(spec=specs[i], record=record.cpu().tolist(), native=native.description())
            )
        (directory / "accuracy.json").write_text(json.dumps(checks, indent=2) + "\n")
        return
    proof = json.loads(dispatch.read_text())
    assert proof["dtype"] == dtype and len(proof["cases"]) == len(specs)
    assert proof["specs"] == specs and all(sha256(p) == v for p, v in proof["artifacts"].items())
    assert os.environ.get("ASCEND_GLOBAL_LOG_LEVEL") == "3"

    def call(i, name, r=1):
        return [natives[i]() if name == "native_cann" else public(cases[i]) for _ in range(r)]

    (directory / "npu-before.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )
    wall = []
    for i in range(len(specs)):
        for _ in range(5):
            for name in ("fa3", "native_cann"):
                result = call(i, name, 4)
                torch.npu.synchronize()
        for round_id in range(5):
            for iteration in range(20):
                for r in (1, 4):
                    names = ("fa3", "native_cann")
                    for name in names if round_id % 2 == 0 else names[::-1]:
                        start = time.perf_counter_ns()
                        result = call(i, name, r)
                        torch.npu.synchronize()
                        wall.append(
                            dict(
                                case=i,
                                name=name,
                                repeats=r,
                                round=round_id,
                                iteration=iteration,
                                e2e_us=(time.perf_counter_ns() - start) / 1000,
                            )
                        )
    del result
    (directory / "wall-samples.json").write_text(json.dumps(wall, indent=2) + "\n")
    # Kernel-only sampling uses the same validated record, while public wall
    # samples above include v28 generation on every varlen call.
    kernels = sample_kernels(
        lambda i, name: invoke(cases[i], records[i]) if name == "fa3" else natives[i](),
        specs,
        directory,
        dtype,
    )
    rows = []
    for i, spec in enumerate(specs):
        native = proof["cases"][i]
        row = dict(
            case=i,
            spec=spec,
            native_template=native["native_template"],
            native_fd=native["native_fd"],
            native_call=natives[i].description(),
        )
        row["e2e_applicable"] = e2e_applicable(
            row["native_template"],
            row["native_fd"],
            representation="prefill",
            include_native_fd=spec["splits"] == 0,
        )
        for r in (1, 4):
            times = {
                name: statistics.median(
                    x["e2e_us"]
                    for x in wall
                    if x["case"] == i and x["name"] == name and x["repeats"] == r
                )
                for name in ("fa3", "native_cann")
            }
            row[f"r{r}"] = dict(
                fa3=times["fa3"],
                cann=times["native_cann"],
                ratio=times["fa3"] / times["native_cann"],
            )
        row["kernel_us"] = {
            name: statistics.median(
                x["kernel_us"] for x in kernels if x["case"] == i and x["name"] == name
            )
            for name in ("fa3", "native_cann")
        }
        rows.append(row)
    (directory / "profile.json").write_text(
        json.dumps(
            dict(
                dtype=dtype,
                representation="prefill",
                causal=False,
                window=True,
                warmup=5,
                rounds=5,
                wall_iterations=20,
                repeats=[1, 4],
                profiler_iterations=10,
                results=rows,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            ),
            indent=2,
        )
        + "\n"
    )
    (directory / "npu-after.log").write_text(
        subprocess.check_output(["npu-smi", "info"], text=True)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dispatch", type=Path)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    run(a.dtype, a.output, a.dispatch, a.check)
