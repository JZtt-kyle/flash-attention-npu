"""BSND Q-used production Kernel versus pinned non-FD source on identical schedules."""

import csv
import ctypes as ct
from decimal import Decimal
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch_npu

from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.public_cases.causal_public import USED_CASES, prepare, source_tiles
from tests.mla.support.components.causal_attention import make_case
from tests.mla.support.components.dense import Schedule


def build_source(probes):
    directory = probes[1]
    command = json.loads((directory / "oracle-command.json").read_text())
    command = [
        str(ROOT / "tests/mla/support/mla_causal_used_probe.cpp")
        if x.endswith("/mla_causal_attention_probe.cpp")
        else x
        for x in command
    ]
    library = directory / "causal-used-source.so"
    command[command.index("-o") + 1] = str(library)
    command[command.index("-MF") + 1] = str(directory / "causal-used-source.d")
    (directory / "causal-used-command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "causal-used-build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    deps = [
        x
        for x in command[: command.index("-MMD")]
        if x != "-shared" and not x.startswith(("-l", "-L", "-Wl,"))
    ]
    depfile = directory / "causal-used-device.d"
    deps += ["--offload-device-only", "-M", "-MF", str(depfile)]
    with (directory / "causal-used-deps.log").open("w") as log:
        subprocess.run(deps, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
    paths = [Path(p) for p in depfile.read_text().replace("\\\n", " ").split(":", 1)[1].split()]
    dependencies = {str(p): sha256(p) for p in paths}
    assert any(p.endswith("fia_kernel_nonquant_mla.h") for p in dependencies)
    identity = dict(library=str(library), sha256=sha256(library), dependencies=dependencies)
    (directory / "causal-used-loaded.json").write_text(json.dumps(identity, indent=2) + "\n")
    loaded = ct.CDLL(str(library))
    loaded.launch_causal_used.argtypes = [ct.c_void_p] * 10 + [ct.c_float] + [ct.c_void_p] * 3
    loaded.launch_causal_used.restype = ct.c_int
    return loaded


def profile(probes, metadata_driver, used_oracle, output):
    output = Path(output)
    output.mkdir(parents=True)
    source = build_source(probes)
    cases = []
    for spec in USED_CASES:
        raw, device, lengths, used, details = prepare(probes, spec)
        tile = source_tiles(probes, spec, used_oracle, details)
        case = make_case(probes, spec)
        for name in raw:
            case["device"][name].copy_(raw[name])
        batch, sq, heads, sk = spec["shape"]
        kv_lengths = spec.get("lengths") or [sk] * batch
        # The CPU producer is independently source-differential tested; its wire
        # schedule exactly models the production full-core launch and used spans.
        cores = int(case["details"][2])
        request = [
            batch,
            sq,
            heads,
            sk,
            1 if probes[0]["dtype"] == torch.float16 else 2,
            1,
            0 if spec.get("implicit_kv") else 1,
            cores,
            *itertools.chain.from_iterable(zip(spec["used"], kv_lengths)),
        ]
        words = metadata_driver.evaluate([request])[0]
        assert words[5] == 0
        assert words[15:17] == tile
        schedule = Schedule()
        for field, value in zip(
            (
                "batch",
                "sq",
                "heads",
                "sk",
                "m_base",
                "s2_base",
                "cores",
                "block_dim",
                "ratio",
                "r1",
                "r2",
            ),
            (
                batch,
                sq,
                heads,
                sk,
                words[15],
                words[16],
                words[20],
                words[21],
                2,
                words[367],
                words[369],
            ),
        ):
            setattr(schedule, field, value)
        for axis, start in (("b_end", 23), ("m_end", 49), ("s_end", 75)):
            getattr(schedule, axis)[:] = words[start : start + 26]
        case.update(
            schedule=schedule,
            production=device,
            used=used,
            lengths32=None if spec.get("implicit_kv") else lengths,
            used64=torch.tensor(spec["used"], dtype=torch.int64, device="npu"),
            source_workspace=torch.empty(
                2 * schedule.block_dim * (6 * schedule.r1 + 8 * schedule.r2),
                dtype=torch.uint8,
                device="npu",
            ),
        )
        cases.append(case)
    (output / "schedules.json").write_text(
        json.dumps(
            [
                dict(
                    config=spec,
                    block_dim=c["schedule"].block_dim,
                    cores=c["schedule"].cores,
                    tile=[c["schedule"].m_base, c["schedule"].s2_base],
                    endpoints=list(
                        zip(c["schedule"].b_end, c["schedule"].m_end, c["schedule"].s_end)
                    ),
                )
                for spec, c in zip(USED_CASES, cases)
            ],
            indent=2,
        )
        + "\n"
    )
    names = ("oracle", "production")

    def call(index, name):
        case = cases[index]
        if name == "production":
            result = torch.ops.flash_attn_npu_3._mla_causal_bsnd_forward(
                **case["production"], cache_seqlens=case["lengths32"], seqused_q=case["used"]
            )
        else:
            d = case["device"]
            ptrs = [
                d["qv"],
                d["q"],
                d["v"],
                d["k"],
                case["desc"],
                *case["outputs"]["oracle"],
                case["source_workspace"],
            ]
            assert (
                source.launch_causal_used(
                    torch.npu.current_stream().npu_stream,
                    *(x.data_ptr() for x in ptrs),
                    ct.byref(case["schedule"]),
                    case["scale"],
                    case["device_lengths"]["oracle"].data_ptr(),
                    probes[0]["mask"].data_ptr(),
                    case["used64"].data_ptr(),
                )
                == 0
            )
            result = case["outputs"]["oracle"]
        torch.npu.synchronize()
        return result

    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    try:
        (output / "npu-before.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        for index, spec in enumerate(USED_CASES):
            for _ in range(5):
                expected, actual = (call(index, name) for name in names)
            kv_lengths = spec.get("lengths") or [spec["shape"][3]] * spec["shape"][0]
            for b, (sq, sk) in enumerate(zip(spec["used"], kv_lengths)):
                torch.testing.assert_close(actual[0][b, :sq], expected[0][b, :sq], rtol=0, atol=0)
                empty = max(0, sq - sk)
                torch.testing.assert_close(
                    actual[1][b, :, empty:sq], expected[1][b, :, empty:sq], rtol=0, atol=0
                )
        walls = []
        for index in range(len(cases)):
            for round_id in range(5):
                for iteration in range(20):
                    for name in names if round_id % 2 == 0 else names[::-1]:
                        begin = time.perf_counter_ns()
                        result = call(index, name)
                        walls.append(
                            dict(
                                case=index,
                                name=name,
                                round=round_id,
                                iteration=iteration,
                                wall_us=(time.perf_counter_ns() - begin) / 1000,
                            )
                        )
        (output / "wall.json").write_text(json.dumps(walls, indent=2) + "\n")
        labels, trace = [], output / "profiler"
        profiler = torch_npu.profiler
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            schedule=profiler.schedule(wait=0, warmup=1, active=len(cases) * 5 * 10 * 2, repeat=1),
            on_trace_ready=profiler.tensorboard_trace_handler(str(trace)),
            experimental_config=profiler._ExperimentalConfig(
                profiler_level=profiler.ProfilerLevel.Level0,
                export_type=profiler.ExportType.Text,
                data_simplification=False,
            ),
        ) as prof:
            call(0, "oracle")
            prof.step()
            for index in range(len(cases)):
                for round_id in range(5):
                    for iteration in range(10):
                        for name in names if round_id % 2 == 0 else names[::-1]:
                            result = call(index, name)
                            labels.append(
                                dict(case=index, name=name, round=round_id, iteration=iteration)
                            )
                            prof.step()
        del result
        csvs, raws = list(trace.rglob("kernel_details.csv")), list(trace.rglob("op_summary_*.csv"))
        assert len(csvs) == len(raws) == 1
        with csvs[0].open() as f:
            rows = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Start Time(us)"].strip()))
        with raws[0].open() as f:
            raw = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
        assert len(rows) == len(raw)
        for row, source_row in zip(rows, raw):
            assert row["Name"] == source_row["Op Name"]
            assert Decimal(row["Start Time(us)"].strip()) == Decimal(
                source_row["Task Start Time(us)"].strip()
            )
            assert float(row["Duration(us)"]) == float(source_row["Task Duration(us)"])
        producers = [r for r in rows if "ComputeMlaCausalBsndMetadata" in r["Name"]]
        rows = [r for r in rows if r not in producers]
        assert len(producers) in (0, len(labels) // 2)
        assert len(rows) == len(labels)
        for label, row in zip(labels, rows):
            assert ("cann_causal_used" if label["name"] == "oracle" else "mla_bsnd_used") in row[
                "Name"
            ]
            label.update(kernel_us=float(row["Duration(us)"]), kernel_name=row["Name"])
        results = []
        for index, spec in enumerate(USED_CASES):
            item = dict(config=spec)
            for name in names:
                item[name] = dict(
                    kernel_us=statistics.median(
                        x["kernel_us"] for x in labels if x["case"] == index and x["name"] == name
                    ),
                    wall_us=statistics.median(
                        x["wall_us"] for x in walls if x["case"] == index and x["name"] == name
                    ),
                )
            item["kernel_ratio"] = item["production"]["kernel_us"] / item["oracle"]["kernel_us"]
            results.append(item)
        report = dict(
            scope="Production Q-used attention Kernel versus fixed source with identical causal full-M schedules; source mask/int64 lengths/contiguous views and H2D prepared outside timing. Production wall includes metadata and allocations, source wall uses preallocated storage, so wall is diagnostic only.",
            dtype=str(probes[0]["dtype"]),
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            kernel_csv=str(csvs[0]),
            raw_csv=str(raws[0]),
            metadata_tasks=len(producers),
            results=results,
        )
        (output / "kernel-samples.json").write_text(json.dumps(labels, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "npu-after.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        return report
    finally:
        os.sched_setaffinity(0, affinity)
