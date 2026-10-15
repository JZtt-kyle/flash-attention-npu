"""Production BSND/TND/mixed Kernel versus pinned, identically scheduled CANN.

The production wall includes internal metadata generation; the source probe has
preallocated buffers. Neither wall is a public FA3/CANN E2E comparison.
"""

import argparse
import csv
from decimal import Decimal
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch_npu

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.components.metadata import load_probe


def profile(probe_dir, producer_manifest, output, layout="bsnd"):
    paged = layout in ("paged-bsnd", "paged-tnd")
    if paged:
        if layout == "paged-tnd":
            from tests.mla.support.public_cases import paged_tnd_production as cases_api
        else:
            from tests.mla.support.public_cases import paged_bsnd_production as cases_api
        from tests.mla.support.components.paged_attention import (
            configure_producer,
            launch,
            load_attention,
        )

        production_case = cases_api.make_case
    elif layout in ("tnd-stride", "mixed-stride"):
        if layout == "tnd-stride":
            from tests.mla.support.public_cases import tnd_stride_production as cases_api
        else:
            from tests.mla.support.public_cases import mixed_stride_production as cases_api
        from tests.mla.support.components.tnd_stride import (
            configure_producer,
            launch,
            load_attention,
        )

        production_case = cases_api.production_case
    elif layout == "bsnd-stride":
        from tests.mla.support.public_cases import bsnd_stride_production as cases_api
        from tests.mla.support.components.bsnd_stride import (
            configure_producer,
            launch,
            load_attention,
        )

        production_case = cases_api.make_case
    elif layout == "mixed":
        from tests.mla.support.public_cases import mixed_used as cases_api
    else:
        assert layout == "bsnd"
        from tests.mla.support.public_cases import bsnd_used as cases_api

    if layout not in ("bsnd-stride", "tnd-stride", "mixed-stride", "paged-bsnd", "paged-tnd"):
        from tests.mla.support.components.used_attention import (
            configure_producer,
            launch,
            load_attention,
        )

        production_case = cases_api.production_case
    probes = load_attention(probe_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if paged:
        # Persist the full matrix and native dispatch before collecting samples.
        dispatch = json.loads(
            (
                Path(probe_dir)
                / (
                    "tnd-production-schedules.json"
                    if layout == "paged-tnd"
                    else "production-schedules.json"
                )
            ).read_text()
        )
        (output / "matrix.json").write_text(
            json.dumps(cases_api.PERFORMANCE_CASES, indent=2) + "\n"
        )
        (output / "dispatch.json").write_text(json.dumps(dispatch, indent=2) + "\n")
    call_forward = cases_api.forward if paged else lambda args: api._flash_attn_forward(**args)
    manifest = json.loads(Path(producer_manifest).read_text())
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    assert sha256(Path(manifest["library"])) == manifest["library_sha256"]
    with load_probe(manifest["library"], output / "producer") as producer:
        configure_producer(producer)
        cases = [production_case(probes, producer, spec) for spec in cases_api.PERFORMANCE_CASES]
        arguments = [cases_api.forward_args(case) for case in cases]
        errors = []
        for case, args in zip(cases, arguments):
            launch(probes, case, "oracle")
            errors.append(cases_api.check_result(case, call_forward(args), source=True))

        if paged:
            # Both timed paths share one fixed CPU. Reference math above is
            # outside the timing boundary and can use the configured threads.
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

        def call(index, name):
            if name == "source":
                launch(probes, cases[index], "oracle")
            else:
                call_forward(arguments[index])
            torch.npu.synchronize()

        (output / "npu-before.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        names = ("source", "production")
        for index in range(len(cases)):
            for _ in range(5):
                for name in names:
                    call(index, name)
        walls = []
        for index in range(len(cases)):
            for round_id in range(5):
                for iteration in range(20):
                    for name in names if round_id % 2 == 0 else names[::-1]:
                        start = time.perf_counter_ns()
                        call(index, name)
                        walls.append(
                            dict(
                                case=index,
                                name=name,
                                round=round_id,
                                iteration=iteration,
                                wall_us=(time.perf_counter_ns() - start) / 1000,
                            )
                        )
        (output / "wall.json").write_text(json.dumps(walls, indent=2) + "\n")

        labels = []
        trace = output / "profiler"
        trace.mkdir(exist_ok=False)
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
            call(0, "source")
            prof.step()
            for index in range(len(cases)):
                for round_id in range(5):
                    for iteration in range(10):
                        for name in names if round_id % 2 == 0 else names[::-1]:
                            call(index, name)
                            labels.append(
                                dict(case=index, name=name, round=round_id, iteration=iteration)
                            )
                            prof.step()
        (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
        csvs = list(trace.rglob("kernel_details.csv"))
        raws = list(trace.rglob("op_summary_*.csv"))
        assert len(csvs) == len(raws) == 1
        with csvs[0].open() as f:
            rows = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Start Time(us)"].strip()))
        with raws[0].open() as f:
            raw = sorted(csv.DictReader(f), key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
        assert len(rows) == len(raw) == len(labels) * 3 // 2
        for row, r in zip(rows, raw):
            assert row["Name"] == r["Op Name"] and row["Device_id"] == r["Device_id"]
            assert Decimal(row["Start Time(us)"].strip()) == Decimal(
                r["Task Start Time(us)"].strip()
            )
            assert float(row["Duration(us)"]) == float(r["Task Duration(us)"])
        pos = 0
        for label in labels:
            if label["name"] == "production":
                version = 6 if paged else cases[label["case"]].get("config", {}).get("version", 5)
                producer_name = {
                    1: "ComputeMlaDenseMetadata",
                    2: "ComputeMlaBsndKvLengthMetadata",
                    3: "ComputeMlaTndMetadata",
                    4: "ComputeMlaTndBsndMetadataFromPrefix",
                    5: "ComputeMlaUsedMetadata",
                    6: "ComputeMlaPagedMetadata",
                }[version]
                assert producer_name in raw[pos]["Op Name"]
                label["generation_us"] = float(raw[pos]["Task Duration(us)"])
                pos += 1
            row = raw[pos]
            expected_name = (
                (
                    "cann_paged_attention"
                    if paged
                    else "cann_tnd_stride"
                    if layout in ("tnd-stride", "mixed-stride")
                    else "cann_bsnd_stride"
                    if layout == "bsnd-stride"
                    else "cann_used"
                )
                if label["name"] == "source"
                else (
                    ("mla_paged_tnd" if layout == "paged-tnd" else "mla_paged_bsnd")
                    if paged
                    else ("mla_tnd_used" if version == 5 else "mla_tnd")
                    if layout == "tnd-stride"
                    else ("mla_tnd_used" if version == 5 else "mla_tnd_bsnd")
                    if layout == "mixed-stride"
                    else "mla_dense"
                    if layout == "bsnd-stride" and version != 5
                    else ("mla_tnd_used" if layout == "mixed" else "mla_bsnd_used")
                )
            )
            assert expected_name in row["Op Name"]
            label.update(kernel_us=float(row["Task Duration(us)"]), kernel_name=row["Op Name"])
            pos += 1
        assert pos == len(raw)
        (output / "kernel-samples.json").write_text(json.dumps(labels, indent=2) + "\n")
        results = []
        for index, case in enumerate(cases):
            result = dict(item=case.get("config", case["item"]), errors=errors[index])
            if layout in ("bsnd-stride", "tnd-stride", "mixed-stride", "paged-bsnd", "paged-tnd"):
                result["source_metadata"] = case["expected"]
                result["launched_blocks"] = (
                    case["context"]
                    if layout in ("tnd-stride", "mixed-stride", "paged-bsnd", "paged-tnd")
                    or case["config"]["version"] == 5
                    else case["dense_context"]
                ).block_dim
                result["input_views"] = {
                    n: dict(shape=list(t.shape), stride=list(t.stride()), offset=t.storage_offset())
                    for n, t in case["devices"]["migrated"].items()
                }
                if paged:
                    assert case["expected"] == dispatch["metadata"][index]
                    result["native_fd"] = bool(dispatch["native"][index][5])
            for name in names:
                samples = [s for s in labels if s["case"] == index and s["name"] == name]
                result[name] = dict(
                    kernel_us=statistics.median(s["kernel_us"] for s in samples),
                    wall_us=statistics.median(
                        s["wall_us"] for s in walls if s["case"] == index and s["name"] == name
                    ),
                )
                if name == "production":
                    result[name]["generation_us"] = statistics.median(
                        s["generation_us"] for s in samples
                    )
            result["kernel_ratio"] = (
                result["production"]["kernel_us"] / result["source"]["kernel_us"]
            )
            results.append(result)
        extension = Path(api.flash_attn_npu_3.__file__)
        report = dict(
            scope=(
                "Production internal BSND/TND PA versus pinned CANN PA/non-FD Kernel with identical logical inputs, schedules and launch blocks. Source Q/cache-row/table materialization is before timing. Production wall includes internal v6 generation, allocations and device validation; source probe buffers are preallocated. Not public PA E2E; native FD dispatch is separate."
                if paged
                else "Production BSND/TND/mixed stride versus pinned non-FD source, with identical logical inputs, schedules and launch blocks. Source receives contiguous materializations and packed effective TND segments prepared before timing; physical input layouts differ. Production wall includes internal metadata generation; source probe buffers are preallocated. Not public metadata E2E."
                if layout in ("bsnd-stride", "tnd-stride", "mixed-stride")
                else __doc__
            ),
            layout=layout,
            seed=20261011 if paged else 20261010,
            warmup=5,
            rounds=5,
            wall_iterations=20,
            profiler_iterations=10,
            dtype=str(probes[0]["dtype"]),
            device=torch.npu.get_device_name(0),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            torch_version=torch.__version__,
            torch_npu_version=torch_npu.__version__,
            extension=str(extension),
            extension_sha256=sha256(extension),
            probe_manifest=str(Path(probe_dir) / "build-manifest.json"),
            producer_manifest=str(producer_manifest),
            raw_csv=str(raws[0]),
            kernel_csv=str(csvs[0]),
            results=results,
        )
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "npu-after.log").write_text(
            subprocess.check_output(["npu-smi", "info"], text=True)
        )
        print([(r["kernel_ratio"], r["production"]) for r in results], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir")
    parser.add_argument("producer_manifest")
    parser.add_argument("output")
    parser.add_argument(
        "--layout",
        choices=(
            "bsnd",
            "mixed",
            "bsnd-stride",
            "tnd-stride",
            "mixed-stride",
            "paged-bsnd",
            "paged-tnd",
        ),
        default="bsnd",
    )
    args = parser.parse_args()
    profile(args.probe_dir, args.producer_manifest, args.output, args.layout)
