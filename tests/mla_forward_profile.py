"""Public FA3 dense E2E diagnostics, including implicit MLA metadata generation."""

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
from tests.mla_cann_baseline import (
    CannCall,
    Case,
    check_vendor_configuration,
    environment,
    make_inputs,
)
from tests.mla_profile_cases import (
    BSND_CASES,
    CASES,
    KV_CASES,
    LONG_KV_CASES,
    S1G_CASES,
    STRIDE_CASES,
)


def check_causal_accuracy(actual, source, cpu, shape, scale, lengths):
    from tests.mla_source_compare import assert_mla_source_close
    from tests.test_flash_attn_npu_v3_mla_causal_attention import references

    batch, sq, heads, sk = shape
    full_refs = references(
        dict(
            shape=shape,
            inputs=cpu,
            lengths=lengths if lengths is not None else [sk] * batch,
            scale=scale,
        )
    )
    reports = []
    for b, length in enumerate(lengths if lengths is not None else [sk] * batch):
        refs = [ref[b : b + 1] for ref in full_refs]
        got = [actual[0].cpu()[b : b + 1], actual[1].cpu()[b : b + 1]]
        native = [source[0].cpu()[b : b + 1], source[1].cpu()[b : b + 1]]
        empty = max(0, sq - length)
        assert (got[0][:, :empty] == 0).all() and got[1][:, :, :empty].isneginf().all()
        assert (native[0][:, :empty] == 0).all() and native[1][:, :, :empty].isposinf().all()
        native[1][:, :, :empty] = -torch.inf
        if length == 0:
            refs[1].fill_(-torch.inf)
            refs[3].fill_(-torch.inf)
        comparison = {}
        for i, name in enumerate(("out", "lse")):
            if torch.equal(got[i], native[i]):
                comparison[name] = assert_mla_source_close(
                    got[i], native[i], refs[i], refs[i + 2], name=name
                )
            else:
                # Native automatic FD can differ in arithmetic order. Its
                # dispatch is retained below, with independent math checks.
                assert_fa_close(got[i], refs[i], refs[i + 2], name=name)
                assert_fa_close(native[i], refs[i], refs[i + 2], name="native-" + name)
                comparison[name] = dict(source="different", math="passed")
        reports.append(comparison)
    identity = {
        name: dict(
            shape=list(t.shape),
            stride=list(t.stride()),
            offset=t.storage_offset(),
            sha256=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        )
        for name, t in cpu.items()
    }
    return dict(
        shape=shape, scale=scale, lengths=lengths, causal=True, inputs=identity, comparisons=reports
    )


def profile(
    directory,
    vendor,
    dtype_name="fp16",
    metadata_reuse=False,
    cases=CASES,
    kv_lengths=None,
    stride_profiles=None,
    dispatch_only=False,
    causal=False,
):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    timed_cpu_threads = torch.get_num_threads()
    reference_cpu_threads = int(os.environ.get("MLA_FORWARD_REFERENCE_THREADS", timed_cpu_threads))
    assert reference_cpu_threads > 0
    vendor = check_vendor_configuration(vendor)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    inputs, native, lengths, metadata = [], [], [], []
    if kv_lengths is None:
        kv_lengths = [None] * len(cases)
    assert len(kv_lengths) == len(cases)
    assert metadata_reuse or not any(x is not None for x in kv_lengths)
    if stride_profiles is not None:
        assert len(stride_profiles) == len(cases)

    def generate(index):
        q, k = inputs[index]["q"], inputs[index]["k"]
        return api.get_scheduler_metadata(
            q.shape[0],
            q.shape[1],
            k.shape[1],
            q.shape[2],
            1,
            64,
            lengths[index],
            qkv_dtype=dtype,
            headdim_v=512,
            num_splits=1,
            causal=causal,
        )

    def cached(index, record):
        tensors = inputs[index]
        return api.flash_attn_with_kvcache(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            qv=tensors["qv"],
            cache_seqlens=lengths[index],
            scheduler_metadata=record,
            softmax_scale=cases[index][1],
            num_splits=1,
            causal=causal,
            return_softmax_lse=True,
        )

    accuracy = []
    for index, (shape, scale) in enumerate(cases):
        batch, sq, heads, sk = shape
        cpu = make_inputs(
            Case("public", "compressed", batch, sq, sk, heads, 1), dtype, seed=20260929
        )
        device = {name: tensor.npu() for name, tensor in cpu.items()}
        if stride_profiles is not None:
            from tests.test_flash_attn_npu_v3_mla_bsnd_stride import NAMES, make_view, strides_for

            profile_name = stride_profiles[index]
            for name, offset in zip(NAMES, (5, 11, 19, 29)):
                device[name], cpu[name], _ = make_view(
                    cpu[name],
                    strides_for(cpu[name].shape, profile_name, name),
                    offset,
                    overlap=profile_name in ("broadcast", "overlap"),
                )
        inputs.append(device)
        native.append(CannCall(device, scale=scale, kv_lengths=kv_lengths[index], causal=causal))
        if metadata_reuse:
            lengths.append(
                torch.tensor(
                    kv_lengths[index] if kv_lengths[index] is not None else [sk] * batch,
                    dtype=torch.int32,
                    device=device["q"].device,
                )
            )
            metadata.append(generate(len(inputs) - 1))
            actual = cached(len(inputs) - 1, metadata[-1])
        else:
            actual = api.flash_attn_func(
                **device, softmax_scale=scale, num_splits=1, return_attn_probs=True, causal=causal
            )
        if dispatch_only:
            torch.npu.synchronize()
            print(f"MLA_STRIDE_NATIVE_BEGIN {dtype_name} {index}", flush=True)
        source = native[-1]()
        if dispatch_only:
            torch.npu.synchronize()
            print(f"MLA_STRIDE_NATIVE_END {dtype_name} {index}", flush=True)
        if causal:
            accuracy.append(
                check_causal_accuracy(actual, source, cpu, shape, scale, kv_lengths[index])
            )
            continue
        # The long-KV CPU oracle is outside every timed region. Restore the
        # caller's thread count even on failure, before any warmup or sampling.
        try:
            torch.set_num_threads(reference_cpu_threads)
            if kv_lengths[index] is None:
                refs = mla_reference_pair(**cpu, softmax_scale=scale)
            else:
                from tests.test_flash_attn_npu_v3_mla_bsnd_attention import references

                refs = references(
                    dict(shape=shape, logical_lengths=kv_lengths[index], inputs=cpu, scale=scale)
                )
        finally:
            torch.set_num_threads(timed_cpu_threads)
        for i, name in enumerate(("out", "lse")):
            nonempty = (
                torch.tensor(kv_lengths[index] if kv_lengths[index] is not None else [sk] * batch)
                > 0
            )
            if sk <= 512:
                torch.testing.assert_close(
                    actual[i].cpu()[nonempty], source[i].cpu()[nonempty], rtol=0, atol=0
                )
            assert_fa_close(actual[i].cpu(), refs[i], refs[i + 2], name=name)
            if nonempty.any():
                assert_fa_close(
                    source[i].cpu()[nonempty], refs[i][nonempty], refs[i + 2][nonempty], name=name
                )
            if i == 0:
                assert (source[i].cpu()[~nonempty] == 0).all()
            else:
                assert source[i].cpu()[~nonempty].isposinf().all()
        accuracy.append(
            dict(
                shape=shape,
                scale=scale,
                lengths=kv_lengths[index],
                source_equal=all(torch.equal(actual[i].cpu(), source[i].cpu()) for i in range(2)),
                fa3_max_error=[
                    float((actual[i].cpu() - refs[i]).abs().nan_to_num().max()) for i in range(2)
                ],
                cann_max_error=[
                    float((source[i].cpu()[nonempty] - refs[i][nonempty]).abs().max())
                    if nonempty.any()
                    else 0
                    for i in range(2)
                ],
            )
        )
    (directory / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    source_root = Path(
        os.environ.get(
            "MLA_CANN_SOURCE_ROOT", Path(__file__).resolve().parents[2] / "ops-transformer"
        )
    )
    env = environment(source_root, vendor)
    extension = Path(api.flash_attn_npu_3.__file__).resolve()
    env["fa3_extension"] = dict(
        path=str(extension), sha256=hashlib.sha256(extension.read_bytes()).hexdigest()
    )
    (directory / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    if dispatch_only:
        return

    dispatch = None
    if causal:
        from tests.mla_native_dispatch import sha256

        dispatch = json.loads(Path(os.environ["MLA_FORWARD_NATIVE_DISPATCH"]).read_text())
        assert dispatch["dtype"] == dtype_name and len(dispatch["cases"]) == len(cases)
        assert all(sha256(p) == digest for p, digest in dispatch["artifacts"].items())
        identity = [{k: v for k, v in item.items() if k != "comparisons"} for item in accuracy]
        assert json.loads(json.dumps(identity)) == dispatch["inputs"], (
            "Native dispatch inputs differ from timing"
        )
        assert os.environ.get("ASCEND_GLOBAL_LOG_LEVEL", "3") == "3", (
            "Detailed logging cannot be timed"
        )

    def snapshot(label):
        (directory / label).write_text(subprocess.check_output(["npu-smi", "info"], text=True))

    def call(index, name):
        if name == "native_cann":
            return native[index]()
        if name in ("func", "func_4"):
            return [
                api.flash_attn_func(
                    **inputs[index],
                    softmax_scale=cases[index][1],
                    num_splits=1,
                    return_attn_probs=True,
                    causal=True,
                )
                for _ in range(4 if name == "func_4" else 1)
            ]
        if metadata_reuse:
            if name == "generate":
                return generate(index)
            if name == "native_cann_4":
                return [native[index]() for _ in range(4)]
            if name == "implicit":
                return cached(index, None)
            if name == "implicit_4":
                return [cached(index, None) for _ in range(4)]
            if name == "reuse_4":
                return [cached(index, metadata[index]) for _ in range(4)]
            if name in ("generate_1", "generate_4"):
                record = generate(index)
                return [cached(index, record) for _ in range(1 if name == "generate_1" else 4)]
            return cached(index, metadata[index])
        return api.flash_attn_func(
            **inputs[index],
            softmax_scale=cases[index][1],
            num_splits=1,
            return_attn_probs=True,
            causal=causal,
        )

    names = ("fa3", "native_cann")
    wall_names = (
        names + ("generate", "implicit", "generate_1", "generate_4", "native_cann_4")
        if metadata_reuse
        else names
    )
    if causal and metadata_reuse:
        wall_names += ("reuse_4", "implicit_4")

    def case_modes(index):
        return (
            wall_names + ("func", "func_4") if causal and kv_lengths[index] is None else wall_names
        )

    # Causal references use the existing bounded sample pool before this point.
    # Pin only the timed section; parent orchestration preserves all CPU cores
    # for untimed reference generation and cache verification.
    if causal:
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    snapshot("npu-before.log")
    assert torch.get_num_threads() == timed_cpu_threads
    for index in range(len(cases)):
        for _ in range(5):
            for name in case_modes(index):
                result = call(index, name)
                torch.npu.synchronize()
    wall = []
    for index in range(len(cases)):
        for r in range(5):
            for i in range(20):
                for name in case_modes(index) if r % 2 == 0 else case_modes(index)[::-1]:
                    start = time.perf_counter_ns()
                    result = call(index, name)
                    torch.npu.synchronize()
                    wall.append(
                        dict(
                            case=index,
                            name=name,
                            round=r,
                            iteration=i,
                            e2e_us=(time.perf_counter_ns() - start) / 1000,
                        )
                    )
    (directory / "wall-samples.json").write_text(json.dumps(wall, indent=2) + "\n")

    labels = []
    profiler = torch_npu.profiler
    trace = directory / "profiler"
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
        result = call(0, "fa3")
        torch.npu.synchronize()
        prof.step()
        for index in range(len(cases)):
            for r in range(5):
                for i in range(10):
                    for name in names if r % 2 == 0 else names[::-1]:
                        result = call(index, name)
                        torch.npu.synchronize()
                        labels.append(dict(case=index, name=name, round=r, iteration=i))
                        prof.step()
    del result
    (directory / "profile-labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    csvs, raw_csvs = list(trace.rglob("kernel_details.csv")), list(trace.rglob("op_summary_*.csv"))
    assert len(csvs) == len(raw_csvs) == 1
    with csvs[0].open() as f:
        rows = sorted(csv.DictReader(f), key=lambda row: Decimal(row["Start Time(us)"].strip()))
    with raw_csvs[0].open() as f:
        raw = sorted(csv.DictReader(f), key=lambda row: Decimal(row["Task Start Time(us)"].strip()))
    assert len(rows) == len(raw), (len(rows), len(raw))
    for row, source in zip(rows, raw):
        assert row["Name"] == source["Op Name"] and row["Device_id"] == source["Device_id"]
        assert Decimal(row["Start Time(us)"].strip()) == Decimal(
            source["Task Start Time(us)"].strip()
        )
        assert float(row["Duration(us)"]) == float(source["Task Duration(us)"])

    # Some profiler versions include AICPU tasks here, others export only
    # attention kernels. Never fold producer duration into the attention
    # comparison, and explicitly report whether it was captured.
    def is_metadata(row):
        return any(
            name in row["Name"]
            for name in (
                "ComputeMlaDenseMetadata",
                "ComputeMlaBsndKvLengthMetadata",
                "ComputeMlaCausalBsndMetadata",
            )
        )

    metadata_rows = [row for row in rows if is_metadata(row)]
    if metadata_reuse:
        assert not metadata_rows, "explicit reuse unexpectedly regenerated metadata"
    rows = [row for row in rows if not is_metadata(row)]
    conversion_rows = []
    if stride_profiles is not None:
        # Native FIA materializes noncontiguous arguments through its copy
        # operators. Keep all device tasks in the report/E2E; FA3 must have none.
        conversion_rows = [
            row
            for row in rows
            if row["Name"].startswith("aclnnFusedInferAttentionScore")
            and row["Name"].endswith(("_Slice", "_AsStrided", "_BroadcastTo"))
        ]
        rows = [row for row in rows if row not in conversion_rows]
    assert len(rows) == len(labels), (len(rows), len(labels))
    fa3_count = sum(label["name"] == "fa3" for label in labels)
    assert len(metadata_rows) in (0, fa3_count), (len(metadata_rows), fa3_count)
    producers = iter(metadata_rows)
    previous_end = Decimal(0)
    conversion_index = 0
    for row, label in zip(rows, labels):
        expected_name = (
            f"mla_{'bsnd_used' if causal else 'dense'}_{dtype_name}"
            if label["name"] == "fa3"
            else "FusedInferAttentionScore"
        )
        assert expected_name in row["Name"]
        label.update(kernel_us=float(row["Duration(us)"]), kernel_name=row["Name"])
        if stride_profiles is not None:
            copies = []
            begin = Decimal(row["Start Time(us)"].strip())
            while conversion_index < len(conversion_rows):
                copy = conversion_rows[conversion_index]
                copy_begin = Decimal(copy["Start Time(us)"].strip())
                if copy_begin >= begin:
                    break
                assert label["name"] == "native_cann" and previous_end <= copy_begin
                assert copy_begin + Decimal(copy["Duration(us)"].strip()) <= begin
                copies.append(copy)
                conversion_index += 1
            native_call = native[label["case"]]
            expected_copies = (
                sum(
                    t is not None and not t.is_contiguous()
                    for t in (
                        native_call.query,
                        native_call.key,
                        native_call.value,
                        native_call.kwargs["query_rope"],
                        native_call.kwargs["key_rope"],
                    )
                )
                if label["name"] == "native_cann"
                else 0
            )
            assert len(copies) == expected_copies, (label, copies, expected_copies)
            label["conversion_us"] = sum(float(c["Duration(us)"]) for c in copies)
            label["conversion_names"] = [c["Name"] for c in copies]
        if metadata_rows and label["name"] == "fa3":
            producer = next(producers)
            begin = Decimal(producer["Start Time(us)"].strip())
            end = begin + Decimal(producer["Duration(us)"].strip())
            assert previous_end <= begin < end <= Decimal(row["Start Time(us)"].strip())
            label["metadata_kernel_us"] = float(producer["Duration(us)"])
        previous_end = Decimal(row["Start Time(us)"].strip()) + Decimal(row["Duration(us)"].strip())
    assert conversion_index == len(conversion_rows)
    results = []
    for index, (shape, scale) in enumerate(cases):
        item = dict(
            shape=shape,
            scale=scale,
            lengths=kv_lengths[index],
            native_call=native[index].description(),
            stride_profile=None if stride_profiles is None else stride_profiles[index],
        )
        for name in case_modes(index):
            kernels = [x for x in labels if x["case"] == index and x["name"] == name]
            walls = [x for x in wall if x["case"] == index and x["name"] == name]
            item[name] = dict(
                median_e2e_us=statistics.median(x["e2e_us"] for x in walls),
                round_mean_e2e_us=[
                    statistics.mean(x["e2e_us"] for x in walls if x["round"] == r) for r in range(5)
                ],
            )
            if kernels:
                item[name].update(
                    median_kernel_us=statistics.median(x["kernel_us"] for x in kernels),
                    round_mean_kernel_us=[
                        statistics.mean(x["kernel_us"] for x in kernels if x["round"] == r)
                        for r in range(5)
                    ],
                )
                if stride_profiles is not None:
                    item[name]["median_conversion_us"] = statistics.median(
                        x["conversion_us"] for x in kernels
                    )
        item["e2e_ratio"] = item["fa3"]["median_e2e_us"] / item["native_cann"]["median_e2e_us"]
        if metadata_reuse:
            item["generate_1_ratio"] = (
                item["generate_1"]["median_e2e_us"] / item["native_cann"]["median_e2e_us"]
            )
            item["generate_4_ratio"] = (
                item["generate_4"]["median_e2e_us"] / item["native_cann_4"]["median_e2e_us"]
            )
        if causal:
            item.update(
                {
                    name: dispatch["cases"][index][name]
                    for name in (
                        "native_tiling_key",
                        "native_template",
                        "native_fd",
                        "e2e_applicable",
                    )
                }
            )
            if metadata_reuse:
                item["reuse_4_ratio"] = (
                    item["reuse_4"]["median_e2e_us"] / item["native_cann_4"]["median_e2e_us"]
                )
                item["implicit_ratio"] = (
                    item["implicit"]["median_e2e_us"] / item["native_cann"]["median_e2e_us"]
                )
                item["implicit_4_ratio"] = (
                    item["implicit_4"]["median_e2e_us"] / item["native_cann_4"]["median_e2e_us"]
                )
                if kv_lengths[index] is None:
                    item["func_ratio"] = (
                        item["func"]["median_e2e_us"] / item["native_cann"]["median_e2e_us"]
                    )
                    item["func_4_ratio"] = (
                        item["func_4"]["median_e2e_us"] / item["native_cann_4"]["median_e2e_us"]
                    )
        results.append(item)
    snapshot("npu-after.log")
    report = dict(
        dtype=dtype_name,
        causal=causal,
        stride_boundary=(
            None
            if stride_profiles is None
            else "Identical physical strided inputs; one-time view creation and H2D excluded symmetrically. Native CANN Slice/AsStrided/BroadcastTo conversions remain inside every timed call; FA3 consumes original views."
        ),
        scope=(
            "Public read-only dense kvcache metadata reuse vs fixed native CANN"
            if metadata_reuse
            else "Public flash_attn_func with implicit AICPU metadata generation vs fixed native CANN; diagnostics, not explicit metadata reuse acceptance"
        ),
        metadata_lifecycle=(
            dict(
                reuse_forward="precomputed once before measurement",
                complete_segments=[1, 4],
                complete_segment="one public metadata generation plus R forwards; CANN performs the same R forwards; R fixed before collection",
            )
            if metadata_reuse
            else "implicit generation for each forward"
        ),
        pid=os.getpid(),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        seed=20260929,
        warmup=5,
        rounds=5,
        wall_iterations=20,
        profiler_iterations=10,
        reference_cpu_threads=reference_cpu_threads,
        timed_cpu_threads=timed_cpu_threads,
        timing="public call through device completion, including validation, scheduling, output/workspace allocation and launch; input H2D excluded on both paths",
        kernel_csv=str(csvs[0]),
        raw_kernel_csv=str(raw_csvs[0]),
        metadata_kernel_rows=len(metadata_rows),
        results=results,
        kernel_samples=labels,
    )
    (directory / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return report


if __name__ == "__main__":
    cases = (
        CASES
        + (S1G_CASES if os.environ.get("MLA_FORWARD_S1G_MULTIBLOCK") == "1" else [])
        + (KV_CASES if os.environ.get("MLA_FORWARD_KV_MULTIBLOCK") == "1" else [])
        + (LONG_KV_CASES if os.environ.get("MLA_FORWARD_LONG_KV") == "1" else [])
    )
    lengths = [None] * len(cases)
    strides = None
    if os.environ.get("MLA_FORWARD_BSND_LENGTHS") == "1":
        cases += [(shape, scale) for shape, scale, _ in BSND_CASES]
        lengths += [values for _, _, values in BSND_CASES]
    if os.environ.get("MLA_FORWARD_BSND_STRIDES") == "1":
        cases = [(shape, scale) for shape, scale, _, _ in STRIDE_CASES]
        lengths = [values for _, _, values, _ in STRIDE_CASES]
        strides = [profile_name for _, _, _, profile_name in STRIDE_CASES]
    causal = os.environ.get("MLA_FORWARD_CAUSAL") == "1"
    if causal:
        from tests.test_flash_attn_npu_v3_mla_causal_public import (
            NATIVE_PREFLIGHT_CASES,
            PERFORMANCE_CASES,
        )

        specs = PERFORMANCE_CASES
        if os.environ.get("MLA_FORWARD_CAUSAL_PREFLIGHT") == "1":
            specs = NATIVE_PREFLIGHT_CASES
        cases = [(s["shape"], s.get("scale")) for s in specs]
        lengths = [s.get("lengths") for s in specs]
        strides = [s.get("stride", "contiguous") for s in specs]
    profile(
        os.environ["MLA_FORWARD_ARTIFACT_DIR"],
        os.environ["MLA_CANN_VENDOR"],
        os.environ.get("MLA_FORWARD_DTYPE", "fp16"),
        os.environ.get("MLA_FORWARD_METADATA_REUSE") == "1",
        cases,
        lengths,
        strides,
        os.environ.get("MLA_FORWARD_DISPATCH_ONLY") == "1",
        causal=causal,
    )
