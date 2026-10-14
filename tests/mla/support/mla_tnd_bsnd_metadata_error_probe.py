"""One release-mode v4 Trap per process; no poisoned stream is reused."""

import ctypes as ct
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_metadata_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_aicpu import (
    MixedArgs,
    enqueue,
)
from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_device import (
    load_consumer,
    make_case,
    launch,
    check,
)


def inject(producer, case, fault):
    mutations = dict(
        v1=(1, 1),
        v2=(2, 2),
        v3=(1, 3),
        span=(406, -1),
        wide_start=(408, 1),
        capacity=(405, 1),
        workspace=(386, -1),
    )
    if fault in mutations:
        word, value = mutations[fault]
        case["metadata"][word] = value
        case["expected"][word] = value & 0xFFFFFFFF
    if fault in ("stale_q", "empty_stale", "empty_batch"):
        case["query"][-1] = 1
    if fault == "stale_k":
        case["kv"][1] = 1
    if fault in ("suffix", "variable_m"):
        for word, value in (
            (20, 1),
            (23, 1 if fault == "suffix" else 2),
            (49, 0 if fault == "suffix" else 1),
            (75, 0),
        ):
            case["metadata"][word] = value
            case["expected"][word] = value
    if fault == "generation":
        case["query"][0] = -1
        enqueue(producer, case["args"], case["metadata_storage"], case["query"], case["kv"])
        case["expected"][5:7] = [2, 1]
        case["expected"][8:] = [0] * (len(case["expected"]) - 8)
    return {"words": dict(words=401), "missing_q": dict(qaddress=0), "count_k": dict(kcount=1)}.get(
        fault, {}
    )


def run(manifest_path, producer_path, dtype, fault, sync, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    directory = Path(directory)
    manifest = json.loads(Path(manifest_path).read_text())
    pm = json.loads(Path(producer_path).read_text())
    for path, digest in pm["sources"].items():
        assert sha256(ROOT / path) == digest, path
    assert sha256(pm["library"]) == pm["library_sha256"]
    consumer = dict(lib=load_consumer(manifest, dtype), dtype=dtype)
    with load_probe(pm["library"], directory / "producer") as producer:
        producer["lib"].launch_mixed_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(MixedArgs)]
        producer["lib"].launch_mixed_metadata.restype = ct.c_int
        producer["lib"].mixed_metadata_host.argtypes = [ct.POINTER(MixedArgs), ct.c_void_p]
        producer["lib"].mixed_metadata_host.restype = ct.c_uint32
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            q, k = [3, 0, 1, 7, 0], [513, 7, 1, 0, 16384]
            if fault == "empty_stale":
                q = [0] * 5
            if fault == "empty_batch":
                q, k = [], []
            case = make_case(consumer, producer, q, k, heads=171, capacity=16384)
            options = inject(producer, case, fault)
            expected = dict(
                v1=2,
                v2=2,
                v3=2,
                wide_start=11,
                capacity=5,
                stale_k=11,
                count_k=11,
                words=1,
                suffix=8,
                variable_m=8,
                workspace=9,
                generation=3,
            ).get(fault, 12)
            launch(consumer, case, checked=False, **options)
            check(case, expected)
            stage = "launch"
            start = time.monotonic()
            try:
                with forbid_host_readback():
                    options = inject(producer, case, fault)
                    launch(consumer, case, **options)
                stage = sync
                if sync == "device":
                    torch.npu.synchronize()
                elif sync == "stream":
                    stream.synchronize()
                else:
                    done = torch.npu.Event()
                    done.record(stream)
                    done.synchronize()
            except RuntimeError as error:
                report = dict(
                    fault=fault,
                    expected_status=expected,
                    dtype=dtype,
                    pid=os.getpid(),
                    device=torch.npu.get_device_name(0),
                    visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                    stage=stage,
                    elapsed_seconds=time.monotonic() - start,
                    error=str(error),
                )
                (directory / "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
            raise AssertionError("invalid v4 record did not Trap")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), *sys.argv[4:])
