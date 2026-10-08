"""One v2 release-mode Trap per process; never reuse a poisoned device stream."""

import ctypes as ct
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch
import torch_npu  # noqa: F401

from tests.mla_metadata_build import ROOT, sha256
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata import Args
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata_aicpu import enqueue
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata_device import (
    load_consumer,
    make_case,
    launch,
    check,
)


def inject(producer, case, fault):
    mutations = dict(header=(1, 1), snapshot=(401, -1), workspace=(386, -1))
    if fault in mutations:
        word, value = mutations[fault]
        case["metadata"][word] = value
        case["expected"][word] = value & 0xFFFFFFFF
    if fault in ("stale", "empty_stale"):
        case["lengths"][-1] = 1
    if fault in ("suffix", "partial_suffix"):
        for word, value in (
            (20, 1),
            (23, 1 if fault == "suffix" else 3),
            (49, 0 if fault == "suffix" else 1),
            (75, 0),
        ):
            case["metadata"][word] = value
            case["expected"][word] = value
    if fault == "generation":
        case["lengths"][0] = -1
        enqueue(producer, case["args"], case["metadata_storage"], case["lengths"])
        case["expected"][5:7] = [2, 1]
        case["expected"][8:] = [0] * (len(case["expected"]) - 8)
    return {
        "words": dict(words=401),
        "missing": dict(length_address=0, count=0),
        "count": dict(count=1),
    }.get(fault, {})


def run(manifest_path, producer_path, dtype, fault, sync, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    directory = Path(directory)
    manifest = json.loads(Path(manifest_path).read_text())
    producer_manifest = json.loads(Path(producer_path).read_text())
    for path, digest in producer_manifest["sources"].items():
        assert sha256(ROOT / path) == digest, path
    assert sha256(producer_manifest["library"]) == producer_manifest["library_sha256"]
    consumer = dict(lib=load_consumer(manifest, dtype), dtype=dtype)
    with load_probe(producer_manifest["library"], directory / "producer") as producer:
        producer["lib"].launch_bsnd_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(Args)]
        producer["lib"].launch_bsnd_metadata.restype = ct.c_int
        producer["lib"].bsnd_metadata_host.argtypes = [ct.POINTER(Args), ct.c_void_p]
        producer["lib"].bsnd_metadata_host.restype = ct.c_uint32
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            lengths = [1, 0, 7, 0, 0]
            shape = (5, 0 if fault == "empty_stale" else 3, 171, 513)
            case = make_case(consumer, producer, lengths, shape=shape)
            options = inject(producer, case, fault)
            expected = {
                "header": 2,
                "words": 1,
                "suffix": 8,
                "partial_suffix": 8,
                "workspace": 9,
                "generation": 3,
            }.get(fault, 11)
            launch(consumer, case, checked=False, **options)
            check(case, expected)  # Diagnostic rejection must not write outputs.
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
            raise AssertionError("invalid v2 record did not Trap")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), *sys.argv[4:])
