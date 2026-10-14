"""One release-mode v5 Trap per process; no poisoned stream is reused."""

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
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_aicpu import UsedArgs
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_device import (
    load_consumer,
    fault_case,
    launch,
    check,
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
        producer["lib"].launch_used_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(UsedArgs)]
        producer["lib"].launch_used_metadata.restype = ct.c_int
        producer["lib"].used_metadata_host.argtypes = [ct.POINTER(UsedArgs), ct.c_void_p]
        producer["lib"].used_metadata_host.restype = ct.c_uint32
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            case, options, expected = fault_case(consumer, producer, fault)
            launch(consumer, case, checked=False, **options)
            check(case, expected)
            stage = "launch"
            start = time.monotonic()
            try:
                with forbid_host_readback():
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
            raise AssertionError("invalid v5 record did not Trap")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], int(sys.argv[3]), *sys.argv[4:])
