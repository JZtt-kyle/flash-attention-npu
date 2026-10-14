"""One release Trap through the actual PA attention probe per process."""

import json
import os
from pathlib import Path
import resource
import sys
import time

import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_attention import (
    load_attention,
    configure_producer,
    fault_case,
    launch,
    check_rejected,
)


def run(attention, producer_manifest, fault, output, layout="0"):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    output = Path(output)
    probes = load_attention(attention)
    manifest = json.loads(Path(producer_manifest).read_text())
    assert sha256(Path(manifest["library"])) == manifest["library_sha256"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    with load_probe(manifest["library"], output / "producer") as producer:
        configure_producer(producer)
        case, options, status = fault_case(probes, producer, fault, layout=int(layout))
        assert launch(probes, case, checked=False, diagnostic=True, **options) == 0
        check_rejected(case, status)
        start = time.monotonic()
        try:
            with forbid_host_readback():
                assert launch(probes, case, **options) == 0
            if fault == "stale_q":
                event = torch.npu.Event()
                event.record()
                event.synchronize()
            elif fault == "stale_k":
                torch.npu.synchronize()
            else:
                torch.npu.current_stream().synchronize()
        except RuntimeError as error:
            report = dict(
                fault=fault,
                layout=int(layout),
                expected_status=status,
                error=str(error),
                seconds=time.monotonic() - start,
                dtype=str(probes[0]["dtype"]),
                device=torch.npu.get_device_name(0),
                visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            )
            (output / "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        raise AssertionError("bad v6 input did not Trap before PA attention")


if __name__ == "__main__":
    run(*sys.argv[1:])
