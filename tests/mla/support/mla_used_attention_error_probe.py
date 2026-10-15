"""Release Trap through the actual used-length attention launcher, one process per fault."""

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
from tests.mla.support.components.metadata import load_probe
from tests.mla.support.components.used_attention import (
    configure_producer,
    launch,
    load_attention,
    fault_case,
    check_rejected,
)


def run(attention, producer_manifest, fault, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    directory = Path(directory)
    probes = load_attention(attention)
    manifest = json.loads(Path(producer_manifest).read_text())
    for path, digest in manifest["sources"].items():
        assert sha256(ROOT / path) == digest, path
    assert sha256(manifest["library"]) == manifest["library_sha256"]
    with load_probe(manifest["library"], directory / "producer") as producer:
        configure_producer(producer)
        case, options, expected = fault_case(probes, producer, fault)
        launch(probes, case, checked=False, **options)
        check_rejected(case, expected)
        start = time.monotonic()
        try:
            with forbid_host_readback():
                launch(probes, case, **options)
            if fault == "stale_uq":
                done = torch.npu.Event()
                done.record()
                done.synchronize()
            elif fault == "stale_uk":
                torch.npu.synchronize()
            else:
                torch.npu.current_stream().synchronize()
        except RuntimeError as error:
            report = dict(
                fault=fault,
                expected_status=expected,
                dtype=str(probes[0]["dtype"]),
                elapsed_seconds=time.monotonic() - start,
                error=str(error),
                device=torch.npu.get_device_name(0),
                visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            )
            (directory / "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        raise AssertionError("invalid v5 metadata did not Trap before attention")


if __name__ == "__main__":
    run(*sys.argv[1:])
