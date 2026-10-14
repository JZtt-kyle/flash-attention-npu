"""Release Trap through the actual mixed-layout attention launcher, one process per fault."""

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
from tests.mla.support.mla_tnd_bsnd_metadata_error_probe import inject
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_attention import (
    configure_producer,
    launch,
    load_attention,
    make_case,
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
        q, k = [3, 0, 1, 7, 0], [513, 7, 1, 0, 16384]
        if fault == "empty_stale":
            q = [0] * 5
        if fault == "empty_batch":
            q, k = [], []
        if fault == "empty_kv":
            k = [0] * 5
        case = make_case(probes, producer, (q, k, 0 if fault == "empty_kv" else 16384, 171, None))
        options = inject(producer, case, "stale_k" if fault == "empty_kv" else fault)
        expected = dict(
            v1=2,
            v2=2,
            v3=2,
            words=1,
            wide_start=11,
            capacity=5,
            stale_k=11,
            count_k=11,
            empty_kv=11,
            suffix=8,
            variable_m=8,
            workspace=9,
            generation=3,
        ).get(fault, 12)
        launch(probes, case, checked=False, **options)
        assert (case["status"][::16].cpu() == expected).all()
        for tensor in case["outputs"]["migrated"]:
            assert (tensor.cpu() == 12345).all()
        assert (case["ws_storage"]["migrated"].cpu() == 205).all()
        start = time.monotonic()
        try:
            with forbid_host_readback():
                launch(probes, case, **options)
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
        raise AssertionError("invalid TND metadata did not Trap before attention")


if __name__ == "__main__":
    run(*sys.argv[1:])
