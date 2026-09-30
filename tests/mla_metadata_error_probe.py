"""One real device failure per process; a trapped stream cannot be reused."""

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from tests.mla_dense_build import ROOT, sha256
from tests.test_flash_attn_npu_v3_mla_metadata import Args, Shape, enqueue, load_probe
from tests.test_flash_attn_npu_v3_mla_metadata_dense import (
    WORDS,
    assert_status,
    check_metadata_case,
    configure,
    launch_consumer,
    make_metadata_case,
)


# One representative of every decoder status, plus producer failures and the
# size/alignment checks that must precede any GM read. Exhaustive mutations
# remain in the diagnostic consumer tests, where outputs can still be copied.
FAULTS = {
    "null": 1,
    "unaligned": 1,
    "truncated": 1,
    "magic": 2,
    "version": 2,
    "schedule_failure": 3,
    "length_count": 3,
    "length_negative": 3,
    "length_zero": 3,
    "length_short": 3,
    "length_over_capacity": 3,
    "dtype": 4,
    "shape": 5,
    "platform": 6,
    "tiling": 7,
    "endpoints": 8,
    "workspace_high_word": 9,
    "local_memory": 10,
}


@contextmanager
def forbid_host_readback():
    """Fail on accidental Python scalar reads or host waits in the enqueue path."""
    with ExitStack() as stack:
        for obj, names in (
            (torch.Tensor, ("cpu", "item", "tolist", "numpy")),
            (torch.npu, ("synchronize",)),
            (torch.npu.Stream, ("synchronize",)),
            (torch.npu.Event, ("synchronize",)),
        ):
            for name in names:
                stack.enter_context(
                    patch.object(obj, name, side_effect=AssertionError(f"host read/wait: {name}"))
                )
        yield


def inject(producer, case, fault):
    args = Args(0, WORDS, case["context"].dtype, Shape(*case["shape"]), producer["platform"])
    if fault == "schedule_failure":
        args.shape.sk = 1025
    lengths = case["cache_seqlens"]
    if fault == "length_count":
        lengths = lengths[:1]
    values = dict(length_negative=-1, length_zero=0, length_short=256, length_over_capacity=258)
    if fault in values:
        lengths[1] = values[fault]
    enqueue(producer, args, case["metadata_storage"], cache_seqlens=lengths)
    mutations = dict(
        magic=(0, 0),
        version=(1, 2),
        dtype=(4, 3),
        shape=(8, 0),
        platform=(19, 1),
        tiling=(15, 0),
        endpoints=(23, 0),
        workspace_high_word=(368, -1),
        local_memory=(391, 0),
    )
    if fault in mutations:
        word, value = mutations[fault]
        case["metadata"][word] = value
    if fault == "null":
        return dict(address=0)
    if fault == "unaligned":
        return dict(address=case["metadata"].data_ptr() + 2)
    return dict(words=WORDS - 1) if fault == "truncated" else {}


def run(manifest_path, producer_path, fault, sync, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    directory = Path(directory)
    manifest = json.loads(Path(manifest_path).read_text())
    producer_manifest = json.loads(Path(producer_path).read_text())
    for record in (manifest, producer_manifest):
        for path, digest in record["sources"].items():
            assert sha256(ROOT / path) == digest, path
    for row in manifest["libraries"].values():
        assert sha256(Path(row["path"])) == row["sha256"]
    if "production_extension" in manifest:
        # The C bridge links the real extension, so an old/rebuilt extension
        # invalidates this run even if the bridge itself has not changed.
        row = manifest["production_extension"]
        assert sha256(Path(row["path"])) == row["sha256"]
    assert sha256(Path(producer_manifest["library"])) == producer_manifest["library_sha256"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[manifest["dtype"]]
    probes = configure(
        {name: row["path"] for name, row in manifest["libraries"].items()}, directory, dtype
    )
    with load_probe(producer_manifest["library"], directory / "producer") as producer:
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            case = make_metadata_case(probes, producer, (2, 1, 8, 257))
            check_metadata_case(probes, case, repeats=1)
            options = inject(producer, case, fault)
            launch_consumer(probes, case, checked=False, **options)
            assert_status(case, FAULTS[fault])
            # The generation -> consumption chain below contains no length or
            # metadata D2H: inject again after the independent diagnostic
            # check. Only the final ordinary synchronization may fail.
            stage = "launch"
            start = time.monotonic()
            try:
                with forbid_host_readback():
                    options = inject(producer, case, fault)
                    launch_consumer(probes, case, **options)
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
                    expected_decoder_status=FAULTS[fault],
                    dtype=manifest["dtype"],
                    pid=os.getpid(),
                    device=torch.npu.get_device_name(0),
                    visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                    stage=stage,
                    elapsed_seconds=time.monotonic() - start,
                    error=str(error),
                    library=manifest["libraries"]["metadata_checked"],
                    production_extension=manifest.get("production_extension"),
                )
                (directory / "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
                # Keep all tensors/streams alive until process exit. Their
                # destructors can enqueue allocator events on the trapped
                # context and abort before normal interpreter shutdown. The
                # parent owns and cleans this entire test process group.
                os._exit(0)
            raise AssertionError(f"{fault}: invalid metadata silently completed")


if __name__ == "__main__":
    run(*sys.argv[1:])
