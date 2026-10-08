"""910B generation of real BSND length records; no attention consumer yet."""

import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla_metadata_build import ROOT, build_probe, sha256
from tests.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    length_artifacts as length_artifacts,
    length_oracle as length_oracle,
    length_case,
)
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata import (
    Args,
    MAGIC,
    PREFIX,
    SENTINEL,
    arguments,
    check_source,
    metadata_cases,
)
from tests.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.mla_metadata_error_probe import forbid_host_readback


@pytest.fixture(scope="module")
def bsnd_probe(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_BSND_METADATA_DEVICE_DIR", tmp_path_factory.mktemp("mla-bsnd-aicpu"))
    )
    library = os.environ.get("MLA_BSND_METADATA_LIBRARY")
    if library:
        manifest = json.loads((Path(library).parent / "build-manifest.json").read_text())
        assert sha256(library) == manifest["library_sha256"]
        for name, digest in manifest["sources"].items():
            assert sha256(ROOT / name) == digest, name
    else:
        library = build_probe(directory, bsnd=True)
    with load_probe(library, directory) as probe:
        assert probe["lib"].metadata_args_size() == ct.sizeof(Args)
        for name in ("launch_metadata", "launch_bsnd_metadata"):
            function = getattr(probe["lib"], name)
            function.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(Args)]
            function.restype = ct.c_int
        probe["lib"].bsnd_metadata_host.argtypes = [ct.POINTER(Args), ct.c_void_p]
        probe["lib"].bsnd_metadata_host.restype = ct.c_uint32
        yield probe


def host_record(probe, values, dtype):
    args = arguments(values, dtype)
    lengths = (ct.c_int32 * max(1, len(values[16:])))(*values[16:])
    args.cache_seqlens = ct.addressof(lengths) if values[13] else 0
    args.cache_seqlens_count = values[14]
    result = (ct.c_uint32 * args.words)()
    assert probe["lib"].bsnd_metadata_host(ct.byref(args), result) == 0
    assert list(result)[:5] == [MAGIC, 2, 2, args.words, dtype]
    return list(result)


def enqueue(probe, args, storage, lengths=None, worker=0, *, dense=False):
    output = storage[1:-1]
    assert output.numel() == args.words and output.is_contiguous()
    args.output = output.data_ptr()
    if lengths is not None:
        assert lengths.is_contiguous() and lengths.ndim == 1 and lengths.dtype == torch.int32
        assert lengths.device == output.device
        args.cache_seqlens = lengths.data_ptr()
        args.cache_seqlens_count = lengths.numel()
    stream = probe["workers"][worker]
    ready, done = probe["events"][2 * worker : 2 * worker + 2]
    launch = probe["lib"].launch_metadata if dense else probe["lib"].launch_bsnd_metadata
    assert (
        launch(
            torch.npu.current_stream().npu_stream, stream.npu_stream, ready, done, ct.byref(args)
        )
        == 0
    )
    output.record_stream(stream)
    if lengths is not None:
        lengths.record_stream(stream)


def device_lengths(values):
    if not values[13]:
        return None
    storage = torch.full((values[0] + 2,), SENTINEL, dtype=torch.int32, device="npu")
    storage[1:-1].copy_(torch.tensor(values[16:], dtype=torch.int32, device="npu"))
    return storage[1:-1]


def check_storage(storage, expected):
    words = [w & 0xFFFFFFFF for w in storage.cpu().tolist()]
    assert words[0] == words[-1] == SENTINEL
    assert words[1:-1] == expected
    return words[1:-1]


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("aic,ratio", [(None, 2), (1, 1), (15, 2), (26, 1)])
def test_aicpu_source_snapshots(bsnd_probe, length_oracle, dtype, aic, ratio):
    cases = metadata_cases(aic or bsnd_probe["platform"].aic, ratio)
    if aic is None:
        platform = bsnd_probe["platform"]
        assert list(platform.block_dims)[: platform.aic] == list(range(1, platform.aic + 1))
        for values in cases:
            values[6:11] = [
                getattr(platform.memory, name) for name in ("l1", "ub", "l0a", "l0b", "l0c")
            ]
            values[11] = platform.libapi
    expected = [host_record(bsnd_probe, c, dtype) for c in cases]
    pending = []
    for index, values in enumerate(cases):
        args = arguments(values, dtype)
        storage = torch.full((args.words + 2,), SENTINEL, dtype=torch.int32, device="npu")
        lengths = device_lengths(values)
        with forbid_host_readback():
            enqueue(bsnd_probe, args, storage, lengths, index % 2)
        pending.append(storage)
    rows = [check_storage(storage, host) for storage, host in zip(pending, expected)]
    check_source(
        rows, cases, length_oracle, bsnd_probe["directory"], f"device-{dtype}-{aic}-{ratio}"
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_device_empty_capacity_records(bsnd_probe, dtype):
    for b, sq, sk in itertools.product((0, 2), (0, 3), (0, 16384)):
        if b * sq * sk:
            continue
        for lengths in (None, [sk] * b, [0] * b):
            values = length_case(lengths, batch=b, sq=sq, sk=sk)
            args = arguments(values, dtype)
            storage = torch.full((args.words + 2,), SENTINEL, dtype=torch.int32, device="npu")
            enqueue(bsnd_probe, args, storage, device_lengths(values))
            row = check_storage(storage, host_record(bsnd_probe, values, dtype))
            assert row[5:8] == [0, 0, 0] and not any(row[23:PREFIX])


@pytest.mark.parametrize("dtype", (1, 2))
def test_device_failure_clear_and_recovery(bsnd_probe, dtype):
    values = length_case([0, 513], batch=2, sk=16384)
    args = arguments(values, dtype)
    lengths = device_lengths(values)
    storage = torch.full((args.words + 2,), SENTINEL, dtype=torch.int32, device="npu")
    good = host_record(bsnd_probe, values, dtype)
    for index in (0, 1):
        for value in (-(2**31), -1, 16385, 2**31 - 1):
            enqueue(bsnd_probe, args, storage, lengths)
            check_storage(storage, good)
            lengths[index] = value
            enqueue(bsnd_probe, args, storage, lengths)
            check_storage(
                storage, [MAGIC, 2, 2, args.words, dtype, 2, 1, 0] + [0] * (args.words - 8)
            )
            lengths[index] = values[16 + index]
    for address, count in (
        (0, 2),
        (lengths.data_ptr(), 1),
        (lengths.data_ptr(), 3),
        (lengths.data_ptr(), 2**64 - 1),
        (lengths.data_ptr() + 2, 2),
        (2**64 - 4, 2),
    ):
        args.cache_seqlens, args.cache_seqlens_count = address, count
        enqueue(bsnd_probe, args, storage)
        check_storage(storage, [MAGIC, 2, 2, args.words, dtype, 3, 0, 0] + [0] * (args.words - 8))
    for shape in (
        {"sq": -1},
        {"sk": 16385},
        {"heads": 0},
        {"aic": 0},
        {"ub": 0},
        {"libapi": 2**64 - 1},
    ):
        invalid = length_case([0, 513], **(dict(batch=2, sk=16384) | shape))
        bad = arguments(invalid, dtype)
        enqueue(bsnd_probe, bad, storage, lengths)
        check_storage(storage, host_record(bsnd_probe, invalid, dtype))
    for bad_dtype in (0, 3, 2**32 - 1):
        bad = arguments(values, bad_dtype)
        enqueue(bsnd_probe, bad, storage, lengths)
        check_storage(storage, host_record(bsnd_probe, values, bad_dtype))
    enqueue(bsnd_probe, args, storage, lengths)
    check_storage(storage, good)
    assert lengths.cpu().tolist() == values[16:]
    assert lengths._base.cpu().tolist() == [SENTINEL, *values[16:], SENTINEL]


@pytest.mark.parametrize("dtype", (1, 2))
def test_every_length_update_and_snapshot_ownership(bsnd_probe, dtype):
    batch = 2 * bsnd_probe["platform"].aic + 1
    values = length_case([8193] * batch, batch=batch, sq=3, heads=171, sk=16384)
    args = arguments(values, dtype)
    lengths = device_lengths(values)
    storage = torch.full((args.words + 2,), SENTINEL, dtype=torch.int32, device="npu")
    for index in range(batch):
        lengths[index] = index % 2
        values[16 + index] = index % 2
        with forbid_host_readback():
            enqueue(bsnd_probe, args, storage, lengths)
        check_storage(storage, host_record(bsnd_probe, values, dtype))
    snapshot = storage.clone()
    lengths.fill_(16384)
    assert torch.equal(snapshot.cpu(), storage.cpu())
    # Generation has no hidden ownership of the length tensor after its event.
    del lengths
    check_storage(storage, host_record(bsnd_probe, values, dtype))


def test_stream_ordering_and_allocator_lifetime(bsnd_probe):
    pending = []
    for worker, stream in enumerate([torch.npu.Stream(), torch.npu.Stream()]):
        with torch.npu.stream(stream):
            for iteration in range(24):
                values = length_case(
                    [0, 1, 513, 0] if iteration % 2 else [8193] * 4,
                    batch=4,
                    sq=3,
                    heads=171,
                    sk=16384,
                )
                args = arguments(values, worker + 1)
                lengths = device_lengths(values)
                storage = torch.full((args.words + 2,), SENTINEL, dtype=torch.int32, device="npu")
                with forbid_host_readback():
                    enqueue(bsnd_probe, args, storage, lengths, worker)
                    copied = storage.clone()
                pending.append((copied, host_record(bsnd_probe, values, worker + 1)))
                del args, lengths, storage
                torch.empty((32768,), dtype=torch.int32, device="npu").fill_(SENTINEL)
    for storage, expected in pending:
        check_storage(storage, expected)


@pytest.mark.parametrize("dtype,ratio", itertools.product((1, 2), (1, 2)))
def test_zero_suffix_preserves_source_endpoint(bsnd_probe, length_oracle, dtype, ratio):
    values = length_case(
        [0, 1, 511, 512, 513] * 3 + [0], batch=16, sq=1, heads=1, sk=513, aic=15, ratio=ratio
    )
    args = arguments(values, dtype)
    storage = torch.full((args.words + 2,), SENTINEL, dtype=torch.int32, device="npu")
    enqueue(bsnd_probe, args, storage, device_lengths(values))
    row = check_storage(storage, host_record(bsnd_probe, values, dtype))
    check_source([row], [values], length_oracle, bsnd_probe["directory"], f"suffix-{dtype}-{ratio}")
    assert row[23 + row[20] - 1] == 15 and row[-1] == 0


@pytest.mark.parametrize("dtype", (1, 2))
def test_generation_diagnostics(bsnd_probe, dtype):
    # Generator-only diagnostic; v1/full and v2/full do identical scheduling.
    # v2/short has no v1 equivalent and is reported without a speedup claim.
    records = []
    for b, sq, h, sk in (
        (2, 1, 8, 257),
        (1, 7, 16, 65),
        (2, 2, 128, 511),
        (21, 171, 3, 8193),
        (41, 3, 171, 16384),
    ):
        variants = {}
        for name, lens, dense in [
            ("v1-full", [sk] * b, True),
            ("v2-full", [sk] * b, False),
            ("v2-short", [0 if i % 2 else max(1, sk // 2) for i in range(b)], False),
        ]:
            values = length_case(lens, batch=b, sq=sq, heads=h, sk=sk)
            args = arguments(values, dtype)
            if dense:
                args.words = PREFIX
            lengths = device_lengths(values)
            storage = torch.full((args.words + 2,), SENTINEL, dtype=torch.int32, device="npu")
            variants[name] = (args, storage, lengths, dense)
        samples = {name: [] for name in variants}
        for round in range(25):
            names = list(variants) if round % 2 == 0 else list(variants)[::-1]
            for name in names:
                args, storage, lengths, dense = variants[name]
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                enqueue(bsnd_probe, args, storage, lengths, dense=dense)
                torch.npu.synchronize()
                if round >= 5:
                    samples[name].append((time.perf_counter_ns() - start) / 1000)
        dense_row = variants["v1-full"][1][1:-1].cpu().tolist()
        full_row = variants["v2-full"][1][1:-1].cpu().tolist()
        assert dense_row[8:] == full_row[8:PREFIX]
        assert full_row[PREFIX:] == [sk] * b
        records.append(
            dict(
                shape=[b, sq, h, sk],
                wall_us=samples,
                median_us={name: statistics.median(v) for name, v in samples.items()},
            )
        )
    (bsnd_probe["directory"] / f"timing-{dtype}.json").write_text(
        json.dumps(
            dict(
                dtype=dtype,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                records=records,
                scope="AICPU generation, ctypes launch/events plus per-call synchronization; no attention or public E2E",
            ),
            indent=2,
        )
        + "\n"
    )
