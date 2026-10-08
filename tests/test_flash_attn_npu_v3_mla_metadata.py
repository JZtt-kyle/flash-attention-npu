"""AICPU dense schedule component: pinned source, host and device must agree.

Only this test bridge exposes the component; public MLA metadata stays rejected.
Synthetic platform inputs exercise the shared CPU algorithm on AICPU and do not
constitute attention-kernel acceptance on other chips or C1V1 hardware.
"""

import ctypes as ct
from contextlib import contextmanager
import hashlib
import itertools
import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla_metadata_build import build_probe
from tests.test_flash_attn_npu_v3_mla_schedule import artifact_dir as artifact_dir
from tests.test_flash_attn_npu_v3_mla_schedule import oracle as oracle
from tests.test_flash_attn_npu_v3_mla_schedule import case, evaluate

WORDS = 401
MAGIC = 0x4D4C4131
SENTINEL = 0x12345678


class Shape(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in ("batch", "sq", "heads", "sk")]


class Memory(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in ("l1", "ub", "l0a", "l0b", "l0c")]


class Platform(ct.Structure):
    _fields_ = [
        ("aic", ct.c_uint32),
        ("aiv", ct.c_uint32),
        ("memory", Memory),
        ("libapi", ct.c_uint64),
        ("block_dims", ct.c_uint32 * 26),
    ]


class Args(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", Shape),
        ("platform", Platform),
        ("cache_seqlens", ct.c_uint64),
        ("cache_seqlens_count", ct.c_uint64),
    ]


def arguments(values, dtype):
    return Args(
        0,
        WORDS,
        dtype,
        Shape(*values[:4]),
        Platform(
            values[4],
            values[5],
            Memory(*values[6:11]),
            values[11],
            (ct.c_uint32 * 26)(*[values[12] or c for c in range(1, 27)]),
        ),
    )


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_METADATA_ARTIFACT_DIR", tmp_path_factory.mktemp("mla-metadata"))
    )
    with load_probe(build_probe(directory), directory) as result:
        yield result


@contextmanager
def load_probe(library, directory):
    """Own test-only AICPU streams/events, also for standalone diagnostics."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    assert "910B" in torch.npu.get_device_name(0)
    lib = ct.CDLL(str(library))
    lib.metadata_args_size.restype = ct.c_uint32
    lib.metadata_words.restype = ct.c_uint32
    assert lib.metadata_args_size() == ct.sizeof(Args)
    assert lib.metadata_words() == WORDS
    lib.metadata_platform.argtypes = [ct.POINTER(Platform)]
    lib.metadata_platform.restype = ct.c_int
    lib.metadata_host.argtypes = [ct.POINTER(Args), ct.c_void_p]
    lib.metadata_host.restype = None
    lib.launch_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(Args)]
    lib.launch_metadata.restype = ct.c_int
    p = Platform()
    assert lib.metadata_platform(ct.byref(p)) == 0
    acl = ct.CDLL(str(Path(os.environ["ASCEND_HOME_PATH"]) / "lib64/libascendcl.so"))
    acl.aclrtCreateEvent.argtypes = [ct.POINTER(ct.c_void_p)]
    acl.aclrtDestroyEvent.argtypes = [ct.c_void_p]
    events = [ct.c_void_p() for _ in range(4)]
    for event in events:
        assert acl.aclrtCreateEvent(ct.byref(event)) == 0
    result = dict(
        lib=lib,
        directory=directory,
        platform=p,
        events=events,
        workers=[torch.npu.Stream(), torch.npu.Stream()],
    )
    (directory / "platform.json").write_text(
        json.dumps(
            dict(
                aic=p.aic,
                aiv=p.aiv,
                memory=[getattr(p.memory, n) for n, _ in Memory._fields_],
                libapi=p.libapi,
                block_dims=list(p.block_dims),
            ),
            indent=2,
        )
        + "\n"
    )
    try:
        yield result
    finally:
        torch.npu.synchronize()
        for event in events:
            assert acl.aclrtDestroyEvent(event) == 0


def enqueue(probe, args, storage, worker=0, cache_seqlens=None):
    output = storage[1:-1]
    assert output.is_contiguous() and output.numel() == WORDS
    args.output = output.data_ptr()
    if cache_seqlens is not None:
        assert cache_seqlens.dtype == torch.int32 and cache_seqlens.is_contiguous()
        assert cache_seqlens.device == output.device and cache_seqlens.ndim == 1
        args.cache_seqlens = cache_seqlens.data_ptr()
        args.cache_seqlens_count = cache_seqlens.numel()
    producer = torch.npu.current_stream()
    worker_stream = probe["workers"][worker]
    ready, done = probe["events"][2 * worker : 2 * worker + 2]
    assert (
        probe["lib"].launch_metadata(
            producer.npu_stream, worker_stream.npu_stream, ready, done, ct.byref(args)
        )
        == 0
    )
    # The fixture also retains storage until synchronization; explicitly record
    # the auxiliary stream, as the production allocator integration will do.
    output.record_stream(worker_stream)
    if cache_seqlens is not None:
        cache_seqlens.record_stream(worker_stream)
    # Independent host schedule for full capacity only; never read device
    # lengths here. Rejection tests separately assert their expected header.
    host = (ct.c_uint32 * WORDS)()
    probe["lib"].metadata_host(ct.byref(args), host)
    return list(host)


def wide(values):
    return [lo | (hi << 32) for lo, hi in zip(values[::2], values[1::2])]


def source_projection(row, inputs):
    s = row[8:]
    assert s[:7] == [inputs[0], 1, inputs[2], inputs[1], inputs[3], 512, 64]
    assert s[9:12] == [inputs[4], inputs[5], inputs[5] // inputs[4]]
    cost = wide(s[93:95])[0]
    fd, w = s[95:99], wide(s[359:383])
    assert fd[:3] == [0, 0, 0]
    assert wide(s[383:393]) == [516096, 174080 if s[11] == 2 else 178176, 65536, 65536, 131072]
    # Check segment boundaries independently from the host reserve calculation.
    cursor = 0
    for offset, size in zip(w[4:8], (w[0] * 4, w[0] * 2, w[1] * 4, w[1] * 4)):
        assert offset == cursor and offset % 32 == 0
        cursor += 2 * s[13] * size
    assert cursor == w[8] <= w[9]
    return [0, s[7], s[8], s[12], cost, *fd, *w[:4], *w[9:12], s[13], s[14]] + s[15:93] + s[99:359]


def compare(probe, cases, dtype, oracle=None, name="cases"):
    storage = torch.full((len(cases), WORDS + 2), SENTINEL, dtype=torch.int32, device="npu")
    hosts = [enqueue(probe, arguments(c, dtype), storage[i]) for i, c in enumerate(cases)]
    actual = storage.cpu().tolist()
    source = evaluate(oracle, cases) if oracle is not None else None
    rows = []
    for i, (row, host) in enumerate(zip(actual, hosts)):
        assert row[0] == row[-1] == SENTINEL
        row = [word & 0xFFFFFFFF for word in row[1:-1]]
        assert row == host, (cases[i], row, host)
        assert row[:5] == [MAGIC, 1, 1, WORDS, dtype]
        assert row[7] == 0
        if row[5] == 0:
            assert row[6] == 0
            projected = source_projection(row, cases[i])
            if source is not None:
                assert projected == source[i], (cases[i], projected, source[i])
        else:
            assert not any(row[8:])  # Error output never retains a previous payload.
        rows.append(row)
    record = dict(
        name=name,
        dtype=dtype,
        cases=len(cases),
        pinned_source=source is not None,
        input_sha256=hashlib.sha256(json.dumps(cases).encode()).hexdigest(),
        output_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
    )
    with (probe["directory"] / "comparisons.jsonl").open("a") as log:
        log.write(json.dumps(record) + "\n")
    return rows


@pytest.mark.parametrize("dtype", (1, 2))
def test_real_platform_boundaries(probe, oracle, dtype):
    p = probe["platform"]
    assert list(p.block_dims)[: p.aic] == list(range(1, p.aic + 1))
    cases = [
        case(
            b,
            sq,
            h,
            sk,
            p.aic,
            2,
            **{n: getattr(p.memory, n) for n, _ in Memory._fields_},
            libapi=p.libapi,
        )
        for b, (sq, h), sk in itertools.product(
            sorted({1, 2, p.aic // 2, p.aic - 1, p.aic, p.aic + 1, 2 * p.aic + 1}),
            [(1, h) for h in (1, 15, 16, 17, 127, 128, 129, 255, 256)]
            + [(7, 16), (2, 128), (128, 2), (256, 1)],
            (1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512),
        )
    ]
    compare(probe, cases, dtype, oracle, "real-platform")


@pytest.mark.parametrize("aic,ratio", itertools.product((1, 2, 3, 20, 24, 26), (1, 2)))
def test_synthetic_resources_on_aicpu(probe, oracle, aic, ratio):
    cases = [
        case(b, sq, h, sk, aic, ratio)
        for b, (sq, h), sk in itertools.product(
            sorted({1, max(1, aic // 2), aic + 1, 2 * aic + 1}),
            ((1, 1), (7, 16), (2, 128)),
            (1, 257, 512),
        )
    ]
    compare(probe, cases + cases[::-1], 1, oracle, f"synthetic-{aic}-{ratio}")


@pytest.mark.parametrize("dtype", (1, 2))
def test_multiblock_on_aicpu(probe, oracle, dtype):
    cases = [
        case(b, sq, h, sk)
        for b, (sq, h), sk in itertools.product(
            (1, 19, 20, 21),
            ((257, 1), (7, 73), (4, 128), (3, 171), (17, 257), (129, 128)),
            (1, 65, 512),
        )
    ]
    cases += [case(10, 2, m, 65) for m in range(128, 257)]
    compare(probe, cases, dtype, oracle, "multiblock")


@pytest.mark.parametrize("dtype", (1, 2))
def test_wide_fields_and_platform_policy(probe, oracle, dtype):
    cases = [case(batch=3, block_dim=20), case(batch=4, aic=3), case(batch=10, heads=256)]
    budget = evaluate(oracle, [cases[0]])[0][13]
    cases.append(case(batch=3, block_dim=20, libapi=2**64 - 1 - budget))
    rows = compare(probe, cases, dtype, oracle, "wide-policy")
    assert wide(rows[-1][8 + 359 : 8 + 383])[-1] == 2**64 - 1
    assert rows[0][8 + 13] == 20
    assert rows[1][8 + 12] == 2  # Equal source cost retains the first split.
    assert rows[2][8 + 7] == 128


@pytest.mark.parametrize("dtype", (1, 2))
def test_device_failure_status(probe, dtype):
    failures = [
        ({"batch": 0}, 1),
        ({"sq": -1}, 1),
        ({"heads": 2**23}, 2),
        ({"sk": 16385}, 1),
        ({"batch": 2**32}, 2),
        ({"sq": 2**63 - 1}, 2),
        ({"aic": 0}, 3),
        ({"aic": 27}, 3),
        ({"aiv": 39}, 3),
        ({"l1": 516095}, 4),
        ({"ub": 174079}, 4),
        ({"l0a": 65535}, 4),
        ({"l0b": 65535}, 4),
        ({"l0c": 131071}, 4),
        ({"libapi": 2**64 - 1}, 5),
        ({"batch": 41, "block_dim": 1}, 3),
    ]
    rows = compare(probe, [case(**options) for options, _ in failures], dtype, name="failures")
    assert [r[5:7] for r in rows] == [[2, status] for _, status in failures]


@pytest.mark.parametrize("dtype", (1, 2))
def test_two_kv_block_source_schedules(probe, dtype, oracle):
    cases = [
        case(b, sq, h, sk)
        for b, (sq, h), sk in itertools.product(
            (1, 2, 10, 20, 21, 41), ((1, 17), (2, 128), (171, 3), (129, 128)), (513, 769, 1024)
        )
    ]
    compare(probe, cases, dtype, oracle, "two-kv-blocks")


@pytest.mark.parametrize("dtype", (1, 2))
def test_long_kv_source_schedules(probe, dtype, oracle):
    cases = [
        case(b, sq, h, sk)
        for b, (sq, h), sk in itertools.product(
            (1, 2, 10, 20, 21, 41),
            ((1, 17), (2, 128), (171, 3), (129, 128)),
            (1025, 1536, 4097, 8192, 8193, 16384),
        )
    ]
    compare(probe, cases, dtype, oracle, "long-kv")


@pytest.mark.parametrize("dtype", (0, 3, 0xFFFFFFFF))
def test_unsupported_dtype(probe, dtype):
    assert compare(probe, [case()], dtype, name="dtype")[0][5:7] == [1, 0]


def test_repeated_overwrite_and_streams(probe):
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    outputs = []
    for i in range(2):
        with torch.npu.stream(streams[i]):
            storage = torch.full((WORDS + 2,), SENTINEL, dtype=torch.int32, device="npu")
            for iteration in range(4):
                for c in (
                    case(batch=2, heads=8, sk=257),
                    case(sk=16385),
                    case(sq=7, heads=16, sk=65),
                ):
                    host = enqueue(probe, arguments(c, 1 + iteration % 2), storage, worker=i)
                    outputs.append((storage.clone(), host))
    for output, host in outputs:
        words = output.cpu().tolist()
        assert words[0] == words[-1] == SENTINEL
        assert [w & 0xFFFFFFFF for w in words[1:-1]] == host


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("sk", (1, 17, 65, 257, 512, 513, 1024, 1025, 8193, 16384))
def test_device_cache_lengths(probe, dtype, sk):
    # Nonzero, only 4-byte-aligned offset, with a batch spanning multiple cores.
    batch = 2 * probe["platform"].aic + 1
    lengths_storage = torch.full((batch + 2,), SENTINEL, dtype=torch.int32, device="npu")
    lengths = lengths_storage[1:-1]
    lengths.fill_(sk)
    storage = torch.full((WORDS + 2,), SENTINEL, dtype=torch.int32, device="npu")
    args = Args(0, WORDS, dtype, Shape(batch, 1, 8, sk), probe["platform"])
    expected = enqueue(probe, args, storage, cache_seqlens=lengths)
    assert [v & 0xFFFFFFFF for v in storage[1:-1].cpu().tolist()] == expected
    # Every position must be inspected, including the final sample. The
    # mismatching value is updated on device before the AICPU stream handshake.
    for index in range(batch):
        lengths.fill_(sk)
        lengths[index] = sk - 1
        enqueue(probe, args, storage, cache_seqlens=lengths)
        words = storage.cpu().tolist()
        assert words[0] == words[-1] == SENTINEL
        assert words[1:9] == [MAGIC, 1, 1, WORDS, dtype, 4, 0, 0]
        assert not any(words[9:-1])
    lengths.fill_(sk)
    enqueue(probe, args, storage, cache_seqlens=lengths)
    assert [v & 0xFFFFFFFF for v in storage[1:-1].cpu().tolist()] == expected
    assert lengths_storage.cpu().tolist() == [SENTINEL, *([sk] * batch), SENTINEL]


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("value", (-(2**31), -1, 0, 1, 256, 258, 512, 513, 2**31 - 1))
def test_unsupported_cache_length(probe, dtype, value):
    args = Args(0, WORDS, dtype, Shape(2, 1, 8, 257), probe["platform"])
    storage = torch.full((WORDS + 2,), SENTINEL, dtype=torch.int32, device="npu")
    for values in ([value, value], [257, value], [value, 257]):
        lengths = torch.tensor(values, dtype=torch.int32, device="npu")
        enqueue(probe, args, storage, cache_seqlens=lengths)
        words = storage.cpu().tolist()
        assert words[0] == words[-1] == SENTINEL
        assert words[1:9] == [MAGIC, 1, 1, WORDS, dtype, 4, 0, 0]
        assert not any(words[9:-1])
        assert lengths.cpu().tolist() == values


def test_device_cache_length_buffer_bounds(probe):
    args = Args(0, WORDS, 1, Shape(2, 1, 8, 257), probe["platform"])
    storage = torch.full((WORDS + 2,), SENTINEL, dtype=torch.int32, device="npu")
    lengths = torch.full((2,), 257, dtype=torch.int32, device="npu")
    expected = enqueue(probe, args, storage, cache_seqlens=lengths)
    for address, count in (
        (0, 2),
        (lengths.data_ptr(), 0),
        (lengths.data_ptr(), 1),
        (lengths.data_ptr(), 3),
        (lengths.data_ptr(), 2**64 - 1),
        (lengths.data_ptr() + 1, 2),
        (lengths.data_ptr() + 2, 2),
        (2**64 - 4, 2),
    ):
        args.cache_seqlens, args.cache_seqlens_count = address, count
        enqueue(probe, args, storage)
        words = storage.cpu().tolist()
        assert words[0] == words[-1] == SENTINEL
        assert words[1:9] == [MAGIC, 1, 1, WORDS, 1, 3, 0, 0]
        assert not any(words[9:-1])
    enqueue(probe, args, storage, cache_seqlens=lengths)
    assert [v & 0xFFFFFFFF for v in storage[1:-1].cpu().tolist()] == expected


def test_cache_length_updates_and_streams(probe):
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    pending = []
    for i, stream in enumerate(streams):
        with torch.npu.stream(stream):
            lengths = torch.empty((2,), dtype=torch.int32, device="npu")
            storage = torch.full((WORDS + 2,), SENTINEL, dtype=torch.int32, device="npu")
            args = Args(0, WORDS, 1 + i, Shape(2, 1, 8, 257), probe["platform"])
            for value in (257, 0, 256, 258, 257, -1, 257):
                lengths.fill_(value)
                expected = enqueue(probe, args, storage, worker=i, cache_seqlens=lengths)
                if value != 257:
                    expected = [MAGIC, 1, 1, WORDS, 1 + i, 4, 0, 0] + [0] * (WORDS - 8)
                pending.append((storage.clone(), expected))
            # The producer records the worker stream before these local
            # references are dropped; the ordered caller stream owns updates.
            del args, lengths, storage
    for storage, expected in pending:
        words = storage.cpu().tolist()
        assert words[0] == words[-1] == SENTINEL
        assert [v & 0xFFFFFFFF for v in words[1:-1]] == expected
