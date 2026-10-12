"""Standalone PA attention helpers and BSND tests with source/independent O/LSE math.

Only the source adapter materializes layouts outside the CANN ABI. Neither
oracle reads the v6 spans to interpret query-used, logical KV or page IDs.
"""

import ctypes as ct
import hashlib
import itertools
import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.mla_paged_reference import references as references
from tests.mla_dense_build import ROOT, build_probes, sha256
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_dense import Schedule
from tests.test_flash_attn_npu_v3_mla_bsnd_stride import make_view, strides_for
from tests.test_flash_attn_npu_v3_mla_used_attention import check_math
from tests.test_flash_attn_npu_v3_mla_paged_metadata_aicpu import (
    paged_probe as paged_probe,
    PagedArgs,
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.test_flash_attn_npu_v3_mla_paged_metadata_consumer import (
    Context,
    Query,
    Inputs,
    Array,
    Cache,
    CacheLayout,
    TableView,
    TensorView,
    ARRAY_NAMES,
    SENTINEL,
    CORRUPTIONS,
    CACHE_FAULTS,
    change,
    context_for,
    expected_schedule,
)
from tests.test_flash_attn_npu_v3_mla_paged_schedule import (
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    paged_case,
    execute,
)
from tests.test_flash_attn_npu_v3_mla_paged_metadata import decode_record


class PagedSchedule(ct.Structure):
    _fields_ = [("tiling", Schedule), ("query", Query)] + [
        (name, ct.c_uint32) for name in ("flags", "page", "capacity")
    ]


def config(
    page=112,
    profile="independent",
    q=(3, 3),
    lengths=None,
    uq=(3, 1),
    heads=8,
    capacity=None,
    pool=7,
    aic=None,
    repeat=False,
    layout=0,
):
    return dict(
        page=page,
        profile=profile,
        q=list(q),
        lengths=list(lengths if lengths is not None else (page + 1, 2 * page - 3)),
        uq=None if uq is None else list(uq),
        heads=heads,
        capacity=capacity,
        pool=pool,
        aic=aic,
        repeat=repeat,
        layout=layout,
    )


# Frozen before measurement; identical non-FD schedules on both sides. Zero KV
# rows are included, while all-zero/empty-table source rejection is tested separately.
# Allocate enough physical pages to avoid folding long sequences into a tiny pool.
PERFORMANCE_CASES = [
    config(p, profile, q, kv, uq, h, pool=2 + sum((n + p - 1) // p for n in kv))
    for p in (16, 112, 128, 256, 528, 1024)
    for profile, q, kv, uq, h in (
        ("contiguous", (1, 1), (257, 0), None, 128),
        ("pa02_both", (3, 3), (8193, 1025), (3, 1), 128),
    )
]


def configure_producer(producer):
    lib = producer["lib"]
    lib.launch_paged_metadata.argtypes = [ct.c_void_p] * 4 + [
        ct.POINTER(PagedArgs),
        ct.c_uint64,
        ct.c_bool,
    ]
    lib.launch_paged_metadata.restype = ct.c_int
    lib.paged_metadata_host.argtypes = [ct.POINTER(PagedArgs), ct.c_void_p]
    lib.paged_metadata_host.restype = ct.c_uint32


def load_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["paged"] and manifest["ndebug"]
    for path, digest in manifest["sources"].items():
        assert sha256(ROOT / path) == digest, path
    for path, digest in manifest["dependencies"].items():
        assert sha256(Path(path)) == digest, path
    libs = {}
    for name in ("oracle", "migrated"):
        row = manifest["libraries"][name]
        assert sha256(Path(row["path"])) == row["sha256"]
        lib = ct.CDLL(row["path"])
        lib.launch_paged_attention.argtypes = [ct.c_void_p] * 8 + [
            ct.c_uint32,
            ct.POINTER(Context),
            ct.POINTER(Inputs),
            ct.POINTER(Cache),
            ct.POINTER(PagedSchedule),
            ct.POINTER(TensorView),
            ct.c_float,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_bool,
            ct.c_void_p,
            ct.c_bool,
            ct.c_uint32,
        ]
        lib.launch_paged_attention.restype = ct.c_int
        lib.build_attention_cache.argtypes = [
            ct.POINTER(TableView),
            ct.POINTER(TensorView),
            ct.POINTER(TensorView),
            ct.POINTER(CacheLayout),
        ]
        lib.build_attention_cache.restype = ct.c_uint32
        libs[name] = lib
    libs["dtype"] = torch.float16 if manifest["dtype"] == "fp16" else torch.bfloat16
    (directory / "loaded.json").write_text(json.dumps(manifest["libraries"], indent=2) + "\n")
    return libs, directory


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def paged_attention(tmp_path_factory, request):
    directory = (
        Path(os.environ.get("MLA_PAGED_ATTENTION_DIR", tmp_path_factory.mktemp("paged-attention")))
        / request.param
    )
    if os.environ.get("MLA_PAGED_ATTENTION_REUSE") != "1":
        build_probes(directory, request.param, paged=True)
    return load_attention(directory)


def address(t):
    return (
        0 if t is None else t.untyped_storage().data_ptr() + t.storage_offset() * t.element_size()
    )


def descriptor(t, table=False):
    rank = 2 if table else 4
    cls = TableView if table else TensorView
    return cls(
        (ct.c_int64 * rank)(*t.shape),
        (ct.c_int64 * rank)(*t.stride()),
        t.storage_offset(),
        t.untyped_storage().nbytes(),
    )


def cache_for(probes, data, table):
    layout = CacheLayout()
    assert (
        probes[0]["migrated"].build_attention_cache(
            ct.byref(descriptor(table, True)),
            ct.byref(descriptor(data["k"])),
            ct.byref(descriptor(data["v"])),
            ct.byref(layout),
        )
        == 0
    )
    return Cache(
        layout,
        *[
            Array(address(t), t.untyped_storage().nbytes() // t.element_size() - t.storage_offset())
            for t in (table, data["k"], data["v"])
        ],
    )


def cache_strides(shape, profile, name):
    pool, page, _, dim = shape
    if profile == f"wide_{name}_1":
        # P>=16: an individual >uint32 row stride would need >120 GiB.
        # Instead actually read a last-row address beyond uint32 in ~8 GiB.
        return (page * dim + 19, (1 << 32) // (page - 1) + 17, dim, 1)
    if profile.startswith("pa02_"):
        mode = profile[5:]
        pad = (17 if name == "k" else 29 if mode == "v" else 37) if mode in (name, "both") else 0
        return (page * dim + pad, dim, dim, 1)
    return strides_for(shape, profile, name)


def make_case(
    probes, producer, spec, *, scale=None, perturb=None, amplitude=1.0, full_launch=False
):
    spec = dict(spec)
    dtype = probes[0]["dtype"]
    q, uq, kv, p = spec["q"], spec["uq"], spec["lengths"], spec["page"]
    tnd = spec["layout"] == 2
    batch, sq, heads = len(q), max(q, default=0), spec["heads"]
    assert (tnd or all(x == sq for x in q)) and len(kv) == batch
    used = q if uq is None else uq
    platform = producer["platform"]
    item = paged_case(
        spec["layout"],
        q,
        kv,
        uq,
        p,
        spec["capacity"],
        heads=heads,
        aic=spec["aic"] or platform.aic,
        ratio=2,
        libapi=platform.libapi,
        **{n: getattr(platform.memory, n) for n in ("l1", "ub", "l0a", "l0b", "l0c")},
    )
    if full_launch and batch and sq and spec["capacity"]:
        item["block_dim"] = platform.block_dims[platform.aic - 1]
    kind = 1 if dtype == torch.float16 else 2
    row = host_record(producer, item, kind)
    context = context_for(row, item, kind)
    args, inputs = arguments(item, kind), device_inputs(item)
    storage = storage_for(args)
    schedule = PagedSchedule.from_buffer_copy((ct.c_uint32 * 96)(*expected_schedule(row)))
    width, pool = item["sk"] // p, spec["pool"]
    table = torch.full((batch, width), -1, dtype=torch.int32)
    active = torch.zeros((pool, p), dtype=torch.bool)
    for b, length in enumerate(kv):
        for j in range((length + p - 1) // p):
            page_id = (pool - 2 - 2 * b - (j % 2 if spec["repeat"] else j)) % pool
            table[b, j] = page_id
            active[page_id, : min(p, length - j * p)] = True
    if spec["profile"] == "table_broadcast" and batch:
        assert len(set(kv)) == 1
        table[:] = table[0].clone()
        active[:] = True  # Table aliases are checked independently below.
    gen = torch.Generator().manual_seed(20261011)
    host, devices, checks = {}, {name: {} for name in ("oracle", "migrated")}, []
    for name, offset in zip(("q", "qv", "k", "v"), (5, 11, 19, 29)):
        is_query = name in ("q", "qv")
        shape = (
            (1 if tnd else batch, sum(q) if tnd else sq, heads, 64 if name == "q" else 512)
            if is_query
            else (pool, p, 1, 64 if name == "k" else 512)
        )
        data = torch.randn(shape, generator=gen).to(dtype)
        if is_query:
            data.mul_(amplitude)
            if perturb is not None:
                target, lo, hi = (("qv", 0, 256), ("qv", 256, 512), ("q", 0, 32), ("q", 32, 64))[
                    perturb
                ]
                if name == target:
                    data[..., :lo] = 0
                    data[..., hi:] = 0
                else:
                    data.zero_()
            pos = 0
            for b, (allocation, length) in enumerate(zip(q, used)):
                if tnd:
                    data[0, pos + length : pos + allocation] = torch.nan
                    pos += allocation
                else:
                    data[b, length:] = torch.nan
        else:
            data[~active] = torch.nan
        profile = spec["profile"]
        if profile == "ntd":
            profile = "bnsd"
        if profile.startswith("table_") or (is_query and profile.startswith("pa02_")):
            profile = "contiguous"
        strides = (
            strides_for(shape, profile, name) if is_query else cache_strides(shape, profile, name)
        )
        device, logical, guards = make_view(
            data, strides, offset, overlap=profile in ("broadcast", "overlap")
        )
        host[name] = logical
        devices["migrated"][name] = device[0] if tnd and is_query else device
        checks += guards
        # Preserve native PA02 page stride. Q, cache row and table views outside
        # the source ABI are independently materialized before its invocation.
        native = not is_query and spec["profile"].startswith("pa02_")
        source_data = logical
        if tnd and is_query:
            # Independently pack effective Q segments from allocation/used inputs.
            # The source parser uses these cumulative tails for both starts and lengths.
            pos, chunks = 0, []
            for allocation, length in zip(q, used):
                chunks.append(logical[:, pos : pos + length])
                pos += allocation
            source_data = torch.cat(chunks, dim=1) if chunks else logical
        src_strides = strides if native else strides_for(source_data.shape, "contiguous", name)
        source, _, guards = make_view(source_data, src_strides, offset + 2)
        devices["oracle"][name] = source
        checks += guards
    # Preserve int32 page-table offset/row stride, including legal row aliases.
    table_stride = 0 if spec["profile"] == "table_broadcast" else width + 3
    if spec["profile"] == "table_wide":
        table_stride = (1 << 32) + 17
    if spec["profile"] == "table_overlap":
        table_stride = 1
        assert batch == 2 and kv[1] <= max(0, kv[0] - p)
        table[1, : max(0, (kv[1] + p - 1) // p)] = table[0, 1 : 1 + (kv[1] + p - 1) // p]
    span = (batch - 1) * table_stride + width if batch and width else 0
    if spec["profile"] == "table_wide":
        table_device_base = torch.empty(span + 34, dtype=torch.int32, device="npu")
        table_device_base[:1024].fill_(-1)  # Poison uint32-truncated row addresses.
        for start in (0, span + 17):
            guard = torch.full((17,), SENTINEL, dtype=torch.int32)
            table_device_base[start : start + 17].copy_(guard.npu())
            checks.append((guard, table_device_base[start : start + 17]))
        for b in range(batch):
            start = 17 + b * table_stride
            table_device_base[start : start + width].copy_(table[b].npu())
            checks.append((table[b], table_device_base[start : start + width]))
        table_host = table
    else:
        table_base = torch.full((span + 34,), SENTINEL, dtype=torch.int32)
        for b in range(batch):
            for j in range(width):
                # Overlapping padding must not overwrite an earlier effective ID.
                if table[b, j] >= 0 or table_base[17 + b * table_stride + j] == SENTINEL:
                    table_base[17 + b * table_stride + j] = table[b, j]
        table_host = table_base.as_strided((batch, width), (table_stride, 1), 17)
        table_device_base = table_base.npu()
        checks.append((table_base, table_device_base))
    tables = {
        "migrated": table_device_base.as_strided((batch, width), (table_stride, 1), 17),
        "oracle": table_host.contiguous().npu(),
    }
    caches = {n: cache_for(probes, devices[n], tables[n]) for n in devices}
    outputs, output_bases, workspace, ws_bases = {}, {}, {}, {}
    for n in devices:
        outputs[n], output_bases[n] = [], []
        total = sum(used) if n == "oracle" else sum(q)
        output_shape = (total, heads, 512) if tnd else (batch, sq, heads, 512)
        lse_shape = (
            ((total, heads) if n == "oracle" else (heads, total)) if tnd else (batch, heads, sq)
        )
        for shape, typ in ((output_shape, dtype), (lse_shape, torch.float32)):
            count = 1
            for size in shape:
                count *= size
            base = torch.full((count + 64,), 123.0, dtype=typ, device="npu")
            output_bases[n].append(base)
            outputs[n].append(base[32:-32].view(shape))
        ws_bases[n] = torch.full((context.workspace + 1024,), 205, dtype=torch.uint8, device="npu")
        workspace[n] = ws_bases[n][512:-512]
    case = dict(
        config=spec,
        item=item,
        context=context,
        schedule=schedule,
        args=args,
        inputs=inputs,
        metadata_storage=storage,
        metadata=storage[1:-1],
        expected=row,
        host=host,
        devices=devices,
        tables=tables,
        table_host=table_host.clone(),
        caches=caches,
        input_checks=checks,
        outputs=outputs,
        output_bases=output_bases,
        workspace=workspace,
        ws_bases=ws_bases,
        scale=1 / 24 if scale is None else scale,
        source_q=(
            torch.tensor(used, dtype=torch.int64).cumsum(0)
            if tnd
            else torch.tensor(used, dtype=torch.int64)
        ).npu(),
        # PA KV lengths stay per-sample, including with TND Q.
        source_k=torch.tensor(kv, dtype=torch.int64).npu(),
        status=torch.full((3 * context.block_dim * 16,), SENTINEL, dtype=torch.int32, device="npu"),
    )
    case["lists"] = {
        n: torch.tensor([24, 1 << 32, 0xFFFFFFFF, address(d["v"])], dtype=torch.int64).npu()
        for n, d in devices.items()
    }
    case["views"] = {
        n: (TensorView * 2)(
            *[descriptor(d[k].unsqueeze(0) if d[k].ndim == 3 else d[k]) for k in ("q", "qv")]
        )
        for n, d in devices.items()
    }
    with forbid_host_readback():
        enqueue(producer, args, storage, inputs)
    return case


def launch(
    probes,
    case,
    name="migrated",
    *,
    checked=True,
    diagnostic=False,
    words=None,
    context=None,
    cache_changes=None,
    descriptors=None,
    metadata_address=None,
    views=None,
):
    c = context or case["context"]
    inputs = Inputs(*[Array(address(t), 0 if t is None else t.numel()) for t in case["inputs"]])
    change(inputs, descriptors)
    cache = Cache.from_buffer_copy(case["caches"][name])
    change(cache, cache_changes)
    data = case["devices"][name]
    tensors = [
        data["qv"],
        data["q"],
        case["lists"][name],
        *case["outputs"][name],
        case["workspace"][name],
        case["metadata"],
    ]
    stream = torch.npu.current_stream()
    for t in [
        *tensors,
        *data.values(),
        case["tables"][name],
        *case["inputs"],
        case["source_q"],
        case["source_k"],
        case["status"],
    ]:
        if t is not None:
            t.record_stream(stream)
    pointers = [address(t) for t in tensors]
    if metadata_address is not None:
        pointers[-1] = metadata_address
    return probes[0][name].launch_paged_attention(
        stream.npu_stream,
        *pointers,
        case["metadata"].numel() if words is None else words,
        ct.byref(c),
        ct.byref(inputs),
        ct.byref(cache),
        ct.byref(case["schedule"]),
        case["views"][name] if views is None else views,
        case["scale"],
        address(case["source_q"]),
        address(case["source_k"]),
        any(x == 0 for x in case["item"]["uk"]),
        address(case["status"]) if diagnostic else 0,
        checked,
        c.block_dim,
    )


def valid_outputs(case, outputs, *, source=False):
    q = case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"]
    if not q:
        return [
            outputs[0].reshape(0, case["context"].heads, 512),
            outputs[1].reshape(case["context"].heads, 0),
        ]
    if case["item"]["layout"] == 2:
        if source:
            return [outputs[0], outputs[1].T]
        pos, out, lse = 0, [], []
        for allocation, length in zip(case["item"]["q"], q):
            out.append(outputs[0][pos : pos + length])
            lse.append(outputs[1][:, pos : pos + length])
            pos += allocation
        return [torch.cat(out), torch.cat(lse, dim=1)]
    return [
        torch.cat([outputs[0][b, :length] for b, length in enumerate(q)]),
        torch.cat([outputs[1][b, :, :length] for b, length in enumerate(q)], dim=1),
    ]


def check_case(probes, case, *, repeats=2):
    used = case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"]
    empty = torch.tensor(
        [kv == 0 for q, kv in zip(used, case["config"]["lengths"]) for _ in range(q)],
        dtype=torch.bool,
    )
    source_ok = bool(
        case["context"].batch
        and case["context"].query.maximum
        and case["schedule"].tiling.sk
        and (case["item"]["layout"] == 0 or sum(used))
    )
    first = source = None
    for _ in range(repeats):
        with forbid_host_readback():
            if source_ok:
                assert launch(probes, case, "oracle") == 0
            assert launch(probes, case) == 0
        actual = valid_outputs(case, [t.cpu() for t in case["outputs"]["migrated"]])
        if source_ok:
            source = valid_outputs(case, [t.cpu() for t in case["outputs"]["oracle"]], source=True)
            torch.testing.assert_close(actual[0], source[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1][:, ~empty], source[1][:, ~empty], rtol=0, atol=0)
            assert torch.isposinf(source[1][:, empty]).all()
        if first is not None:
            for a, b in zip(actual, first):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        first = actual
    refs = references(case)
    errors = check_math(first, refs)
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    for buffers in case["output_bases"].values():
        for t in buffers:
            assert (t[:32].cpu() == 123).all() and (t[-32:].cpu() == 123).all()
    for t in case["ws_bases"].values():
        assert (t[:512].cpu() == 205).all() and (t[-512:].cpu() == 205).all()
    assert [w & 0xFFFFFFFF for w in case["metadata_storage"].cpu().tolist()] == [
        SENTINEL,
        *case["expected"],
        SENTINEL,
    ]
    report = dict(
        config=case["config"],
        dtype=str(probes[0]["dtype"]),
        scale=case["scale"],
        errors=errors,
        source_compared=source_ok,
        m_base=case["schedule"].tiling.m_base,
        s2_base=case["schedule"].tiling.s2_base,
        expected_metadata=case["expected"],
        inputs={
            n: dict(
                shape=list(t.shape),
                stride=list(t.stride()),
                offset=t.storage_offset(),
                storage_bytes=t.untyped_storage().nbytes(),
            )
            for n, t in case["devices"]["migrated"].items()
        },
    )
    digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:20]
    target = probes[1] / "accuracy"
    target.mkdir(exist_ok=True)
    torch.save(
        dict(
            host=case["host"],
            pages=case["table_host"],
            actual=first,
            source=source,
            references=refs,
            report=report,
        ),
        target / f"{digest}.pt",
    )
    report["tensors"] = str(target / f"{digest}.pt")
    with (probes[1] / "accuracy.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    return report


@pytest.mark.parametrize("page", range(16, 1025, 16))
@pytest.mark.parametrize("profile", ("contiguous", "pa02_k", "pa02_v", "pa02_both"))
def test_all_page_sizes(paged_attention, paged_probe, page, profile):
    # PA01 Q=[3,1], and full BSND allocation in the compact control.
    case = make_case(
        paged_attention,
        paged_probe,
        config(page, profile, uq=None if profile == "contiguous" else (3, 1)),
    )
    check_case(paged_attention, case)


@pytest.mark.parametrize(
    "spec",
    PERFORMANCE_CASES
    + [
        config(
            112,
            "independent",
            (9,) * 5,
            (17, 65, 513, 0, 1),
            (3, 0, 1, 7, 0),
            171,
            aic=1,
            repeat=True,
        ),
        config(528, "independent", (3,) * 3, (512, 0, 513), (2, 0, 1), 256),
        config(16, "independent", (3, 3), (16384, 8193), (1, 2), 8, capacity=16384),
        config(128, "independent", (3,) * 10, (511,) + (0,) * 9, (2,) * 10, 128, aic=15),
        *[
            config(112, "independent", (3, 3), (127, 129), (1, 2), h)
            for h in (1, 15, 16, 127, 128, 129, 255, 256)
        ],
    ],
)
def test_task_ring(paged_attention, paged_probe, spec):
    check_case(
        paged_attention,
        make_case(paged_attention, paged_probe, spec, full_launch=spec["heads"] == 255),
    )


@pytest.mark.parametrize(
    "profile",
    ["bnsd", "independent", "broadcast", "overlap", "table_broadcast", "table_overlap"]
    + [
        f"{axis}_{s}"
        for axis, s in itertools.product(
            ("row", "matrix", "head", "combined"), (65535, 65536, 65537)
        )
    ],
)
def test_strides(paged_attention, paged_probe, profile):
    lengths = (257, 257) if profile == "table_broadcast" else (257, 17)
    check_case(
        paged_attention,
        make_case(paged_attention, paged_probe, config(112, profile, lengths=lengths)),
    )


@pytest.mark.parametrize("pool,capacity", ((7, 0), (0, 0), (0, 224), (7, 224)))
@pytest.mark.parametrize("used", (None, (3, 1), (0, 0)))
def test_empty_cache(paged_attention, paged_probe, pool, capacity, used):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            config(lengths=(0, 0), uq=used, pool=pool, capacity=capacity),
        ),
    )


@pytest.mark.parametrize("layout,pool,used", itertools.product((0, 2), (0, 7), (False, True)))
def test_zero_lengths_with_static_workspace(paged_attention, paged_probe, layout, pool, used):
    spec = config(
        layout=layout,
        q=(3, 3) if layout == 0 else (3, 1),
        lengths=(0, 0),
        uq=(2, 0) if used else None,
        pool=pool,
        capacity=448,
    )
    case = make_case(paged_attention, paged_probe, spec, full_launch=True)
    assert case["context"].block_dim == paged_probe["platform"].aic
    assert case["expected"][20] == 1
    case["context"].workspace = 1 << 20
    case["context"].libapi = paged_probe["platform"].libapi
    base = torch.full((case["context"].workspace + 1024,), 205, dtype=torch.uint8, device="npu")
    case["ws_bases"]["migrated"] = base
    case["workspace"]["migrated"] = base[512:-512]
    check_case(paged_attention, case)
    assert (base.cpu() == 205).all()  # An empty schedule never consumes the reservation.


@pytest.mark.parametrize(
    "q,kv,used,capacity",
    (([], [], None, 0), ([0, 0], [257, 1], None, 448), ([3, 3], [257, 1], [0, 0], 448)),
)
def test_zero_query(paged_attention, paged_probe, q, kv, used, capacity):
    check_case(
        paged_attention,
        make_case(
            paged_attention, paged_probe, config(q=q, lengths=kv, uq=used, capacity=capacity)
        ),
    )


@pytest.mark.parametrize("scale", (0.0, -0.125, 0.25, None))
@pytest.mark.parametrize("perturb", range(4))
def test_scale_halves_and_max_change(paged_attention, paged_probe, scale, perturb):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            config(112, lengths=(1025, 513)),
            scale=scale,
            perturb=perturb,
            amplitude=4,
        ),
    )


def test_independent_schedules(paged_attention, paged_probe, paged_oracles):
    specs = PERFORMANCE_CASES + [config(p) for p in range(16, 1025, 16)]
    items = []
    rows = []
    for spec in specs:
        case = make_case(paged_attention, paged_probe, spec)
        items.append(case["item"])
        rows.append(case["expected"])
    source, native = execute(paged_oracles[0], items), execute(paged_oracles[0], items, native=True)
    kind = 1 if paged_attention[0]["dtype"] == torch.float16 else 2
    for item, row, wanted in zip(items, rows, source):
        decode_record(row, item, kind, wanted)
    (paged_attention[1] / "schedules.json").write_text(
        json.dumps(dict(items=items, source=source, native=native, metadata=rows), indent=2) + "\n"
    )


def check_rejected(case, status):
    assert (case["status"][::16].cpu() == status).all()
    assert (case["status"].view(-1, 16)[:, 1:].cpu() == SENTINEL).all()
    for t in case["output_bases"]["migrated"]:
        assert (t.cpu() == 123).all()
    assert (case["ws_bases"]["migrated"].cpu() == 205).all()


def rejection_case(probes, producer):
    return make_case(
        probes, producer, config(q=(1, 1), uq=(1, 1), lengths=(257, 257), capacity=448, aic=1)
    )


def test_corruption_precedes_attention(paged_attention, paged_probe):
    case = rejection_case(paged_attention, paged_probe)
    for word, value, status in CORRUPTIONS:
        old = case["expected"][word]
        assert old != value
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        with forbid_host_readback():
            assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
        check_rejected(case, status)
        case["metadata"][word] = old
    check_case(paged_attention, case)


def test_cache_descriptors_precede_attention(paged_attention, paged_probe):
    case = rejection_case(paged_attention, paged_probe)
    for path, value, status in CACHE_FAULTS:
        with forbid_host_readback():
            assert (
                launch(
                    paged_attention,
                    case,
                    checked=False,
                    diagnostic=True,
                    cache_changes={path: value},
                )
                == 0
            )
        check_rejected(case, status)
    for words in (0, 8, 407, case["metadata"].numel() - 1, case["metadata"].numel() + 1, 2**32 - 1):
        assert launch(paged_attention, case, checked=False, diagnostic=True, words=words) == 0
        check_rejected(case, 1)
    for pointer in (0, 1, 2**64 - 4):
        assert (
            launch(paged_attention, case, checked=False, diagnostic=True, metadata_address=pointer)
            == 0
        )
        check_rejected(case, 1)
    for axis, name in enumerate(ARRAY_NAMES):
        for field, value in (("address", 1), ("address", 2**64 - 4), ("count", 2**64 - 1)):
            assert (
                launch(
                    paged_attention,
                    case,
                    checked=False,
                    diagnostic=True,
                    descriptors={f"{name}.{field}": value},
                )
                == 0
            )
            check_rejected(case, (12, 13, 11)[axis])
    check_case(paged_attention, case)


@pytest.mark.parametrize("axis", (1, 2))
def test_stale_lengths_and_regeneration(paged_attention, paged_probe, axis):
    case = rejection_case(paged_attention, paged_probe)
    case["inputs"][axis][1] = 0
    with forbid_host_readback():
        assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
    check_rejected(case, 13 if axis == 1 else 11)
    # The next call regenerates from the changed device length arrays.
    spec = dict(case["config"])
    field = "uq" if axis == 1 else "lengths"
    spec[field] = [spec[field][0], 0]
    updated = make_case(paged_attention, paged_probe, spec)
    updated["metadata_storage"] = case["metadata_storage"]
    updated["metadata"] = case["metadata"]
    updated["inputs"] = case["inputs"]
    with forbid_host_readback():
        enqueue(
            paged_probe, updated["args"], updated["metadata_storage"], updated["inputs"], worker=1
        )
    check_case(paged_attention, updated)


@pytest.mark.parametrize("empty_query", (False, True))
def test_pages_rechecked_even_without_queries(paged_attention, paged_probe, empty_query):
    spec = config(q=(0, 0) if empty_query else (3, 3), uq=None if empty_query else (3, 0))
    case = make_case(paged_attention, paged_probe, spec)
    # Batch 1 has no active Q, but its entire effective page prefix is required.
    for value in (-1, -(2**31), spec["pool"], 2**31 - 1):
        case["tables"]["migrated"][1, 1] = value
        assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
        check_rejected(case, 15)


@pytest.mark.parametrize("profile", ("contiguous", "pa02_both", "independent", "broadcast"))
def test_reuse_with_new_cache_and_scale(paged_attention, paged_probe, profile):
    old = make_case(paged_attention, paged_probe, config())
    new = make_case(paged_attention, paged_probe, config(profile=profile, pool=11), scale=-0.125)
    assert new["expected"] == old["expected"]
    new["metadata_storage"], new["metadata"] = old["metadata_storage"], old["metadata"]
    check_case(paged_attention, new)


@pytest.mark.parametrize(
    "name,axis",
    [
        (name, axis)
        for name in ("q", "qv", "k", "v")
        for axis in ((0, 1, 2) if name in ("q", "qv") else (0, 1))
    ],
)
def test_wide_active_stride(paged_attention, paged_probe, name, axis):
    case = make_case(
        paged_attention,
        paged_probe,
        config(16, f"wide_{name}_{axis}", q=(2, 2), lengths=(17, 17), uq=(2, 1), heads=2, pool=2),
    )
    check_case(paged_attention, case)


@pytest.mark.parametrize("field", ("negative", "inner", "storage", "offset", "shape", "overflow"))
def test_invalid_query_view(paged_attention, paged_probe, field):
    case = rejection_case(paged_attention, paged_probe)
    views = (TensorView * 2).from_buffer_copy(case["views"]["migrated"])
    if field == "negative":
        views[0].strides[1] = -1
    elif field == "inner":
        views[1].strides[3] = 2
    elif field == "storage":
        views[1].storage = 2
    elif field == "offset":
        views[1].offset = 2**63 - 1
    elif field == "shape":
        views[0].sizes[1] += 1
    else:
        views[0].strides[0] = 2**63 - 1
    assert launch(paged_attention, case, views=views) == -3
    for t in case["output_bases"]["migrated"]:
        assert (t.cpu() == 123).all()
    assert (case["ws_bases"]["migrated"].cpu() == 205).all()


def test_stream_and_allocator_lifetime(paged_attention, paged_probe):
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    saved = []
    for index in range(6):
        stream = streams[index % 2]
        with torch.npu.stream(stream):
            case = make_case(paged_attention, paged_probe, config(112 if index % 2 else 528))
            refs = references(case)
            with forbid_host_readback():
                assert launch(paged_attention, case) == 0
            event = torch.npu.Event()
            event.record(stream)
            saved.append((case["outputs"]["migrated"], case["item"], refs, event))
        del case
        # Recycle inputs/workspace on the default stream while worker kernels
        # still own them via record_stream; each invocation owns its workspace.
        churn = torch.empty(8 << 20, dtype=torch.uint8, device="npu").fill_(17)
        del churn
    for outputs, item, refs, event in saved:
        event.synchronize()
        actual = valid_outputs(dict(item=item), [t.cpu() for t in outputs])
        check_math(actual, refs)


FAULTS = (
    "version",
    "stale_q",
    "stale_k",
    "page",
    "unused_q_page",
    "workspace",
    "words",
    "empty_pool",
    "empty_query",
    "empty_budget_payload",
)


def fault_case(probes, producer, fault, *, layout=0):
    spec = config(aic=1, layout=layout)
    if layout == 2:
        spec.update(q=[5, 2], uq=[3, 1])
    if fault == "unused_q_page":
        spec["uq"] = [3, 0]
    if fault == "empty_query":
        spec.update(q=[0, 0], uq=None)
    if fault == "empty_pool":
        spec.update(pool=0, lengths=[0, 0])
    if fault == "empty_budget_payload":
        spec.update(lengths=[0, 0])
    case = make_case(probes, producer, spec)
    options = {}
    if fault in ("version", "empty_pool"):
        case["metadata"][1] = 5
        expected = 2
    elif fault == "empty_budget_payload":
        case["context"].workspace = 1 << 20
        case["context"].libapi = producer["platform"].libapi
        case["metadata"][385] = 16
        expected = 7
    elif fault in ("stale_q", "stale_k"):
        case["inputs"][1 if fault == "stale_q" else 2][1] = 0
        expected = 13 if fault == "stale_q" else 11
    elif fault == "stale_prefix":
        case["inputs"][0][1] = 4
        expected = 12
    elif fault in ("page", "unused_q_page", "empty_query"):
        case["tables"]["migrated"][1, 1] = -1
        expected = 15
    elif fault == "workspace":
        c = Context.from_buffer_copy(case["context"])
        c.workspace = 0
        options["context"] = c
        expected = 9
    else:
        options["words"] = case["metadata"].numel() - 1
        expected = 1
    return case, options, expected


@pytest.mark.parametrize("fault", FAULTS)
def test_release_trap(paged_attention, paged_probe, fault):
    import signal
    import subprocess
    import sys

    directory = paged_attention[1] / f"error-{fault}"
    directory.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla_paged_attention_error_probe",
        str(paged_attention[1]),
        str(paged_probe["directory"] / "build-manifest.json"),
        fault,
        str(directory),
    ]
    (directory / "command.json").write_text(json.dumps(command) + "\n")
    with (directory / "subprocess.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            result = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    assert result == 0, (directory / "subprocess.log").read_text()[-3000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["fault"] == fault and report["error"]


def test_wide_page_table_and_singletons(paged_attention, paged_probe):
    torch.npu.empty_cache()
    case = make_case(paged_attention, paged_probe, config(profile="table_wide"))
    assert case["tables"]["migrated"].stride(0) > 2**32
    check_case(paged_attention, case)
    del case
    torch.npu.empty_cache()
    case = make_case(
        paged_attention,
        paged_probe,
        config(16, "singleton", q=(1,), lengths=(1,), uq=None, heads=1, pool=1),
    )
    check_case(paged_attention, case)


@pytest.mark.parametrize("heads", (1, 15, 129))
def test_empty_small_row_counts(paged_attention, paged_probe, heads):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            config(
                16, "contiguous", q=(1,), lengths=(0,), uq=None, heads=heads, pool=0, capacity=0
            ),
        ),
    )
