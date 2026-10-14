"""910B PA Copy: exact L1 NZ data, fixed source and independent scalar indexing.

The source receives the same physical views when its Copy ABI can express
them. Zero or >uint32 table strides use independently materialized source
views; this conversion is outside component timing and is recorded explicitly.
No PA attention or public forward is invoked.
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

from tests.mla.support.mla_paged_copy_build import Geometry, build_probes, load_probes


SLOT_ELEMENTS = 256 * 288 + 512
GUARD_ELEMENTS = 256
SENTINEL = -113.0


@pytest.fixture(scope="module")
def copy_build(tmp_path_factory):
    directory = Path(os.environ.get("MLA_PAGED_COPY_DIR", tmp_path_factory.mktemp("paged-copy")))
    if os.environ.get("MLA_PAGED_COPY_REUSE") != "1":
        build_probes(directory)
    torch.npu.set_device(0)
    assert "910B" in torch.npu.get_device_name(0)
    torch.set_num_threads(2)
    return directory


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def copy_probes(copy_build, request):
    dtype = torch.float16 if request.param == "fp16" else torch.bfloat16
    return load_probes(copy_build, request.param), copy_build, dtype


def cache_view(page, pages, dim, profile, name, dtype):
    row = dim
    page_stride = page * row
    offset = 7 if name == "k" else 13
    if profile == "independent":
        row = dim + (19 if name == "k" else 37)
        page_stride = page * row + (53 if name == "k" else 97)
    elif profile == "overlap":
        row = 17 if name == "k" else 67
        page_stride = 19 if name == "k" else 83
    elif profile == "broadcast":
        row = 0 if name == "k" else dim
        page_stride = 0
    elif profile.startswith(f"{name}_row_"):
        row = int(profile.split("_")[-1])
        page_stride = page * row + 31
    elif profile == f"wide_{name}_page":
        page_stride = (1 << 32) + 65537
    elif profile == f"wide_{name}_row":
        row = (1 << 29) + 1
        page_stride = page * row + 31
    span = 0 if pages == 0 else (pages - 1) * page_stride + (page - 1) * row + dim
    checks = []
    if span < (1 << 27):
        host = ((torch.arange(span + offset + 32, dtype=torch.int64) * 17 + dim) % 193 - 96).to(
            dtype
        ) / 8
        host[:offset] = SENTINEL
        host[offset + span :] = SENTINEL
        logical = host.as_strided((pages, page, dim), (page_stride, row, 1), offset).contiguous()
        base = host.npu()
        checks.append((host, base))
    else:
        # Allocate the entire legal physical view, including holes. Only the
        # real rows and truncation targets are initialized, not a tiny stand-in.
        base = torch.empty(span + offset + 32, device="npu", dtype=dtype)
        logical = (
            (torch.arange(pages * page * dim).reshape(pages, page, dim) * 13 + dim) % 193 - 96
        ).to(dtype) / 8
        poison = torch.full((dim,), torch.nan, dtype=dtype, device="npu")
        for p, r in itertools.product(range(pages), range(page)):
            address = offset + p * page_stride + r * row
            truncated = offset + ((address - offset) & 0xFFFFFFFF)
            base[truncated : truncated + dim].copy_(poison)
        for p, r in itertools.product(range(pages), range(page)):
            address = offset + p * page_stride + r * row
            expected = logical[p, r].contiguous()
            base[address : address + dim].copy_(expected.npu())
            checks.append((expected, base[address : address + dim]))
        for start, count in ((0, offset), (offset + span, 32)):
            expected = torch.full((count,), SENTINEL, dtype=dtype)
            base[start : start + count].copy_(expected.npu())
            checks.append((expected, base[start : start + count]))
    view = base.as_strided((pages, page, 1, dim), (page_stride, row, 0, 1), offset)
    return view, logical, checks


def page_tasks(page):
    # Both QK halves and all four PV D tiles, distinct batches and row tails.
    return [
        [0, 0, 1, 32, 0],
        [1, page - 16, 127, 144, 1],
        [0, page, 255, 256, 2],
        [1, 2 * page - 16, 128, 144, 3],
        [0, page + 16, 17, 48, 4],
        [1, 3 * page, 129, 160, 5],
    ]


def make_case(
    dtype,
    page=112,
    profile="independent",
    tasks=None,
    *,
    physical_pages=5,
    empty=False,
    diagnostic=True,
):
    tasks = tasks if tasks is not None else page_tasks(page)
    assert tasks and all(
        0 <= t[2] <= t[3] <= 256 and t[3] % 16 == 0 and 0 <= t[4] < 6 for t in tasks
    )
    batch = max(t[0] for t in tasks) + 1
    lengths = [
        max((t[1] + t[2] for t in tasks if t[0] == b and t[2]), default=0) for b in range(batch)
    ]
    columns = 0 if empty else (max(lengths) + page - 1) // page + 2
    if empty:
        physical_pages = 0
    table_stride = columns + 7
    if profile == "table_broadcast":
        table_stride = 0
        lengths = [max(lengths)] * batch
    elif profile == "table_overlap":
        table_stride = 1
        lengths = [max(lengths)] * batch
    elif profile == "wide_table":
        table_stride = (1 << 32) + 7
    table_span = (batch - 1) * table_stride + columns if columns else 0
    logical_table = torch.full((batch, columns), -1, dtype=torch.int32)
    for b in range(batch):
        for j in range((lengths[b] + page - 1) // page):
            # Repetition and reverse/non-monotone order; no contiguous-page assumption.
            logical_table[b, j] = (7 - 3 * (b * table_stride + j)) % physical_pages
    table_checks = []
    if table_span < (1 << 27):
        host = torch.full((table_span + 13,), -1, dtype=torch.int32)
        table = host.as_strided((batch, columns), (table_stride, 1), 5)
        for b in range(batch):
            for j in range((lengths[b] + page - 1) // page):
                table[b, j] = logical_table[b, j]
        logical_table = table.contiguous()
        base = host.npu()
        table_checks.append((host, base))
    else:
        base = torch.empty(table_span + 13, dtype=torch.int32, device="npu")
        for b in range(batch):
            truncated = 5 + ((b * table_stride) & 0xFFFFFFFF)
            base[truncated : truncated + columns].fill_(-1)
        for b in range(batch):
            start = 5 + b * table_stride
            base[start : start + columns].copy_(logical_table[b].npu())
            table_checks.append((logical_table[b], base[start : start + columns]))
    table = base.as_strided((batch, columns), (table_stride, 1), 5)
    k, kh, kc = cache_view(page, physical_pages, 64, profile, "k", dtype)
    v, vh, vc = cache_view(page, physical_pages, 512, profile, "v", dtype)
    geometry = Geometry(
        page,
        batch,
        columns,
        physical_pages,
        table_stride,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
    )
    source_k, source_v, source_table = k, v, table
    materialized = profile in ("broadcast", "table_broadcast", "wide_table")
    if materialized:
        source_k, source_v, source_table = kh.npu(), vh.npu(), logical_table.npu()
    source_geometry = Geometry(
        page,
        batch,
        columns,
        physical_pages,
        source_table.stride(0),
        source_k.stride(0),
        source_k.stride(1),
        source_v.stride(0),
        source_v.stride(1),
    )
    output_count = len(tasks) if diagnostic else 1
    output_bases = {
        name: torch.full((output_count * SLOT_ELEMENTS + 32,), SENTINEL, dtype=dtype, device="npu")
        for name in ("oracle", "migrated")
    }
    return dict(
        profile=profile,
        page=page,
        tasks=tasks,
        lengths=lengths,
        logical=(kh, vh, logical_table),
        geometry=geometry,
        source_geometry=source_geometry,
        device={"migrated": (k, v, table), "oracle": (source_k, source_v, source_table)},
        task_tensor=torch.tensor(tasks, dtype=torch.int32).npu(),
        initial=torch.full((SLOT_ELEMENTS,), SENTINEL, dtype=dtype, device="npu"),
        outputs={n: x[16:-16].view(output_count, SLOT_ELEMENTS) for n, x in output_bases.items()},
        output_bases=output_bases,
        checks=kc + vc + table_checks,
        source_materialized=materialized,
    )


def launch(libraries, case, name, *, dump=True, rounds=3):
    geometry = case["source_geometry"] if name == "oracle" else case["geometry"]
    pointers = [*case["device"][name], case["task_tensor"], case["initial"], case["outputs"][name]]
    status = libraries[name].launch_paged_copy(
        torch.npu.current_stream().npu_stream,
        *(x.data_ptr() for x in pointers),
        ct.byref(geometry),
        len(case["tasks"]),
        rounds,
        dump,
    )
    assert status == 0, status


def scalar_expected(case, task):
    # Direct indexing of the original logical host values. No migrated/source
    # helper, segment iterator, layout descriptor or device gather is used.
    k, v, table = case["logical"]
    b, start, rows, dst_rows, mode = task
    indices, values = [], []
    for r in range(rows):
        token = start + r
        physical_page = int(table[b, token // case["page"]])
        page_row = token % case["page"]
        content = v[physical_page, page_row].tolist()
        rope = k[physical_page, page_row].tolist()
        row_values = (
            (content[:256] + rope[:32] if mode == 0 else rope[32:] + content[256:])
            if mode < 2
            else content[(mode - 2) * 128 : (mode - 1) * 128]
        )
        for col, value in enumerate(row_values):
            indices.append(GUARD_ELEMENTS + (col // 16) * dst_rows * 16 + r * 16 + col % 16)
            values.append(value)
    return torch.tensor(indices, dtype=torch.int64), torch.tensor(values, dtype=k.dtype)


def check_case(probes, case, *, rounds=3, label=None):
    libraries, directory, dtype = probes
    for name in libraries:
        launch(libraries, case, name, rounds=rounds)
    torch.npu.synchronize()
    results = {n: t.cpu() for n, t in case["outputs"].items()}
    checked = 0
    for index, task in enumerate(case["tasks"]):
        positions, expected = scalar_expected(case, task)
        for name in libraries:
            actual = results[name][index, positions]
            torch.testing.assert_close(
                actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
            )
            assert (results[name][index, :GUARD_ELEMENTS] == SENTINEL).all()
            assert (results[name][index, GUARD_ELEMENTS + 256 * 288 :] == SENTINEL).all()
            if task[2] == 0:
                assert (results[name][index] == SENTINEL).all()
        checked += expected.numel()
    for before, device in case["checks"]:
        torch.testing.assert_close(
            device.cpu().view(torch.int16), before.view(torch.int16), rtol=0, atol=0
        )
    for base in case["output_bases"].values():
        assert (base[:16].cpu() == SENTINEL).all() and (base[-16:].cpu() == SENTINEL).all()
    record = dict(
        page=case["page"],
        profile=case["profile"],
        dtype=str(dtype),
        tasks=case["tasks"],
        rounds=rounds,
        checked_elements=checked,
        source_materialized=case["source_materialized"],
        source_scope="Copy header with layout stride injection; no native FIA host call",
        output_sha256={
            n: hashlib.sha256(t.view(torch.int16).numpy().tobytes()).hexdigest()
            for n, t in results.items()
        },
        strides={n: [list(t.stride()) for t in tensors] for n, tensors in case["device"].items()},
        storage_bytes={
            n: [t.untyped_storage().nbytes() for t in tensors]
            for n, tensors in case["device"].items()
        },
    )
    evidence = directory / "accuracy"
    evidence.mkdir(exist_ok=True)
    name = label or hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:20]
    torch.save(
        dict(logical_inputs=case["logical"], results=results, record=record),
        evidence / f"{name}.pt",
    )
    with (directory / "accuracy.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


@pytest.mark.parametrize("page", range(16, 1025, 16))
@pytest.mark.parametrize("profile", ("contiguous", "independent"))
def test_all_page_sizes(copy_probes, page, profile):
    check_case(copy_probes, make_case(copy_probes[2], page, profile))


@pytest.mark.parametrize("profile", ("overlap", "broadcast", "table_overlap", "table_broadcast"))
def test_readonly_aliases(copy_probes, profile):
    check_case(copy_probes, make_case(copy_probes[2], 16, profile))


@pytest.mark.parametrize("target,stride", itertools.product(("k", "v"), (65535, 65536, 65537)))
def test_nd2nz_stride_limit(copy_probes, target, stride):
    check_case(
        copy_probes, make_case(copy_probes[2], 16, f"{target}_row_{stride}", physical_pages=2)
    )


@pytest.mark.parametrize(
    "profile", ("wide_k_page", "wide_v_page", "wide_k_row", "wide_v_row", "wide_table")
)
def test_real_wide_addresses(copy_probes, profile):
    tasks = [[b, 9, 7, 32, mode] for b in (0, 1) for mode in range(6)]
    pages = 1 if profile.endswith("row") else 2
    case = make_case(copy_probes[2], 16, profile, tasks, physical_pages=pages)
    check_case(copy_probes, case)
    del case
    torch.npu.empty_cache()


@pytest.mark.parametrize("start,count", ((15, 17), (1, 1), (31, 31), (16, 256), (0, 0)))
def test_tail_and_zero_count(copy_probes, start, count):
    rows = min(256, ((count + 30) // 16 + 1) * 16)
    tasks = [[0, start, count, rows, mode] for mode in range(6)]
    check_case(copy_probes, make_case(copy_probes[2], 16, tasks=tasks), rounds=5)


def test_empty_pool_table(copy_probes):
    tasks = [[0, 0, 0, 16, mode] for mode in range(6)]
    check_case(copy_probes, make_case(copy_probes[2], 528, tasks=tasks, empty=True))


def test_stream_events_and_buffer_reuse(copy_probes):
    cases = [make_case(copy_probes[2], p) for p in (16, 112, 528)]
    workers = [torch.npu.Stream() for _ in range(2)]
    ready = torch.npu.Event()
    ready.record()
    done = []
    for worker in workers:
        worker.wait_event(ready)
        with torch.npu.stream(worker):
            for case in cases:
                for name in copy_probes[0]:
                    launch(copy_probes[0], case, name, rounds=11)
            event = torch.npu.Event()
            event.record()
            done.append(event)
        # The two workers reuse outputs with an explicit dependency.
        ready = event
    for event in done:
        torch.npu.current_stream().wait_event(event)
    for case in cases:
        check_case(copy_probes, case, rounds=7)
