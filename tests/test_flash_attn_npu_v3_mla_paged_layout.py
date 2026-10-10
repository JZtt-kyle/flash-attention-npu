"""Independent PA CPU address/segment acceptance; no production PA enabled.

Compare with an unbounded-integer scalar oracle, real torch view pointers and
verbatim pinned CANN offset/parser/copy helpers compiled with CPU trace stubs.
MLA_PAGED_LAYOUT_DIR retains commands, source blobs, manifests and full cases;
MLA_PAGED_LAYOUT_SANITIZE=1 instruments both executables with ASan/UBSan.
"""

from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess

import pytest
import torch

from tests.test_flash_attn_npu_v3_mla_schedule import CANN_COMMIT, MLA, ROOT

I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1
SOURCE_DIR = "attention/common/op_kernel/memcopy/"


def source_class(source, token, *, templated=False):
    """Retain the complete class, including its template and byte-exact body."""
    start = source.index(token)
    start = source.rfind("\n", 0, start) + 1
    if templated:
        start = source.rfind("template <", 0, start)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    assert source[end] == ";"
    return source[start : end + 1] + "\n"


@pytest.fixture(scope="session")
def paged_drivers(tmp_path_factory):
    target = os.environ.get("MLA_PAGED_LAYOUT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-paged-layout")
    directory.mkdir(parents=True, exist_ok=True)
    source_root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    names = ["parser.h", "offset_calculator_v2.h", "attn_copy_gm_to_l1.h"]
    blobs = {
        name: subprocess.check_output(
            ["git", "-C", str(source_root), "show", f"{CANN_COMMIT}:{SOURCE_DIR}{name}"],
            timeout=30,
        )
        for name in names
    }
    for name, blob in blobs.items():
        (directory / name).write_bytes(blob)
    selections = [
        ("parser.h", "class BlockTableParser", False),
        (
            "offset_calculator_v2.h",
            "struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD",
            True,
        ),
        ("attn_copy_gm_to_l1.h", "class CopyKvGmToL1", True),
        ("attn_copy_gm_to_l1.h", "class CopyKKropePAGmToL1", True),
    ]
    copyright = blobs["parser.h"].decode().split("*/", 1)[0] + "*/\n"
    extracted = [source_class(blobs[n].decode(), t, templated=p) for n, t, p in selections]
    (directory / "source_helpers.inc").write_text(copyright + "\n".join(extracted))
    license_blob = subprocess.check_output(
        ["git", "-C", str(source_root), "show", f"{CANN_COMMIT}:LICENSE"], timeout=30
    )
    (directory / "LICENSE.cann").write_bytes(license_blob)
    files = [
        MLA / "mla_paged_layout.cpp",
        MLA / "mla_paged_layout.h",
        MLA / "mla_bsnd_layout.cpp",
        MLA / "mla_bsnd_layout.h",
        Path(__file__),
        ROOT / "tests/mla_paged_layout_driver.cpp",
        ROOT / "tests/mla_paged_source_driver.cpp",
    ]
    binaries = []
    for name in ("layout", "source"):
        command = [
            os.environ.get("CXX", "g++"),
            "-std=c++17",
            "-O2",
            "-g",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-D_GLIBCXX_ASSERTIONS",
        ]
        if os.environ.get("MLA_PAGED_LAYOUT_SANITIZE") == "1":
            command += [
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
                "-fno-sanitize-recover=all",
            ]
        binary = directory / name
        if name == "layout":
            command += [
                "-I" + str(MLA),
                str(MLA / "mla_paged_layout.cpp"),
                str(MLA / "mla_bsnd_layout.cpp"),
            ]
        else:
            command += ["-I" + str(directory)]
        command += [str(ROOT / f"tests/mla_paged_{name}_driver.cpp"), "-o", str(binary)]
        (directory / f"{name}-compile.json").write_text(json.dumps(command, indent=2) + "\n")
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (directory / f"{name}-compile.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr
        binaries.append(binary)
    manifest = dict(
        commit=CANN_COMMIT,
        source_files={SOURCE_DIR + n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()},
        selections=selections,
        scope="CPU address/copy trace only; explicit FA3 strides injected into source layout",
        files={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *files,
                *binaries,
                directory / "source_helpers.inc",
                directory / "LICENSE.cann",
            ]
        },
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return binaries


def view(sizes, strides, offset=0, element_bytes=2):
    span = 0 if 0 in sizes else 1 + sum((n - 1) * s for n, s in zip(sizes, strides))
    return dict(
        sizes=list(sizes),
        strides=list(strides),
        offset=offset,
        storage=(offset + span) * element_bytes,
    )


def case(page=16, batch=3, width=5, pages=7, table_stride=None):
    return dict(
        table=view((batch, width), (width + 3 if table_stride is None else table_stride, 1), 3, 4),
        k=view((pages, page, 1, 64), (page * 64 + 19, 64, 64, 1), 7),
        v=view((pages, page, 1, 512), (page * 512 + 31, 512, 512, 1), 13),
    )


def with_buffers(item, lengths=None):
    item = deepcopy(item)
    batch, width = item["table"]["sizes"]
    stride = item["table"]["strides"][0]
    page, pages = item["k"]["sizes"][1], item["k"]["sizes"][0]
    values = [-1] * (0 if not batch or not width else (batch - 1) * stride + width)
    # Fill the logical table, including shared rows, then poison only columns
    # unused by every row. Duplicated and permuted pages are intentional.
    lengths = (
        [min(width * page, (2 * page + 3, 0, page - 1)[b % 3]) for b in range(batch)]
        if lengths is None
        else lengths
    )
    for b, length in enumerate(lengths):
        for col in range((length + page - 1) // page):
            values[b * stride + col] = (4, 1, 4, 0)[col % 4] % max(pages, 1)
    item.update(values=values, lengths=list(lengths))
    return item


def run_rows(binary, rows):
    text = "".join(" ".join(map(str, row)) + "\n" for row in rows)
    label = binary.name + "-" + hashlib.sha256(text.encode()).hexdigest()[:20]
    (binary.parent / (label + ".input")).write_text(text)
    result = subprocess.run([str(binary)], input=text, capture_output=True, text=True, timeout=60)
    (binary.parent / (label + ".output")).write_text(result.stdout)
    (binary.parent / (label + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    result = [list(map(int, line.split(","))) for line in result.stdout.splitlines()]
    assert len(result) == len(rows)
    return result


def evaluate(binary, cases, operation=0):
    rows = []
    for item in cases:
        row = [operation]
        for name in ("table", "k", "v"):
            x = item[name]
            row += [*x["sizes"], *x["strides"], x["offset"], x["storage"]]
        if operation:
            values = item["values"]
            row += [
                len(values),
                item.get("available", len(values)),
                int(item.get("has_table", True)),
                *values,
            ]
            if operation == 1:
                lengths = item["lengths"]
                row += [
                    len(lengths),
                    item.get("length_count", len(lengths)),
                    int(item.get("has_lengths", True)),
                    *lengths,
                ]
            else:
                row += item["range"]  # batch, effective length, start, count
        rows.append(row)
    return run_rows(binary, rows)


def view_oracle(x, element_bytes):
    sizes, strides, offset = x["sizes"], x["strides"], x["offset"]
    if any(s < 0 for s in strides[:-1]) or strides[-1] != 1:
        return 2, 0
    if offset < 0:
        return 3, 0
    span = 0 if 0 in sizes else 1 + sum((n - 1) * s for n, s in zip(sizes, strides))
    if span and element_bytes * (offset + span) > I64_MAX:
        return 4, 0
    if span and element_bytes * (offset + span) > x["storage"]:
        return 3, 0
    return 0, span


def layout_oracle(item):
    table, k, v = (item[n] for n in ("table", "k", "v"))
    batch, width = table["sizes"]
    pages, page, heads, dim = k["sizes"]

    def fail(status):
        return [status, *([17] * 15)]

    if not (
        0 <= pages <= 1 << 31
        and 16 <= page <= 1024
        and page % 16 == 0
        and heads == 1
        and dim == 64
        and v["sizes"] == [pages, page, 1, 512]
    ):
        return fail(1)
    if min(batch, width) < 0:
        return fail(1)
    ts, tspan = view_oracle(table, 4)
    if ts:
        return fail(ts)
    if width * page > I64_MAX:
        return fail(4)
    ks, kspan = view_oracle(k, 2)
    vs, vspan = view_oracle(v, 2)
    if ks or vs:
        return fail(ks or vs)
    return [
        0,
        batch,
        width,
        width * page,
        pages,
        page,
        table["strides"][0],
        tspan,
        *k["strides"][:3],
        kspan,
        *v["strides"][:3],
        vspan,
    ]


def buffer_valid(item):
    _, span = view_oracle(item["table"], 4)
    return item.get("available", len(item["values"])) >= span and (
        not span or item.get("has_table", True)
    )


def pages_oracle(item):
    batch = item["table"]["sizes"][0]
    if (
        not buffer_valid(item)
        or item.get("length_count", len(item["lengths"])) != batch
        or (batch and not item.get("has_lengths", True))
    ):
        return [1]
    page = item["k"]["sizes"][1]
    capacity = item["table"]["sizes"][1] * page
    for b, length in enumerate(item["lengths"][:batch]):
        if not 0 <= length <= capacity:
            return [2]
        # Walk scalar logical tokens, independent of the production ceil loop.
        for t in range(length):
            entry = item["values"][b * item["table"]["strides"][0] + t // page]
            if not 0 <= entry < item["k"]["sizes"][0]:
                return [3]
    return [0]


def segment_oracle(item):
    def fail(status):
        return [status, *([17] * 6)]

    if not buffer_valid(item):
        return fail(1)
    b, length, start, count = item["range"]
    page = item["k"]["sizes"][1]
    if not 0 <= length <= item["table"]["sizes"][1] * page:
        return fail(2)
    if not (b < item["table"]["sizes"][0] and start <= length and start + count <= length):
        return fail(4)
    if not count:
        return [0] * 7
    slot = b * item["table"]["strides"][0] + start // page
    physical = item["values"][slot]
    if not 0 <= physical < item["k"]["sizes"][0]:
        return fail(3)
    # Scalar enumeration stops at the first different logical page, including
    # when that next page happens to map to the same physical ID.
    tokens = list(
        itertools.takewhile(lambda t: t // page == start // page, range(start, start + count))
    )
    offsets = [
        sum(a * s for a, s in zip((physical, start % page, 0, 0), item[n]["strides"]))
        for n in ("k", "v")
    ]
    return [0, slot, physical, start % page, len(tokens), *offsets]


def check(binary, cases, operation=0):
    oracle = (layout_oracle, pages_oracle, segment_oracle)[operation]
    actual = evaluate(binary, cases, operation)
    for item, result in zip(cases, actual):
        assert result == oracle(item), item
    return actual


def source_case(item, copy_range, kd, vd, kw, vw):
    b, start, count = copy_range
    return [
        item["k"]["sizes"][1],
        item["table"]["strides"][0],
        *[item[n]["strides"][i] for n in ("k", "v") for i in (0, 1)],
        b,
        start,
        count,
        kd,
        vd,
        kw,
        vw,
        len(item["values"]),
        *item["values"],
    ]


def trace_oracle(segments, item, kd, vd, kw, vw):
    qk, pv, done = [], [], 0
    for _, _, _, _, rows, ko, vo in segments:
        rope = [0, done * 16, 0, ko + kd, rows, kw, item["k"]["strides"][1], 512]
        content = [1, done * 16, 1, vo + vd, rows, vw, item["v"]["strides"][1], 512]
        qk += [content, rope]
        pv += [content]
        done += rows
    return [
        len(segments),
        len(segments),
        3 * len(segments),
        *itertools.chain.from_iterable(qk + pv),
    ]


@pytest.mark.parametrize("page", range(16, 1025, 16))
@pytest.mark.parametrize("strided", [False, True])
def test_all_page_sizes_source_and_scalar(paged_drivers, page, strided):
    item = case(page, width=8)
    if strided:
        item["k"] = view(item["k"]["sizes"], (page * 151 + 19, 151, I64_MAX, 1), 17)
        item["v"] = view(item["v"]["sizes"], (page * 1103 + 31, 1103, I64_MAX, 1), 31)
    item = with_buffers(item, [5 * page - 1, 0, 2 * page + 3])
    check(paged_drivers[0], [item])
    check(paged_drivers[0], [item], 1)
    ranges = [
        (0, 0, 0),
        (0, 0, 1),
        (0, page - 1, page + 3),
        (0, 2 * page - 1, 2 * page + 1),
        (2, page, page + 3),
    ]
    if 5 * page - 1 >= 1024:
        ranges.append((0, 512, 512))  # Includes P=528 block-internal crossing.
    source_rows, expected = [], []
    for b, start, count in ranges:
        # Determine starts solely from scalar logical page boundaries.
        groups = [
            list(group)
            for _, group in itertools.groupby(range(start, start + count), key=lambda t: t // page)
        ]
        queries = [
            dict(item, range=[b, item["lengths"][b], g[0], start + count - g[0]]) for g in groups
        ]
        segments = check(paged_drivers[0], queries, 2)
        for kd, vd, kw, vw in ((0, 0, 32, 256), (32, 256, 32, 256), (63, 511, 1, 1)):
            source_rows.append(source_case(item, (b, start, count), kd, vd, kw, vw))
            expected.append(trace_oracle(segments, item, kd, vd, kw, vw))
    assert run_rows(paged_drivers[1], source_rows) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["slice", "broadcast", "overlap"])
def test_real_independent_storage_and_table_views(paged_drivers, dtype, kind):
    page, pages, batch, width = 112, 7, 3, 5
    tensors = []
    for dim, off in ((64, 7), (512, 13)):
        if kind == "slice":
            backing = torch.empty((2 * pages + 1, 2 * page + 1, dim + 31), dtype=dtype)
            x = backing[1::2, 1::2, off : off + dim].unsqueeze(2)
        elif kind == "broadcast":
            x = torch.empty(dim + off, dtype=dtype).as_strided(
                (pages, page, 1, dim), (0, 0, I64_MAX, 1), off
            )
        else:
            x = torch.empty(dim + off + pages * 11 + page * 3, dtype=dtype).as_strided(
                (pages, page, 1, dim), (11, 3, I64_MAX, 1), off
            )
        tensors.append(x)
    table = torch.empty(64, dtype=torch.int32).as_strided(
        (batch, width), (0 if kind == "broadcast" else 2 if kind == "overlap" else 11, 1), 5
    )
    item = case(page)
    for name, x in zip(("table", "k", "v"), (table, *tensors)):
        item[name] = dict(
            sizes=list(x.shape),
            strides=list(x.stride()),
            offset=x.storage_offset(),
            storage=x.untyped_storage().nbytes(),
        )
    item = with_buffers(item, [page + 3] * batch)
    flat = torch.empty(0, dtype=torch.int32).set_(
        table.untyped_storage(), table.storage_offset(), (len(item["values"]),), (1,)
    )
    flat.copy_(torch.tensor(item["values"], dtype=torch.int32))
    check(paged_drivers[0], [item])
    check(paged_drivers[0], [item], 1)
    queries = [
        dict(item, range=[b, item["lengths"][b], t, 1])
        for b in range(batch)
        for t in (0, page - 1, page, page + 2)
    ]
    for query, segment in zip(queries, check(paged_drivers[0], queries, 2)):
        _, slot, physical, row, _, ko, vo = segment
        b, _, start, _ = query["range"]
        assert table[b, start // page].data_ptr() == table.data_ptr() + 4 * slot
        assert table[b, start // page].item() == physical
        for x, offset in zip(tensors, (ko, vo)):
            for d in (0, 1, x.shape[-1] - 1):
                assert x[physical, row, 0, d].data_ptr() == x.data_ptr() + 2 * (offset + d)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize(
    "stride", [0, 1, 65535, 65536, 65537, (1 << 32) - 1, 1 << 32, (1 << 32) + 1, I64_MAX]
)
def test_wide_cache_strides_and_offsets(paged_drivers, axis, stride):
    items = []
    for name in ("k", "v"):
        item = with_buffers(case())
        item[name]["strides"][axis] = stride
        item[name]["storage"] = U64_MAX
        items.append(item)
    actual = check(paged_drivers[0], items)
    for item, result in zip(items, actual):
        if result[0] == 0:
            query = dict(item, range=[0, item["lengths"][0], 17, 5])
            check(paged_drivers[0], [item], 1)
            segments = check(paged_drivers[0], [query], 2)
            assert run_rows(paged_drivers[1], [source_case(item, (0, 17, 5), 0, 0, 64, 512)]) == [
                trace_oracle(segments, item, 0, 0, 64, 512)
            ]


@pytest.mark.parametrize("stride", [0, 1, 65535, 65536, 65537])
def test_table_row_stride_source_and_scalar(paged_drivers, stride):
    item = with_buffers(case(table_stride=stride))
    check(paged_drivers[0], [item])
    check(paged_drivers[0], [item], 1)
    query = dict(item, range=[2, 15, 1, 13])
    segments = check(paged_drivers[0], [query], 2)
    assert run_rows(paged_drivers[1], [source_case(item, (2, 1, 13), 32, 256, 32, 256)]) == [
        trace_oracle(segments, item, 32, 256, 32, 256)
    ]


def test_unused_singleton_strides_and_separate_capacities(paged_drivers):
    item = case(batch=1, pages=1, table_stride=I64_MAX)
    for name in ("k", "v"):
        item[name]["strides"][0] = I64_MAX
        item[name]["strides"][2] = I64_MAX
    item = with_buffers(item, [35])
    check(paged_drivers[0], [item])
    check(paged_drivers[0], [item], 1)
    check(paged_drivers[0], [dict(item, range=[0, 35, 32, 3])], 2)
    # The address component has neither the schedule capacity cap nor a
    # requirement that the physical page pool cover the logical capacity.
    item = with_buffers(case(width=1025, pages=3), [17, 0, 16])
    result = check(paged_drivers[0], [item])[0]
    assert result[3:5] == [16400, 3]
    check(paged_drivers[0], [item], 1)


def test_invalid_shape_stride_storage_and_overflow(paged_drivers):
    items = []
    for name, field, axis, values in [
        ("table", "sizes", 0, [-1]),
        ("table", "sizes", 1, [-1]),
        ("k", "sizes", 0, [-1, (1 << 31) + 1]),
        ("k", "sizes", 1, [-16, 0, 1, 15, 17, 1025, I64_MAX]),
        ("k", "sizes", 2, [0, 2]),
        ("v", "sizes", 2, [0, 2]),
        ("k", "sizes", 3, [512]),
        ("v", "sizes", 3, [64]),
        ("v", "sizes", 0, [6]),
        ("v", "sizes", 1, [32]),
        ("table", "strides", 0, [-1]),
        ("table", "strides", 1, [0, 2, -1]),
    ]:
        for value in values:
            item = case()
            item[name][field][axis] = value
            items.append(item)
    for name in ("table", "k", "v"):
        for offset in (-1, I64_MAX):
            item = case()
            item[name].update(offset=offset, storage=U64_MAX)
            items.append(item)
        item = case()
        item[name]["storage"] -= 1  # Whole allocation, not just effective pages.
        items.append(item)
        for axis in range(len(item[name]["strides"])):
            item = case()
            item[name]["strides"][axis] = -1
            items.append(item)
    for stride in (65535, 65536, 65537, (1 << 32) + 1, I64_MAX // 4, I64_MAX):
        item = case(table_stride=stride)
        item["table"]["storage"] = U64_MAX
        items.append(item)
    for batch in (0, 1, 2):
        for width in (I64_MAX // 16, I64_MAX // 16 + 1, I64_MAX):
            item = case(batch=batch, width=width, table_stride=0)
            item["table"]["storage"] = U64_MAX
            items.append(item)
    for name, bytes_per_element in (("table", 4), ("k", 2), ("v", 2)):
        for delta in (-1, 0, 1):
            item = case()
            _, span = view_oracle(item[name], bytes_per_element)
            item[name].update(offset=I64_MAX // bytes_per_element - span + delta, storage=U64_MAX)
            items.append(item)
    check(paged_drivers[0], items)


def test_dynamic_rejection_padding_and_immutable_failure(paged_drivers):
    base = with_buffers(case())
    items = [base]
    for value in (-2147483648, -1, 7, 2147483647):
        item = deepcopy(base)
        item["values"][2] = value  # Final effective page, beyond first copy range.
        items.append(item)
        item = deepcopy(base)
        item["values"][3] = value  # Padding is never interpreted as a page ID.
        items.append(item)
    for length in (-1, 81, 2147483647):
        item = deepcopy(base)
        item["lengths"][1] = length  # Zero-query/zero-KV peer does not suppress validation.
        items.append(item)
    items += [
        dict(base, has_table=False),
        dict(base, available=len(base["values"]) - 1),
        dict(base, has_lengths=False),
        dict(base, length_count=2),
        dict(base, lengths=[*base["lengths"], 0]),
    ]
    check(paged_drivers[0], items, 1)
    ranges = [
        (0, 35, 0, 35),
        (0, 35, 35, 0),
        (0, 0, 0, 0),
        (3, 35, 0, 1),
        (U64_MAX, 35, 0, 0),
        (0, -1, 0, 0),
        (0, 81, 0, 0),
        (0, 35, 36, 0),
        (0, 35, 0, 36),
        (0, 35, 1, U64_MAX),
        (0, 35, U64_MAX, 1),
    ]
    queries = [dict(base, range=r) for r in ranges]
    for item in items:
        queries.append(dict(item, range=[0, 35, 32, 3]))
    check(paged_drivers[0], queries, 2)


def test_empty_and_signed_page_extremes(paged_drivers):
    items = []
    for batch, width, pages in itertools.product((0, 3), (0, 5), (0, 7)):
        item = with_buffers(case(batch=batch, width=width, pages=pages), [0] * batch)
        for name in ("table", "k", "v"):
            if 0 in item[name]["sizes"]:
                item[name].update(offset=I64_MAX, storage=0)
                item[name]["strides"][0] = I64_MAX
        items.append(item)
    check(paged_drivers[0], items)
    check(paged_drivers[0], items, 1)
    queries = [dict(item, range=[0, 0, 0, 0]) for item in items if item["table"]["sizes"][0]]
    check(paged_drivers[0], queries, 2)
    # Maximum signed page ID is valid with 2^31 physical pages. Broadcast
    # storage avoids any giant allocation; the pool has no 16384-row cap.
    item = case(pages=1 << 31)
    for name, dim in (("k", 64), ("v", 512)):
        item[name] = view((1 << 31, 16, 1, dim), (0, dim, I64_MAX, 1), 1)
    item = with_buffers(item, [1, 0, 0])
    item["values"][0] = (1 << 31) - 1
    check(paged_drivers[0], [item])
    check(paged_drivers[0], [item], 1)
    check(paged_drivers[0], [dict(item, range=[0, 1, 0, 1])], 2)
    # Nonempty logical capacity with an empty physical pool only accepts L=0.
    item = with_buffers(case(pages=0), [1, 0, 0])
    assert check(paged_drivers[0], [item], 1) == [[3]]


def test_randomized_big_integer_and_scalar_oracles(paged_drivers):
    rng = random.Random(71001)
    static, dynamic, queries = [], [], []
    for _ in range(600):
        page = rng.randrange(1, 65) * 16
        item = case(
            page,
            batch=rng.randrange(4),
            width=rng.randrange(8),
            pages=rng.randrange(1, 9),
            table_stride=rng.randrange(10),
        )
        for name in ("k", "v"):
            item[name]["strides"][:3] = [
                rng.choice([0, 1, 65535, 65536, 65537, (1 << 32) + 7, I64_MAX]) for _ in range(3)
            ]
            item[name].update(offset=rng.choice([0, 19, I64_MAX // 2]), storage=U64_MAX)
        static.append(item)
        if layout_oracle(item)[0] != 0:
            continue
        capacity = item["table"]["sizes"][1] * page
        lengths = [rng.randrange(capacity + 1) for _ in range(item["table"]["sizes"][0])]
        item = with_buffers(item, lengths)
        if item["values"] and rng.randrange(2):
            item["values"][rng.randrange(len(item["values"]))] = rng.choice(
                [-1, -2147483648, 2147483647]
            )
        dynamic.append(item)
        for b, length in enumerate(lengths):
            start = rng.randrange(length + 1)
            queries.append(dict(item, range=[b, length, start, rng.randrange(length - start + 1)]))
    check(paged_drivers[0], static)
    check(paged_drivers[0], dynamic, 1)
    check(paged_drivers[0], queries, 2)
