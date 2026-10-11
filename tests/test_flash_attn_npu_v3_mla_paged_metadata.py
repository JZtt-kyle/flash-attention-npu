"""v6 CPU wire contract, independently decoded against pinned PA source.

The strict test decoder checks every word (including diagnostic fields) against
an independent call context/source oracle. It is not a PA device consumer;
page-table validation and device rejection remain separate migration steps.
"""

import itertools
import json
import os
from pathlib import Path
import random
import subprocess

import pytest

from tests.mla_metadata_build import MLA, ROOT, sha256
from tests.test_flash_attn_npu_v3_mla_paged_schedule import (
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    encode as schedule_encode,
    execute,
    paged_case,
    spans,
)
from tests.test_flash_attn_npu_v3_mla_tnd_schedule import (
    check_tasks,
    expected_spans,
    tnd_case,
)

PREFIX = 408
MAGIC = 0x4D4C4131


def wide(value):
    return [value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF]


@pytest.fixture(scope="module")
def paged_builder(tmp_path_factory):
    target = os.environ.get("MLA_PAGED_METADATA_CPU_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-paged-wire")
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "metadata"
    sources = [
        MLA / name
        for name in (
            "mla_paged_metadata.cpp",
            "mla_metadata.cpp",
            "mla_tnd_schedule.cpp",
            "mla_schedule.cpp",
            "detail/cann_split_core.cpp",
        )
    ] + [ROOT / "tests/mla_paged_metadata_driver.cpp"]
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
        f"-I{MLA}",
        *map(str, sources),
        "-o",
        str(binary),
    ]
    if os.environ.get("MLA_SCHEDULE_SANITIZE") == "1":
        command += [
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer",
        ]
    (directory / "compile.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "build.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    tracked = [*sources, *MLA.glob("*.h"), *MLA.glob("detail/*.h"), Path(__file__)]
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "binary": str(binary.resolve()),
                "binary_sha256": sha256(binary),
                "sources": {str(p.relative_to(ROOT)): sha256(p) for p in tracked},
            },
            indent=2,
        )
        + "\n"
    )
    return binary, directory


def encode(item):
    layout, q = item["layout"], item["q"]
    values = schedule_encode(item)
    result = [
        values[0],
        values[2],
        0 if layout == 0 else 1 if layout == 2 else 99,
        item.get("total_q", sum(q)),
        values[1],
        item["page"],
        values[3],
        *values[4:13],
    ]
    for name, data, present, count_name in (
        (
            "q",
            item.get("cq", list(itertools.accumulate(q, initial=0)) if layout == 2 else []),
            layout == 2,
            "qcount",
        ),
        ("uq", item["uq"] or [], item["uq"] is not None, "uqcount"),
        ("uk", item["uk"] or [], item["uk"] is not None, "ukcount"),
    ):
        result += [
            int(item.get("has_" + name, present)),
            item.get(count_name, len(data)),
            len(data),
            *data,
        ]
    return result


def run_builder(builder, items, dtype=1):
    stdin = "".join(" ".join(map(str, encode(item))) + "\n" for item in items)
    command = [str(builder[0]), str(dtype)]
    result = subprocess.run(command, input=stdin, capture_output=True, text=True, timeout=120)
    capture = builder[1] / "calls"
    capture.mkdir(exist_ok=True)
    path = capture / f"{len(list(capture.glob('*.json'))):04d}.json"
    path.write_text(
        json.dumps(
            {
                "command": command,
                "stdin": stdin,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        )
        + "\n"
    )
    assert result.returncode == 0 and not result.stderr, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == len(items)
    return rows


def expected_record(item, dtype, source):
    """Numerical decoding specification; never calls migrated normalization/serialization."""
    values = schedule_encode(item)
    batch, sq, heads, capacity, aic, aiv = values[:6]
    q, kv = item["q"], item["uk"]
    s1 = max(q, default=0) if item["layout"] == 2 else sq
    s2 = (max(kv, default=0) + item["page"] - 1) // item["page"] * item["page"]
    empty = batch == 0 or s1 == 0 or s2 == 0
    assert 16 <= item["page"] <= 1024 and item["page"] % 16 == 0
    assert 0 <= capacity <= 16384 and capacity % item["page"] == 0
    assert len(q) == len(kv) == batch and all(0 <= length <= capacity for length in kv)
    if empty:
        assert source is None
        s = [0] * 356
        s[3], s[16], s[17] = 1, values[12] or 1, 1
    else:
        assert source[0] == 0
        s = source
        assert s[356:-4] == spans(item)
        assert s[-4:] == [item["page"], capacity, s1, s2]
    record = [MAGIC, 6, 6, PREFIX + batch * 5, dtype, 0, 0, 0]
    record += [batch, 1, heads, s1, s2, 512, 64, s[1], s[2], aic, aiv, aiv // aic, s[3], s[16], 1]
    record += s[18:96] + wide(s[4]) + s[5:9] + s[96:356]
    r1, r2 = s[9:11]
    slots = 2 * s[16]
    p = slots * r1 * 4
    pv = p + slots * r1 * 2
    vec2 = pv + slots * r2 * 4
    kernel = vec2 + slots * r2 * 4
    record += list(
        itertools.chain.from_iterable(
            wide(v) for v in [*s[9:13], 0, p, pv, vec2, kernel, *s[13:16]]
        )
    )
    memory = [0] * 5 if empty else [516096, 178176 if aiv == aic else 174080, 65536, 65536, 131072]
    record += list(itertools.chain.from_iterable(wide(v) for v in memory))
    assert len(record) == 401
    record += [
        int(item["layout"] == 2),
        *wide(item.get("total_q", sum(q))),
        sq,
        int(item["uq"] is not None),
        item["page"],
        capacity,
    ]
    query_used = q if item["uq"] is None else item["uq"]
    starts = list(itertools.accumulate(q, initial=0))
    for b in range(batch):
        assert 0 <= query_used[b] <= q[b] <= sq
        record += [*wide(starts[b]), q[b], query_used[b], kv[b]]
    return record


def decode_record(record, item, dtype, source):
    """Strict independent test decoder: full wire equality, not just round-trip."""
    expected = expected_record(item, dtype, source)
    if len(record) != len(expected):
        raise ValueError("word count")
    for index, (actual, want) in enumerate(zip(record, expected)):
        if actual != want:
            raise ValueError(f"word {index}: {actual} != {want}")
    return {
        "page": record[406],
        "capacity": record[407],
        "source_s2": record[12],
        "spans": record[PREFIX:],
    }


def check_source(builder, oracles, items, dtype, name):
    rows = run_builder(builder, items, dtype)
    source = execute(oracles[0], items)
    native = execute(oracles[0], items, native=True)
    migrated_native = execute(oracles[1], items, native=True)
    assert native == migrated_native
    counts = {"cases": len(items), "native_non_fd": 0, "fa3_no_split_mapping": 0}
    for row, item, wanted, original in zip(rows, items, source, native):
        decode_record(row, item, dtype, wanted)
        counts["native_non_fd" if original[5] == 0 else "fa3_no_split_mapping"] += 1
        if original[5] == 0:
            assert wanted == original
        compact = tnd_case(
            item["q"] if item["uq"] is None else item["uq"],
            item["uk"],
            heads=item.get("heads", 128),
        )
        check_tasks(compact, [*wanted[:356], *expected_spans(compact)])
    (builder[1] / (name + ".json")).write_text(
        json.dumps(
            {
                **counts,
                "inputs": items,
                "records": rows,
                "source": source,
                "native": native,
            }
        )
        + "\n"
    )
    return rows


@pytest.mark.parametrize(
    "page,layout,dtype", itertools.product(range(16, 1025, 16), (0, 2), (1, 2))
)
def test_all_pages_wire(paged_builder, paged_oracles, page, layout, dtype):
    capacity = 16384 // page * page
    q = [7, 7, 7] if layout == 0 else [7, 0, 3]
    items = [
        paged_case(layout, q, [length, 0, min(17, length)], uq, page, capacity, ratio=ratio)
        for ratio, length, uq in itertools.product(
            (1, 2), sorted({1, page - 1, page + 1, 449, 513, capacity}), (None, [3, 0, 1])
        )
    ]
    check_source(paged_builder, paged_oracles, items, dtype, f"pages-{page}-{layout}-{dtype}")


@pytest.mark.parametrize("aic,ratio", itertools.product(range(1, 27), (1, 2)))
def test_context_cores_and_base_tiles(paged_builder, paged_oracles, aic, ratio):
    items = [
        paged_case(layout, [2] * b, [1025] * b, uq, page, heads=128, aic=aic, ratio=ratio)
        for page, layout, b in itertools.product((112, 528), (0, 2), sorted({1, aic, aic + 1}))
        for uq in (None, [1] * b, [0] * b)
    ]
    check_source(paged_builder, paged_oracles, items, 1, f"cores-{aic}-{ratio}")


@pytest.mark.parametrize("layout,dtype", itertools.product((0, 2), (1, 2)))
def test_empty_snapshots_and_zero_used(paged_builder, layout, dtype):
    items = [
        paged_case(layout, q, kv, uq, capacity=capacity, sq=sq, has_uk=present)
        for q, kv, capacity, sq in (
            ([], [], 224, 7),
            ([0, 0], [17, 0], 224, 0),
            ([7, 7], [0, 0], 224, 7),
            ([7, 7], [0, 0], 0, 7),
        )
        for uq in (None, [0] * len(q), q)
        for present in ((False, True) if not q else (True,))
    ]
    rows = run_builder(paged_builder, items, dtype)
    for row, item in zip(rows, items):
        decode_record(row, item, dtype, None)
        assert row[23:401] == [0] * 378
    # Empty allocation/used never suppresses validation of another length.
    bad = [
        paged_case(layout, [0, 0], [-1, 0], capacity=224),
        paged_case(layout, uq=[0, 0], lengths=[0, 225], capacity=224),
        paged_case(layout, uq=[8, 0], lengths=[0, 0], capacity=0),
    ]
    for row in run_builder(paged_builder, bad, dtype):
        assert row[5:8] == [2, 1, 0] and not any(row[8:])


@pytest.mark.parametrize(
    "updates,status,schedule_status",
    [
        *[({"page": p}, 2, 1) for p in (-1, 0, 15, 17, 1025, 2**63 - 1)],
        ({"sk": 223}, 2, 1),
        ({"sk": 16464}, 2, 1),
        ({"sq": -1}, 2, 1),
        ({"sq": 2**32}, 2, 2),
        ({"heads": 0}, 2, 1),
        ({"heads": 2**23}, 2, 2),
        ({"total_q": -1}, 2, 1),
        ({"total_q": 15}, 2, 1),
        ({"total_q": 2**63 - 1}, 2, 2),
        ({"layout": 1}, 2, 1),
        ({"cq": [1, 7, 14]}, 2, 1),
        ({"cq": [0, 8, 7]}, 2, 1),
        ({"cq": [0, -(2**31), 14]}, 2, 1),
        ({"cq": [0, 7, 13]}, 2, 1),
        ({"uq": [-1, 0]}, 2, 1),
        ({"uq": [8, 0]}, 2, 1),
        ({"uk": [0, -(2**31)]}, 2, 1),
        ({"uk": [0, 225]}, 2, 1),
        ({"has_q": False}, 3, 0),
        ({"qcount": 0}, 3, 0),
        ({"has_uk": False}, 3, 0),
        ({"ukcount": 0}, 3, 0),
        ({"uq": [1], "uqcount": 1}, 3, 0),
        ({"uqcount": 1}, 3, 0),
        ({"aic": 0, "aiv": 0}, 2, 3),
        ({"aic": 27}, 2, 3),
        ({"ratio": 3}, 2, 3),
        ({"block_dim": 21}, 2, 3),
        ({"libapi": 2**64 - 1}, 2, 5),
        ({"l1": 516095}, 2, 4),
    ],
)
def test_failure_clears_entire_payload(paged_builder, updates, status, schedule_status):
    item = dict(paged_case(2, capacity=224), **updates)
    row = run_builder(paged_builder, [item])[0]
    assert row[:8] == [MAGIC, 6, 6, 418, 1, status, schedule_status, 0]
    assert row[8:] == [0] * 410


def test_dtype_and_word_overflow(paged_builder):
    item = paged_case()
    for dtype in (0, 3, 2**32 - 1):
        row = run_builder(paged_builder, [item], dtype)[0]
        assert row[:8] == [MAGIC, 6, 6, 418, dtype, 1, 0, 0] and not any(row[8:])
    for batch in (-1, (2**32 - 1 - PREFIX) // 5 + 1, 2**63 - 1):
        assert run_builder(paged_builder, [dict(item, batch=batch)]) == [[-1]]


def test_buffers_and_old_consumers(paged_builder):
    result = subprocess.run(
        [str(paged_builder[0]), "--check-buffers"], capture_output=True, text=True, timeout=120
    )
    (paged_builder[1] / "buffers.json").write_text(
        json.dumps(
            {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        )
        + "\n"
    )
    assert result.returncode == 0 and not result.stderr, result.stderr


def test_independent_decoder_corruption_and_context(paged_builder, paged_oracles):
    items = [
        paged_case(layout, uq=uq, page=page)
        for layout, uq, page in itertools.product((0, 2), (None, [1, 0]), (112, 528))
    ]
    rows = check_source(paged_builder, paged_oracles, items, 2, "corruption-source")
    sources = execute(paged_oracles[0], items)
    mutations = 0
    for row, item, source in zip(rows, items, sources):
        for offset in range(len(row)):
            damaged = row.copy()
            damaged[offset] ^= 0x80000001
            with pytest.raises(ValueError, match="word"):
                decode_record(damaged, item, 2, source)
            mutations += 1
        for version, kind in itertools.product(range(1, 7), repeat=2):
            if (version, kind) == (6, 6):
                continue
            damaged = row.copy()
            damaged[1:3] = [version, kind]
            with pytest.raises(ValueError, match="word"):
                decode_record(damaged, item, 2, source)
        for damaged in (row[:-1], row + [0], [], row[:401]):
            with pytest.raises(ValueError, match="word count"):
                decode_record(damaged, item, 2, source)
        with pytest.raises(ValueError, match="word 4"):
            decode_record(row, item, 1, source)
    (paged_builder[1] / "corruption.json").write_text(
        json.dumps({"single_word_mutations_rejected": mutations, "records": len(rows)}) + "\n"
    )


def test_lengths_allocation_presence_capacity_and_page_identity(paged_builder, paged_oracles):
    original = paged_case(2, [2] * 10, [511] * 10, page=112, capacity=1120)
    items = [
        original,
        dict(original, uq=[2] * 10),
        dict(original, sq=257),
        dict(original, sk=2240, kv=[2240] * 10),
        dict(original, uk=[510] * 10),
        paged_case(2, [2] * 10, [511] * 10, page=528, capacity=1584),
        paged_case(2, [1, 3] * 5, [511] * 10, page=112, capacity=1120, sq=3),
    ]
    rows = check_source(paged_builder, paged_oracles, items, 1, "identity")
    assert rows[0][408:] == rows[1][408:] and rows[0][405] != rows[1][405]
    assert rows[0][11] == rows[2][11] == 2 and rows[2][404] == 257
    assert rows[0][:407] == rows[3][:407] and rows[0][408:] == rows[3][408:]
    assert rows[0][12] == rows[4][12] and rows[0][408:] != rows[4][408:]
    assert rows[0][402:404] == rows[6][402:404] and rows[0][408:] != rows[6][408:]
    for row, item in zip(rows[1:], items[1:]):
        source = execute(paged_oracles[0], [item])[0]
        with pytest.raises(ValueError):
            decode_record(rows[0], item, 1, source)
        assert row != rows[0]


def test_random_wide_workspace_and_query_starts(paged_builder, paged_oracles):
    rng = random.Random(70703)
    items = []
    for _ in range(120):
        page, layout, batch = rng.randrange(1, 65) * 16, rng.choice((0, 2)), rng.randrange(1, 28)
        capacity = rng.randrange(1, 16384 // page + 1) * page
        q = [rng.choice((1, 7, 17, 257))] * batch
        if layout == 2:
            q[1:] = [rng.randrange(q[0] + 1) for _ in q[1:]]
        lengths = [rng.randrange(1, capacity + 1)] + [rng.randrange(capacity + 1) for _ in q[1:]]
        uq = None if rng.randrange(2) else [rng.randrange(n + 1) for n in q]
        items.append(
            paged_case(
                layout,
                q,
                lengths,
                uq,
                page,
                capacity,
                heads=rng.choice((1, 8, 73, 128, 257)),
                aic=rng.randrange(1, 27),
                ratio=rng.choice((1, 2)),
            )
        )
    rows = check_source(paged_builder, paged_oracles, items, 1, "random")
    budget = rows[0][385] | rows[0][386] << 32
    exact = dict(items[0], libapi=2**64 - 1 - budget)
    row = check_source(paged_builder, paged_oracles, [exact], 1, "wide-workspace")[0]
    assert row[389:391] == [2**32 - 1] * 2
    failed = run_builder(paged_builder, [dict(exact, libapi=exact["libapi"] + 1)])[0]
    assert failed[5:8] == [2, 5, 0] and not any(failed[8:])
    batch = 65538
    wide_query = paged_case(
        0, [65536] * batch, [0] * batch, [0] * batch, page=1024, capacity=16384, heads=1
    )
    row = run_builder(paged_builder, [wide_query])[0]
    decode_record(row, wide_query, 1, None)
    assert row[403] == 1 and row[-4] == 1 and row[-5] == 65536
    full = paged_case(
        2, [2**27] * 15 + [2**27 - 1], [16384] * 16, page=1024, capacity=16384, heads=16
    )
    failed = run_builder(paged_builder, [full])[0]
    assert failed[5:8] == [2, 2, 0] and not any(failed[8:])
    check_source(paged_builder, paged_oracles, [dict(full, uq=[0] * 16)], 2, "zero-tile-bound")
