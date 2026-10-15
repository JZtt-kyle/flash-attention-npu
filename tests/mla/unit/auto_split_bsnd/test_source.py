"""Production v18 versus pinned native partitions and original partial/FD code."""

import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import pytest
import torch
from tests.mla.support.public_cases.auto_split_bsnd_public import (
    make_case,
    metadata,
    invoke,
    geometry,
)
from tests.mla.e2e.auto_split_bsnd.test_public import SPECS
from tests.mla.support.mla_auto_split_bsnd_reference import references
from tests.mla.support.mla_split_kv_build import compile_drivers
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.unit.auto_split_bsnd_metadata.test_metadata import visible
from tests.mla.support.components import split_kv_merge as merge
from tests.mla.support.components import split_kv_attention as partial

merge_probes = merge.merge_probes


@pytest.fixture(scope="module")
def native_partition(tmp_path_factory):
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "native-partition"
    _, native, _ = compile_drivers(directory)
    with CpuDriver(native) as driver:
        yield driver


def native_tasks(c, record, driver):
    b, q, h, k = c["shape"]
    words = record.cpu().tolist()
    base = 410 + 8 * b
    m, n, ends = geometry(record, b)
    cores = words[17]
    kind, left, right = words[base : base + 3]
    source = driver.evaluate(
        [
            [
                1,
                3 if kind == 1 else 4 if kind == 2 else 0,
                2**31 - 1 if kind == 1 else left,
                0 if kind == 1 else right,
                h,
                m,
                n,
                cores,
                2,
                b,
                *itertools.chain.from_iterable(zip(c["qlens"], c["klens"])),
            ]
        ]
    )[0]
    assert words[base + 3] == source[0]
    assert words[base + 82 : base + 86] == [*source[2:5], 8]
    pos = 7
    for offset, size in [
        (4, 26),
        (30, 26),
        (56, 26),
        (86, 26),
        (112, 26),
        (138, 26),
        (164, 26),
        (190, 26),
        (216, 26),
        (242, 52),
        (294, 52),
    ]:
        count = cores if size == 26 else cores * 2
        assert words[base + offset : base + offset + count] == source[pos : pos + count]
        pos += count
    tasks, groups = [], []
    for batch, (ql, kl) in enumerate(zip(c["qlens"], c["klens"])):
        for block in range((ql * h + m - 1) // m):
            start, stop, _, _, first, last = visible(ql, kl, h, m, n, block, (kind, left, right))
            owned = {}
            for tile in range(first, last):
                owner = next(i for i, end in enumerate(ends) if (batch, block, tile) < tuple(end))
                owned.setdefault(owner, []).append(tile)
            group = []
            for split, tiles in enumerate(owned.values()):
                group.append(len(tasks))
                tasks.append((batch, block, split, tiles[0], tiles[-1] + 1))
            groups.append((batch, block, start, stop, group))
    return tasks, groups, (m, n, ends), (kind, left, right), cores


@pytest.mark.parametrize("spec", SPECS)
def test_fixed_source(merge_probes, native_partition, spec):
    p = merge_probes
    dtype = "fp16" if p["dtype"] == torch.float16 else "bf16"
    c = make_case(dtype, **spec)
    record = metadata(c)
    tasks, groups, tile, mask, cores = native_tasks(c, record, native_partition)
    m, n, ends = tile
    raw = [[], [], []]
    o = c["options"]
    for first in range(0, len(tasks), cores):
        wave = tasks[first : first + cores]
        cfg = dict(
            shape=c["shape"],
            splits=26,
            m=m,
            n=n,
            cores=len(wave),
            lengths=c["klens"],
            scale=o["softmax_scale"],
            window=o["window_size"],
            causal=o["causal"],
        )
        case = partial.make_case(p["partial"], cfg)
        for name, value in c["raw"].items():
            case["device"][name].copy_(value)
        for i, task in enumerate(wave):
            case["oracle_partial"].tasks[i] = partial.Task(*task)
        assert partial.launch(p["partial"], case, "oracle") == 0
        torch.npu.synchronize()
        for dest, value in zip(raw, partial.raw_results(case, "oracle")):
            dest.append(value)
    raw = [torch.cat(values) for values in raw]
    actual = [x.cpu() for x in invoke(c, record)]
    expected = [
        torch.zeros_like(actual[0], device="npu"),
        torch.full_like(actual[1], -torch.inf, device="npu"),
    ]
    accum = torch.zeros_like(actual[2])
    alse = torch.full_like(actual[3], -torch.inf)
    empty_rows = []
    for batch, block, start, stop, indices in groups:
        if not indices:
            continue
        count = len(indices)
        values = [v[indices].clone() for v in raw]
        valid = torch.zeros((count, m), dtype=torch.bool)
        for split, index in enumerate(indices):
            a, z = tasks[index][-2:]
            for row in range(stop - start):
                qi, hi = divmod(start + row, c["shape"][2])
                center = qi + c["klens"][batch] - c["qlens"][batch]
                lo = max(a * n, center - mask[1]) if mask[0] == 2 else a * n
                end = (
                    min(z * n, c["klens"][batch], center + (mask[2] if mask[0] == 2 else 0) + 1)
                    if mask[0]
                    else min(z * n, c["klens"][batch])
                )
                valid[split, row] = lo < end
                if lo < end:
                    accum[split, batch, hi, qi] = values[0][split, row]
                    alse[split, batch, hi, qi] = (
                        values[1][split, row, 0].log() + values[2][split, row, 0]
                    )
        values[0][~valid] = 0
        values[1][~valid] = 0
        values[2][~valid] = -torch.finfo(torch.float32).max
        values[1][0, ~valid.any(0)] = 1
        params = merge.Params(
            c["shape"][0],
            c["shape"][1],
            c["shape"][2],
            batch,
            start,
            stop - start,
            m,
            count,
            c["klens"][batch],
        )
        params.adapt = 0
        dev = [x.npu() for x in values]
        assert (
            p["libs"]["oracle"].launch_merge(
                torch.npu.current_stream().npu_stream,
                *(x.data_ptr() for x in [*dev, *expected]),
                0,
                0,
                ct.byref(params),
            )
            == 0
        )
        torch.npu.synchronize()
        for row in range(stop - start):
            if not valid[:, row].any():
                qi, hi = divmod(start + row, c["shape"][2])
                empty_rows.append((batch, hi, qi))
    expected = [x.cpu() for x in expected]
    for idx in empty_rows:
        expected[1][idx] = -torch.inf
    refs = references(
        c["raw"],
        c["qlens"],
        c["klens"],
        ends,
        m,
        n,
        o["causal"],
        o["window_size"],
        o["softmax_scale"],
    )
    comparisons = [
        assert_mla_source_close(
            actual[i], expected[i], refs[name + "_high"], refs[name + "_low"], name="auto-" + name
        )
        for i, name in enumerate(("out", "lse"))
    ]
    torch.testing.assert_close(actual[2], accum, rtol=0, atol=0)
    torch.testing.assert_close(actual[3], alse, rtol=1e-6, atol=1e-6)
    name = "-".join(map(str, c["shape"])) + "-" + spec["mask"]
    path = p["directory"] / (name + ".pt")
    torch.save(
        dict(
            raw=c["raw"],
            actual=actual,
            source=expected,
            accum=accum,
            accum_lse=alse,
            record=record.cpu(),
            references=refs,
        ),
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(comparisons, indent=2) + "\n")


@pytest.mark.parametrize("mask", ["none", "causal", "window"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_used_query_math(native_partition, mask, dtype):
    # BSND base geometry depends on allocation and KV lengths, not Q-used.
    # Obtain only M/N/platform from the public generator; derive all fragments
    # independently with pinned SplitCore using the original Q-used values.
    c = make_case(dtype, shape=(3, 65, 4, 1057), mask=mask, strided=True, used=True)
    c["qlens"] = [65, 4, 0]
    c["usedq"].copy_(torch.tensor(c["qlens"], dtype=torch.int32, device="npu"))
    record = metadata(c).cpu().tolist()
    base = 410 + 8 * c["shape"][0]
    m, n, cores = record[15:18]
    kind, left, right = record[base : base + 3]
    native = native_partition.evaluate(
        [
            [
                1,
                3 if kind == 1 else 4 if kind == 2 else 0,
                2**31 - 1 if kind == 1 else left,
                0 if kind == 1 else right,
                4,
                m,
                n,
                cores,
                2,
                3,
                *itertools.chain.from_iterable(zip(c["qlens"], c["klens"])),
            ]
        ]
    )[0]
    used = native[0]
    ends = list(
        zip(
            native[7 : 7 + used],
            native[7 + cores : 7 + cores + used],
            native[7 + 2 * cores : 7 + 2 * cores + used],
        )
    )
    result = [x.cpu() for x in invoke(c)]
    o = c["options"]
    refs = references(
        c["raw"],
        c["qlens"],
        c["klens"],
        ends,
        m,
        n,
        o["causal"],
        o["window_size"],
        o["softmax_scale"],
    )
    from tests.common.compare import assert_fa_close

    for actual, name in zip(result, ("out", "lse", "accum", "accum_lse")):
        assert_fa_close(actual, refs[name + "_high"], refs[name + "_low"], name="used-auto-" + name)
