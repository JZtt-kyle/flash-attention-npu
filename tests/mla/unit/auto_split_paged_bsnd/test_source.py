"""Pinned native PA/AIC/FD execution against v24, true accumulators and dual math."""

import ctypes as ct
import json
import os
from pathlib import Path
import pytest
import torch
from tests.mla.support.public_cases.auto_split_paged_bsnd_public import (
    prepare,
    metadata,
    call,
    reference,
)
from tests.mla.support.mla_auto_split_paged_bsnd_cases import spec, REPRESENTATIVES
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_split_kv_build import compile_drivers
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.unit.auto_split_bsnd_metadata.test_metadata import visible
from tests.mla.support.mla_auto_split_paged_bsnd_reference import native_schedule
from tests.mla.support.components import split_kv_merge_merge as merge
from tests.mla.support.components import split_kv_attention_attention as partial

from tests.mla.unit.split_paged_bsnd.test_source import paged_source as paged_source

merge_probes = merge.merge_probes


@pytest.fixture(scope="module")
def native_partition():
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "native-partition"
    _, native, _ = compile_drivers(directory)
    from tests.mla.support.components.tnd_schedule import build_oracle

    paged = build_oracle(directory, used=True, paged=True)
    with CpuDriver(native) as driver, CpuDriver(paged) as geometry_source:
        yield driver, geometry_source


SOURCE_CASES = REPRESENTATIVES + [
    spec(page=16, q=3, lengths=(33, 17) * 16, heads=2, window=(-1, -1)),
]


@pytest.mark.parametrize("s", SOURCE_CASES)
def test_fixed_source(paged_source, native_partition, s):
    p, lib = paged_source
    c = prepare(p["dtype"], s)
    a = c["args"]
    qs = [s["q"]] * len(s["lengths"]) if s["used"] is None else s["used"]
    ks = s["lengths"]
    heads = s["heads"]
    record = metadata(c)
    words = record.cpu().tolist()
    m, n = words[15] or 256, words[16] or 512
    cores = words[17]  # Physical AIC count; public accumulator capacity stays 26.
    from tests.mla.support.components.paged_schedule import encode, paged_case

    original = native_partition[1].evaluate(
        [
            encode(
                paged_case(
                    q=[s["q"]] * len(qs),
                    lengths=ks,
                    uq=s["used"],
                    page=s["page"],
                    capacity=s["capacity"],
                    heads=heads,
                    aic=cores,
                    block_dim=cores,
                )
            )
        ]
    )[0]
    assert [m, n, words[12]] == [original[1], original[2], original[-1]]
    source = native_schedule(s, m, n, cores, native_partition[0])
    base = 408 + 5 * len(qs)
    if s["used"] is None:
        assert words[base + 3] == source[0]
        assert words[base + 82 : base + 86] == [*source[2:5], 8]
        pos = 7
        for offset, size in (
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
        ):
            count = cores if size == 26 else cores * 2
            assert words[base + offset : base + offset + size] == (
                source[pos : pos + count] + [0] * (size - count)
            )
            pos += count
    ends = list(zip(*(source[7 + off : 7 + off + source[0]] for off in (0, cores, 2 * cores))))
    mask = tuple(words[base : base + 3])
    tile = m, n, ends
    tasks, groups = [], []
    for batch, (ql, kl) in enumerate(zip(qs, ks)):
        for block in range((ql * heads + m - 1) // m):
            start, stop, _, _, first, last = visible(ql, kl, heads, m, n, block, mask)
            owned = {}
            for j in range(first, last):
                owner = next(i for i, end in enumerate(ends) if (batch, block, j) < tuple(end))
                owned.setdefault(owner, []).append(j)
            indices = []
            for split, tiles in enumerate(owned.values()):
                indices.append(len(tasks))
                tasks.append((batch, block, split, tiles[0], tiles[-1] + 1))
            groups.append((batch, block, start, stop, indices))
    raw = []
    for b, (q, k) in enumerate(zip(qs, ks)):
        batch_tasks = [t for t in tasks if t[0] == b]
        qc = c["raw"]["qv"][b : b + 1, :q].contiguous().npu()
        qr = c["raw"]["q"][b : b + 1, :q].contiguous().npu()
        kv = c["raw"]["v"].contiguous().npu()
        kr = c["raw"]["k"].contiguous().npu()
        table = c["table"][b : b + 1].contiguous().npu()
        desc = torch.tensor(
            [24, 1 << 32, 0xFFFFFFFF, kv.data_ptr()], dtype=torch.int64, device="npu"
        )
        for first in range(0, len(batch_tasks), cores):
            wave = batch_tasks[first : first + cores]
            cfg = dict(
                shape=(1, q, heads, s["capacity"]),
                splits=26,
                m=m,
                n=n,
                cores=len(wave),
                lengths=[k],
                scale=576**-0.5,
                window=s["window"],
                causal=s["causal"],
            )
            item = partial.make_case(p["partial"], cfg)
            for i, (_, block, split, begin, end) in enumerate(wave):
                item["oracle_partial"].tasks[i] = partial.Task(0, block, split, begin, end)
            pointers = [
                qc,
                qr,
                kv,
                kr,
                desc,
                *item["outputs"]["oracle"],
                item["workspace"]["oracle"],
            ]
            assert (
                lib.launch_paged_partial(
                    torch.npu.current_stream().npu_stream,
                    *(x.data_ptr() for x in pointers),
                    ct.byref(item["schedule"]),
                    item["scale"],
                    item["device_lengths"]["oracle"].data_ptr(),
                    item["mask"].data_ptr(),
                    ct.byref(item["oracle_partial"]),
                    s["page"],
                    s["capacity"] // s["page"],
                    table.data_ptr(),
                )
                == 0
            )
            torch.npu.synchronize()
            raw.append(partial.raw_results(item, "oracle"))
            ws = item["ws_storage"]["oracle"]
            assert (ws[:512].cpu() == 205).all() and (ws[-512:].cpu() == 205).all()
    raw = [torch.cat([r[i] for r in raw]) for i in range(3)]
    actual = [t.cpu() for t in call(a, "op")]
    expected = [torch.zeros_like(actual[0]), torch.full_like(actual[1], -torch.inf)]
    accum = torch.zeros_like(actual[2])
    alse = torch.full_like(actual[3], -torch.inf)
    kinds = set()
    for batch, block, start, stop, indices in groups:
        kinds.add("empty" if not indices else "direct" if len(indices) == 1 else "fd")
        if not indices:
            continue
        q, k = qs[batch], ks[batch]
        count = len(indices)
        values = [v[indices].clone() for v in raw]
        valid = torch.zeros((count, m), dtype=torch.bool)
        for split, index in enumerate(indices):
            first, last = tasks[index][-2:]
            for row in range(stop - start):
                qi, hi = divmod(start + row, heads)
                center = qi + k - q
                lo = max(first * n, center - mask[1]) if mask[0] == 2 else first * n
                end = (
                    min(last * n, k, center + (mask[2] if mask[0] == 2 else 0) + 1)
                    if mask[0]
                    else min(last * n, k)
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
        params = merge.Params(1, q, heads, 0, start, stop - start, m, count, k)
        params.adapt = 0
        target = [
            torch.zeros((1, q, heads, 512), dtype=p["dtype"], device="npu"),
            torch.full((1, heads, q), -torch.inf, device="npu"),
        ]
        dev = [v.npu() for v in values]
        assert (
            p["libs"]["oracle"].launch_merge(
                torch.npu.current_stream().npu_stream,
                *(t.data_ptr() for t in [*dev, *target]),
                0,
                0,
                ct.byref(params),
            )
            == 0
        )
        torch.npu.synchronize()
        source_o, source_lse = [t.cpu()[0] for t in target]
        for row in range(stop - start):
            qi, hi = divmod(start + row, heads)
            expected[0][batch, qi, hi] = source_o[qi, hi]
            if valid[:, row].any():
                expected[1][batch, hi, qi] = source_lse[hi, qi]
    refs = reference(c, tile)
    comparisons = [
        assert_mla_source_close(
            actual[i],
            expected[i],
            refs[name + "_high"],
            refs[name + "_low"],
            name="auto-paged-source-" + name,
        )
        for i, name in enumerate(("out", "lse"))
    ]
    torch.testing.assert_close(actual[2], accum, rtol=0, atol=0)
    torch.testing.assert_close(actual[3], alse, rtol=1e-6, atol=1e-6)
    from tests.common.compare import assert_fa_close

    for i, name in enumerate(("accum", "accum_lse"), 2):
        assert_fa_close(
            actual[i], refs[name + "_high"], refs[name + "_low"], name="auto-paged-" + name
        )
    # This named tuple specifically exercises all three task kinds. The last
    # case reuses direct slots across groups after the persistent FD region.
    if s == SOURCE_CASES[1]:
        assert kinds == {"direct", "fd", "empty"}
    key = f"paged-{SOURCE_CASES.index(s)}"
    path = p["directory"] / (key + ".pt")
    torch.save(
        dict(
            spec=s,
            raw=c["raw"],
            record=record.cpu(),
            actual=actual,
            source=expected,
            accumulator=accum,
            accumulator_lse=alse,
            references=refs,
        ),
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(dict(comparisons=comparisons, task_kinds=sorted(kinds)), indent=2) + "\n"
    )
