"""Pinned native AIC/FD and partial/merge execution with compact Q and rectangular BSND KV storage."""

import ctypes as ct
import json
import os
from pathlib import Path
import pytest
import torch
from tests.mla.e2e.auto_split_mixed.test_public import (
    prepare,
    schedule_metadata,
    call,
    reference,
    effective,
)
from tests.mla.support.mla_auto_split_mixed_cases import spec, REPRESENTATIVES
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_split_kv_build import compile_drivers
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.unit.auto_split_bsnd.test_source import native_tasks
from tests.mla.unit.split_kv_merge import test_merge as merge
from tests.mla.unit.split_kv_attention import test_attention as partial

merge_probes = merge.merge_probes


@pytest.fixture(scope="module")
def native_partition():
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "native-partition"
    _, native, _ = compile_drivers(directory)
    with CpuDriver(native) as driver:
        yield driver


SOURCE_CASES = REPRESENTATIVES + [
    spec(q=(1, 2), k=(16384, 8193), capacity=16384, h=2, window=(-1, -1)),
    spec(q=(3,) * 32, k=(33, 17) * 16, capacity=33, h=2, window=(-1, -1)),
]


@pytest.mark.parametrize("s", SOURCE_CASES)
def test_fixed_source(merge_probes, native_partition, s):
    p = merge_probes
    c = prepare(p["dtype"], s)
    a = c["args"]
    qs, ks = effective(s)
    heads = s["heads"]
    record = schedule_metadata(c)
    tasks, groups, tile, mask, cores = native_tasks(
        dict(shape=(len(qs), a["max_seqlen_q"], heads, s["capacity"]), qlens=qs, klens=ks),
        record,
        native_partition,
    )
    m, n, ends = tile
    words = record.cpu().tolist()
    qstart = kstart = 0
    raw = []
    for b, (q, k) in enumerate(zip(qs, ks)):
        assert words[410 + b * 8 : 418 + b * 8] == [
            qstart if s["uq"] is None else sum(qs[:b]),
            0,
            kstart,
            0,
            s["q"][b] if s["uq"] is None else q,
            s["kv"][b],
            q,
            k,
        ]
        batch_tasks = [t for t in tasks if t[0] == b]
        for first in range(0, len(batch_tasks), cores):
            wave = batch_tasks[first : first + cores]
            cfg = dict(
                shape=(1, q, heads, s["kv"][b]),
                splits=26,
                m=m,
                n=n,
                cores=len(wave),
                lengths=[k],
                scale=a["softmax_scale"] if a["softmax_scale"] is not None else 576**-0.5,
                window=s["window"],
                causal=s.get("causal", False),
            )
            source = partial.make_case(p["partial"], cfg)
            for name, value in source["device"].items():
                offset = qstart if name in ("q", "qv") else kstart
                value.copy_(c["raw"][name][offset : offset + value.shape[1]].unsqueeze(0))
            for i, (_, block, split, begin, end) in enumerate(wave):
                source["oracle_partial"].tasks[i] = partial.Task(0, block, split, begin, end)
            assert partial.launch(p["partial"], source, "oracle") == 0
            torch.npu.synchronize()
            raw.append(partial.raw_results(source, "oracle"))
        qstart += s["q"][b]
        kstart += s["kv"][b]
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
        qstart = sum(s["q"][:batch])
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
                    accum[split, hi, qstart + qi] = values[0][split, row]
                    alse[split, hi, qstart + qi] = (
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
            expected[0][qstart + qi, hi] = source_o[qi, hi]
            if valid[:, row].any():
                expected[1][hi, qstart + qi] = source_lse[hi, qi]
    refs = reference(c, tile)
    comparisons = [
        assert_mla_source_close(
            actual[i],
            expected[i],
            refs[name + "_high"],
            refs[name + "_low"],
            name="auto-mixed-source-" + name,
        )
        for i, name in enumerate(("out", "lse"))
    ]
    torch.testing.assert_close(actual[2], accum, rtol=0, atol=0)
    torch.testing.assert_close(actual[3], alse, rtol=1e-6, atol=1e-6)
    from tests.common.compare import assert_fa_close

    for i, name in enumerate(("accum", "accum_lse"), 2):
        assert_fa_close(
            actual[i], refs[name + "_high"], refs[name + "_low"], name="auto-mixed-" + name
        )
    # This named tuple specifically exercises all three task kinds. The last
    # case reuses direct slots across groups after the persistent FD region.
    if s == SOURCE_CASES[0]:
        assert kinds == {"direct", "fd", "empty"}
    key = f"mixed-{SOURCE_CASES.index(s)}"
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
