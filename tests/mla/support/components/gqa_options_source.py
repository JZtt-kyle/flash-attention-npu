"""Execute unchanged fixed-source partial/FD on each original logical head group."""

import torch
from tests.mla.support.components import prefill_attention as prefill
from tests.mla.support.components import split_kv_attention as partial
from tests.mla.support.components import split_kv_merge as merge
from tests.mla.support.components.window_attention import WindowParams
from tests.mla.support.public_cases.prefill_tnd import compact


def logical_inputs(c):
    b, sq, h, sk, hk = c["shape"]
    result = {}
    for name, value in c["raw"].items():
        if value.ndim == 4:
            result[name] = value
            continue
        query = name in ("q", "qv")
        seq, heads = (sq, h) if query else (sk, hk)
        dst = value.new_zeros((b, seq, heads, value.shape[-1]))
        start = 0
        for i, length in enumerate(c["q"] if query else c["k"]):
            dst[i, :length] = value[start : start + length]
            start += length
        result[name] = dst
    return result


def fragment_outputs(probes, spec, raw, tasks, window):
    if probes["representation"] == "prefill":
        case = prefill.make_case(
            dict(spec, shape=(*spec["shape"], 1)), probes["dtype"], probes["libs"]["cores"]
        )
        case["tasks"] = tasks
        case["params"].window = WindowParams(*window)
        for name, value in raw.items():
            case["native"][name].copy_(value.npu())
        chunks = [[], [], []]
        for first in range(0, len(tasks), case["capacity"]):
            descriptors = prefill.wave(case, first)
            for tensor, value in zip(case["raw"]["oracle"], (0, 0, -torch.inf)):
                tensor.fill_(value)
            assert (
                prefill.launch_case(probes["libs"], case, "oracle", descriptors, partial=True) == 0
            )
            torch.npu.synchronize()
            for dest, tensor in zip(chunks, case["raw"]["oracle"]):
                dest.append(tensor[: case["params"].cores].cpu())
        raw_parts = [torch.cat(x) for x in chunks]
        merged = prefill.prepare_merge(case, dict(oracle=raw_parts, migrated=raw_parts))
        assert prefill.launch_merge(probes["libs"], merged, "oracle") == 0
    else:
        case = partial.make_case(probes["partial"], spec)
        assert case["schedule"].cores == len(tasks)
        case["oracle_partial"].window = WindowParams(*window)
        for i, (task, _, _) in enumerate(tasks):
            case["oracle_partial"].tasks[i] = partial.Task(*task)
        for name, value in raw.items():
            case["device"][name].copy_(value.npu())
        assert partial.launch(probes["partial"], case, "oracle") == 0
        torch.npu.synchronize()
        raw_parts = partial.raw_results(case, "oracle")
        merged = merge.allocate_case(
            dict(dtype=probes["torch_dtype"]),
            spec,
            tasks,
            dict(oracle=raw_parts, migrated=raw_parts),
        )
        assert merge.launch(dict(libs=probes["libs"]), merged, "oracle") == 0
    torch.npu.synchronize()
    output, lse = [x.cpu() for x in merged["outputs"]["oracle"]]
    lse[(lse == -torch.finfo(torch.float32).max) | lse.isposinf()] = -torch.inf
    return output[0], lse[0], raw_parts


def execute(probes, c, record):
    b, sq, h, sk, hk = c["shape"]
    r, o = record.cpu().tolist(), c["options"]
    m, n, groups = r[25], r[26], r[28]
    g, dim = h // groups, c["t"]["v"].shape[-1]
    cap = o["num_splits"] or 26
    raw = logical_inputs(c)
    expected = [
        raw["q"].new_zeros((b, sq, h, dim)),
        torch.full((b, h, sq), -torch.inf),
        torch.zeros((cap, b, h, sq, dim)),
        torch.full((cap, b, h, sq), -torch.inf),
    ]
    tail = 32 + 8 * b
    count = r[tail + 3]
    ends = list(
        zip(
            r[tail + 4 : tail + 4 + count],
            r[tail + 30 : tail + 30 + count],
            r[tail + 56 : tail + 56 + count],
        )
    )
    window = r[22:25]
    saved = []
    for batch, (q, k) in enumerate(zip(c["qlens"], c["klens"])):
        assert 0 < q * g <= m and k > 0
        queries = torch.arange(q * g) // g
        keys = torch.arange(k)
        center = queries[:, None] + k - q
        visible = torch.ones((q * g, k), dtype=torch.bool)
        if window[0] == 1:
            visible &= keys <= center
        elif window[0] == 2:
            visible &= (keys >= center - window[1]) & (keys <= center + window[2])
        union = visible.any(0).nonzero().flatten()
        first, last = (int(union[0]) // n, (int(union[-1]) + n) // n) if union.numel() else (0, 0)
        for group in range(groups):
            if o["num_splits"]:
                span = (last - first + cap - 1) // cap
                intervals = [
                    (min(first + s * span, last), min(first + (s + 1) * span, last))
                    for s in range(cap)
                ]
            else:
                owned = {}
                for tile in range(first, last):
                    owner = next(
                        i for i, end in enumerate(ends) if (batch * groups + group, 0, tile) < end
                    )
                    owned.setdefault(owner, []).append(tile)
                intervals = [(v[0], v[-1] + 1) for v in owned.values()] or [(0, 0)]
            tasks = [
                ((0, 0, s, lo, hi), 0, visible & (keys >= lo * n) & (keys < hi * n))
                for s, (lo, hi) in enumerate(intervals)
            ]
            head, kh = group * g, group * g // (h // hk)
            logical = {
                name: value[batch : batch + 1, :q, head : head + g].contiguous()
                if name in ("q", "qv")
                else value[batch : batch + 1, :, kh : kh + 1].contiguous()
                for name, value in raw.items()
            }
            spec = dict(
                shape=(1, q, g, sk),
                lengths=[k],
                splits=len(tasks),
                m=m,
                n=n,
                scale=o["softmax_scale"],
                causal=o["causal"],
                window=o["window_size"],
            )
            output, lse, parts = fragment_outputs(probes, spec, logical, tasks, window)
            expected[0][batch, :q, head : head + g] = output
            expected[1][batch, head : head + g, :q] = lse
            for split, (_, _, mask) in enumerate(tasks):
                valid = mask.any(-1)
                values = parts[0][split, : q * g].clone()
                values[~valid] = 0
                logs = parts[1][split, : q * g, 0].log() + parts[2][split, : q * g, 0]
                logs[~valid] = -torch.inf
                expected[2][split, batch, head : head + g, :q] = values.reshape(
                    q, g, dim
                ).transpose(0, 1)
                expected[3][split, batch, head : head + g, :q] = logs.reshape(q, g).T
            saved.append(dict(batch=batch, group=group, intervals=intervals, raw=parts))
    if c["t"]["q"].ndim == 3:
        expected = [
            compact(c, expected[0], 0),
            compact(c, expected[1].permute(1, 0, 2), 1),
            compact(c, expected[2].permute(0, 2, 1, 3, 4), 2),
            compact(c, expected[3].permute(0, 2, 1, 3), 2),
        ]
    return expected, saved
