"""Shared helpers extracted from tests/mla/unit/split_kv_attention/test_attention.py.

Independent BSND FD partial buffers, pinned source and two CPU equations.
"""

import ctypes as ct
from itertools import islice
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_ref import normalize_fa3_window
from tests.mla.support.mla_split_kv_attention_reference import fragments
from tests.mla.support.components.dense import Schedule, make_case as dense_case
from tests.mla.support.components.window_attention import WindowParams


class Task(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("batch", "m", "split", "begin", "end")]


class Partial(ct.Structure):
    _fields_ = (
        [("window", WindowParams)]
        + [(name, ct.c_uint64) for name in ("out_offset", "sum_offset", "max_offset", "total")]
        + [("tasks", Task * 26)]
    )


def prepare(probes, spec, first=0):
    batch, sq, heads, sk = spec["shape"]
    lengths = spec.get("lengths", [sk] * batch)
    raw = (ct.c_int64 * batch)(*lengths)
    schedule, partial, details = Schedule(), Partial(), (ct.c_uint64 * 16)()
    assert (
        probes[0]["schedule"].prepare_partial(
            batch,
            sq,
            heads,
            sk,
            raw,
            spec.get("cores", 0),
            spec.get("m", 0),
            spec.get("n", 0),
            spec["splits"],
            first,
            *spec.get("window", (-1, -1)),
            int(spec.get("causal", False)),
            ct.byref(schedule),
            ct.byref(partial),
            details,
        )
        == 0
    )
    return schedule, partial, details


def make_case(probes, spec, first=0):
    schedule, partial, details = prepare(probes, spec, first)
    case = dense_case(
        probes,
        *spec["shape"],
        scale=spec.get("scale"),
        offset=1,
        perturb=spec.get("perturb"),
        schedule_details=(schedule, details),
    )
    case["config"], case["partial"], case["first"] = spec, partial, first
    # Source launch fields come from the raw-input oracle, not CPU builder
    # results. The migrated fields are still independently checked below.
    oracle = Partial.from_buffer_copy(partial)
    _, sq, _, sk = spec["shape"]
    left, right = normalize_fa3_window(
        sq, sk, causal=spec.get("causal", False), window_size=spec.get("window", (-1, -1))
    )
    kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
    oracle.window = WindowParams(kind, sk - 1 if left < 0 else left, sq - 1 if right < 0 else right)
    assert bytes(oracle.window) == bytes(partial.window)
    for index, (task, _, _) in enumerate(
        islice(fragments(spec, schedule.m_base, schedule.s2_base), first, first + schedule.cores)
    ):
        oracle.tasks[index] = Task(*task)
        assert bytes(oracle.tasks[index]) == bytes(partial.tasks[index])
    case["oracle_partial"] = oracle
    case["lengths"] = spec.get("lengths", [spec["shape"][3]] * spec["shape"][0])
    case["device_lengths"] = {
        name: torch.tensor(case["lengths"], dtype=dtype).npu()
        for name, dtype in (("oracle", torch.int64), ("migrated", torch.int32))
    }
    case["mask"] = probes[0]["mask"]
    if spec.get("max_change"):
        case["inputs"]["q"].zero_()
        case["inputs"]["qv"].zero_()
        case["inputs"]["qv"][..., 0] = 32
        case["inputs"]["v"][..., 0, 0] = torch.linspace(-24, 24, spec["shape"][3])
    for b, length in enumerate(case["lengths"]):
        for key in ("k", "v"):
            case["inputs"][key][b, length:] = torch.nan
    for tensor, (before, device, prefix) in zip(case["inputs"].values(), case["storage"][:4]):
        before[prefix : prefix + tensor.numel()] = tensor.flatten()
        device.copy_(before)
    return case


def launch(probes, case, name, *, softcap=0.0):
    d = case["device"]
    pointers = [
        d["qv"],
        d["q"],
        d["v"],
        d["k"],
        case["desc"],
        *case["outputs"][name],
        case["workspace"][name],
    ]
    fallback = case["storage"][0][1].data_ptr()
    function = probes[0][name].launch_softcap if softcap != 0 else probes[0][name].launch_partial
    return function(
        torch.npu.current_stream().npu_stream,
        *(x.data_ptr() if x.numel() else fallback for x in pointers),
        ct.byref(case["schedule"]),
        case["scale"],
        case["device_lengths"][name].data_ptr(),
        case["mask"].data_ptr(),
        ct.byref(case["oracle_partial"] if name == "oracle" else case["partial"]),
        *((softcap,) if softcap != 0 else ()),
    )


def raw_results(case, name):
    s, p, w = case["schedule"], case["partial"], case["workspace"][name]
    result = []
    for offset, columns in ((p.out_offset, 512), (p.sum_offset, 8), (p.max_offset, 8)):
        size = s.cores * s.m_base * columns * 4
        result.append(
            w[offset : offset + size].view(torch.float32).reshape(s.cores, s.m_base, columns).cpu()
        )
    return result


def assert_guards(case):
    for i, (before, device, prefix) in enumerate(case["storage"]):
        after = device.cpu()
        if i < 4:
            torch.testing.assert_close(after, before, rtol=0, atol=0, equal_nan=True)
        else:
            assert torch.equal(after[:prefix], before[:prefix]) and torch.equal(
                after[-16:], before[-16:]
            )
    s, p = case["schedule"], case["partial"]
    for ws in case["ws_storage"].values():
        assert (ws[:512].cpu() == 205).all() and (ws[-512:].cpu() == 205).all()
    # Native capacity reserves two slots per AIC. Never write unassigned slots,
    # unused vec2 scratch or the extra host workspace reserve.
    for ws in case["workspace"].values():
        ranges = [(case["details"][13], p.out_offset)]
        ranges += [
            (p.out_offset + s.cores * s.m_base * 512 * 4, p.sum_offset),
            (p.sum_offset + s.cores * s.m_base * 8 * 4, p.max_offset),
            (p.max_offset + s.cores * s.m_base * 8 * 4, ws.numel()),
        ]
        for start, end in ranges:
            assert (ws[start:end].cpu() == 205).all(), (start, end)
