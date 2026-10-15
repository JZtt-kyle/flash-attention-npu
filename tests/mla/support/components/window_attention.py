"""Shared helpers extracted from tests/mla/unit/window_bsnd/test_flash_attn_npu_v3_mla_window_attention.py.

Independent BSND window component, pinned source and dual CPU math.

The source mask/length arrays are constructed directly from raw case inputs;
no oracle reads the migrated schedule to determine which keys are visible.
"""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_ref import mla_reference_pair, normalize_fa3_window
from tests.mla.support.mla_window_attention_cases import PERFORMANCE_CASES as PERFORMANCE_CASES
from tests.mla.support.components.dense import Schedule, make_case as dense_case


class WindowParams(ct.Structure):
    _fields_ = [("kind", ct.c_uint32), ("left", ct.c_int64), ("right", ct.c_int64)]


def normalized_window(spec):
    _, sq, _, sk = spec["shape"]
    max_q, max_k = spec.get("max_q", sq), spec.get("max_k", sk)
    if min(spec["window"]) < -1:
        # This CPU oracle is already checked against both pinned FA3 host
        # normalizers. Preserve the distinction between -1 and other negative
        # bounds in the single-query causal optimization.
        from tests.mla.support.components.window_schedule import normalized

        return WindowParams(*normalized((max_q, max_k, *spec["window"], spec.get("causal", False))))
    left, right = normalize_fa3_window(
        max_q, max_k, causal=spec.get("causal", False), window_size=spec["window"]
    )
    kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
    return WindowParams(kind, max_k - 1 if left < 0 else left, max_q - 1 if right < 0 else right)


def empty_rows(case, length):
    sq = case["shape"][1]
    if not length:
        return sq
    window = normalized_window(case["config"])
    return (
        0
        if window.kind == 0
        else min(sq, max(0, sq - length - (window.right if window.kind == 2 else 0)))
    )


def make_case(probes, spec):
    batch, sq, heads, sk = spec["shape"]
    lengths = spec.get("lengths", [sk] * batch)
    assert len(lengths) == batch
    raw = (ct.c_int64 * max(1, batch))(*lengths)
    schedule, details, window = Schedule(), (ct.c_uint64 * 16)(), WindowParams()
    assert (
        probes[0]["schedule"].prepare_window(
            batch,
            sq,
            heads,
            sk,
            raw,
            spec.get("cores", 0),
            spec.get("m", 0),
            spec.get("n", 0),
            ct.byref(schedule),
            details,
            spec.get("max_q", sq),
            spec.get("max_k", sk),
            *spec["window"],
            int(spec.get("causal", False)),
            ct.byref(window),
        )
        == 0
    )
    case = dense_case(
        probes,
        batch,
        sq,
        heads,
        sk,
        scale=spec.get("scale"),
        offset=spec.get("offset", 1),
        perturb=spec.get("perturb"),
        amplitude=spec.get("amplitude", 1.0),
        schedule_details=(schedule, details),
    )
    independent = normalized_window(spec)
    assert (window.kind, window.left, window.right) == (
        independent.kind,
        independent.left,
        independent.right,
    )
    case["windows"] = {"migrated": window, "oracle": independent}
    case["config"] = spec
    case["lengths"] = list(lengths)
    case["device_lengths"] = {
        "oracle": torch.tensor(lengths or [0], dtype=torch.int64).npu(),
        "migrated": torch.tensor(lengths or [0], dtype=torch.int32).npu(),
    }
    case["mask"] = probes[0]["mask"]
    if spec.get("max_change"):
        for name in ("q", "qv"):
            case["inputs"][name].zero_()
        case["inputs"]["qv"][..., 0] = 32 * spec.get("amplitude", 1.0)
        for b, length in enumerate(lengths):
            case["inputs"]["v"][b, :length, 0, 0] = torch.linspace(-24, 24, length)
    # Poison the unused physical capacity, independently of the schedule.
    for b, length in enumerate(lengths):
        case["inputs"]["k"][b, length:] = float("nan")
        case["inputs"]["v"][b, length:] = float("nan")
    for tensor, (before, device, prefix) in zip(case["inputs"].values(), case["storage"][:4]):
        before[prefix : prefix + tensor.numel()] = tensor.flatten()
        device.copy_(before)
    return case


def launch(probes, case, name):
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
    # Empty tensors have a null data_ptr; no reads are permitted. The valid
    # backing allocation lets the source perform its zero-length init path.
    fallback = case["storage"][0][1].data_ptr()
    return probes[0][name].launch_window(
        torch.npu.current_stream().npu_stream,
        *(x.data_ptr() if x.numel() else fallback for x in pointers),
        ct.byref(case["schedule"]),
        case["scale"],
        case["device_lengths"][name].data_ptr(),
        case["mask"].data_ptr(),
        ct.byref(case["windows"][name]),
    )


def references(case, workers=4):
    # Samples are independent. Keep the original math and intra-op thread
    # count; bound parallelism and memory without changing dot products.
    def sample(b):
        length = case["lengths"][b]
        inputs = {
            key: value[b : b + 1, :length] if key in ("k", "v") else value[b : b + 1]
            for key, value in case["inputs"].items()
        }
        w = normalized_window(case["config"])
        refs = mla_reference_pair(
            **inputs,
            softmax_scale=case["scale"],
            causal=w.kind == 1,
            window_size=(w.left, w.right) if w.kind == 2 else (-1, -1),
        )
        if length == 0 and case["shape"][3]:
            refs[1].fill_(-torch.inf)
            refs[3].fill_(-torch.inf)
        return refs

    batch, sq, heads, _ = case["shape"]
    result = [
        torch.empty((batch, sq, heads, 512), dtype=case["inputs"]["q"].dtype),
        torch.empty((batch, heads, sq), dtype=torch.float32),
    ]
    result += [torch.empty_like(t) for t in result]
    with ThreadPoolExecutor(max_workers=min(workers, max(1, batch))) as pool:
        for b, refs in enumerate(pool.map(sample, range(batch))):
            for i, ref in enumerate(refs):
                result[i][b : b + 1] = ref
    return result
