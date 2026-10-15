"""Shared helpers extracted from tests/mla/unit/causal_bsnd/test_flash_attn_npu_v3_mla_causal_attention.py.

Independent BSND causal component, pinned source and dual CPU math.

The source mask/length arrays are constructed directly from raw case inputs;
no oracle reads the migrated schedule to determine which keys are visible.
"""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.mla_profile_cases import CASES, S1G_CASES, KV_CASES, LONG_KV_CASES
from tests.mla.support.components.dense import Schedule, make_case as dense_case

# Frozen before sampling. Preserve the four existing capacity matrices as
# causal variants; add effective-length, partial/whole masked and zero cases.
PERFORMANCE_CASES = [
    dict(shape=shape, scale=scale) for shape, scale in CASES + S1G_CASES + KV_CASES + LONG_KV_CASES
] + [
    dict(shape=(2, 7, 16, 33), lengths=[3, 0]),
    dict(shape=(2, 513, 3, 257), lengths=[257, 129]),
    dict(shape=(2, 3, 128, 65), lengths=[0, 0]),
]


def make_case(probes, spec):
    batch, sq, heads, sk = spec["shape"]
    lengths = spec.get("lengths", [sk] * batch)
    assert len(lengths) == batch
    raw = (ct.c_int64 * max(1, batch))(*lengths)
    schedule, details = Schedule(), (ct.c_uint64 * 16)()
    assert (
        probes[0]["schedule"].prepare_causal(
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
    return probes[0][name].launch_causal(
        torch.npu.current_stream().npu_stream,
        *(x.data_ptr() if x.numel() else fallback for x in pointers),
        ct.byref(case["schedule"]),
        case["scale"],
        case["device_lengths"][name].data_ptr(),
        case["mask"].data_ptr(),
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
        refs = mla_reference_pair(**inputs, softmax_scale=case["scale"], causal=True)
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
