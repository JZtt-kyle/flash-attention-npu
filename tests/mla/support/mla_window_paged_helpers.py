"""PA window probes and raw-input adapters, independent of production dispatch."""

import ctypes as ct

import torch

from tests.mla.support.mla_window_paged_reference import references as references, window_for
from tests.mla.support.components import paged_attention as base
from tests.mla.support.public_cases import paged_bsnd_production as production
from tests.mla.support.components import paged_metadata_aicpu as metadata_base

config = base.config
valid_outputs = base.valid_outputs
check_math = base.check_math
configure_producer = base.configure_producer
launch = base.launch


class Window(ct.Structure):
    _fields_ = [("kind", ct.c_uint32), ("left", ct.c_int64), ("right", ct.c_int64)]


class WindowConfig(ct.Structure):
    _fields_ = [(n, ct.c_int64) for n in ("q", "k", "left", "right")] + [("causal", ct.c_bool)]


class WindowSchedule(ct.Structure):
    _fields_ = [("schedule", base.PagedSchedule), ("window", Window)]


def load_attention(directory):
    probes = base.load_attention(directory)
    for name in ("oracle", "migrated"):
        function = probes[0][name].launch_paged_attention
        types = list(function.argtypes)
        types[12] = ct.POINTER(WindowSchedule)
        function.argtypes = types
    return probes


def make_case(probes, producer, spec, **kwargs):
    case = production.make_case(probes, producer, dict(spec, causal=False), **kwargs)
    case["config"].update(window=spec.get("window", (2, 1)), causal=spec.get("causal", False))
    case["causal"] = spec.get("causal", False)
    kind = 1 if probes[0]["dtype"] == torch.float16 else 2
    args = metadata_base.arguments(case["item"], kind)
    args.words += 3
    keep = []
    for name, (values, present, _) in zip(
        metadata_base.ARRAY_NAMES, metadata_base.input_arrays(case["item"])
    ):
        data = (ct.c_int32 * len(values))(*values)
        getattr(args, name).address = ct.addressof(data) if present else 0
        keep.append(data)
    cfg = WindowConfig(
        max(spec["q"], default=0), case["item"]["sk"], *case["config"]["window"], case["causal"]
    )
    output = (ct.c_uint32 * args.words)()
    function = producer["lib"].window_paged_metadata_host
    function.argtypes = [ct.POINTER(metadata_base.PagedArgs), ct.c_void_p, ct.POINTER(WindowConfig)]
    function.restype = ct.c_uint32
    assert function(ct.byref(args), output, ct.byref(cfg)) == 0
    row = list(output)
    version = 16 if spec.get("layout") == 2 else 15
    assert row[:6] == [0x4D4C4131, version, version, 411 + 5 * len(spec["q"]), kind, 0]
    context = base.context_for(row, case["item"], kind)
    assert context.workspace == case["context"].workspace
    window = Window(*window_for(case))
    assert row[-3:] == [int(x) % 2**32 for x in (window.kind, window.left, window.right)]
    schedule = base.PagedSchedule.from_buffer_copy((ct.c_uint32 * 96)(*base.expected_schedule(row)))
    case.update(context=context, expected=row, args=args, schedule=WindowSchedule(schedule, window))
    storage = torch.full((args.words + 2,), base.SENTINEL, dtype=torch.int32, device="npu")
    storage[1:-1].copy_(
        torch.tensor([x if x < 2**31 else x - 2**32 for x in row], dtype=torch.int32)
    )
    case.update(metadata_storage=storage, metadata=storage[1:-1])
    case["mask"] = (
        torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu(1) if window.kind else None
    )
    return case
