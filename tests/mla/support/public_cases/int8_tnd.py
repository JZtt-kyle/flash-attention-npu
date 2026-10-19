"""TND INT8 allocation and strided-storage fixtures."""

import torch
from tests.mla.support.mla_int8_cases import SOURCE_CASES
from tests.mla.support.mla_int8_layout import preserve_view
from tests.mla.support.public_cases.int8_bsnd import (
    make_case as common_case,
    kwargs as kwargs,
    metadata as metadata,
    invoke as invoke,
    equal as equal,
    check_math as check_math,
    check_source as check_source,
)

TND_CASES = tuple(s for s in SOURCE_CASES if s[1] == "TND")


def make_case(spec=SOURCE_CASES[12], view="contiguous", splits=0, uniform=False):
    source_spec = spec if spec[1] == "BSND" else None
    if source_spec is not None:
        spec = ("tnd-" + spec[0], "TND", *spec[2:])
    case = common_case(spec, view, splits, uniform)
    if source_spec is not None:
        case["source_spec"] = source_spec
    if view == "allocation-gap":
        for name in ("q", "qv", "qs"):
            raw = case["raw"][name]
            backing = torch.full(
                (raw.shape[0] * 2 + 2, *raw.shape[1:]),
                117 if raw.dtype == torch.int8 else float("nan"),
                dtype=raw.dtype,
            )
            value = backing[1:-1:2]
            value.copy_(raw)
            case["raw"][name] = value
            case["tensors"][name] = preserve_view(value)
    return case
