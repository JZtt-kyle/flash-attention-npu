"""Independent v3/v4/v5 TND strides; source materialization is outside timing.

The fixed CANN non-FD oracle receives independently packed effective inputs.
The migrated Kernel reads original allocation views. Production entry coverage
is maintained separately in test_flash_attn_npu_v3_mla_tnd_stride_production.py.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.components.bsnd_stride import I64_MAX, View

from tests.mla.support.components.tnd_stride import (
    BOUNDARY_CASES as BOUNDARY_CASES,
    EMPTY_CASES as EMPTY_CASES,
    LAYOUT_CASES as LAYOUT_CASES,
    PERFORMANCE_CASES as PERFORMANCE_CASES,
    WIDE_CASES as WIDE_CASES,
    check_case as check_case,
    configure_producer as configure_producer,
    launch as launch,
    load_attention as load_attention,
    make_case as make_case,
    spec as spec,
    stride_attention as stride_attention,
    stride_producer as stride_producer,
)


@pytest.mark.parametrize("config", LAYOUT_CASES + BOUNDARY_CASES + EMPTY_CASES + PERFORMANCE_CASES)
def test_stride_attention(stride_attention, stride_producer, config):
    check_case(stride_attention, make_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("config", WIDE_CASES)
def test_wide_active_stride(stride_attention, stride_producer, config):
    torch.npu.empty_cache()
    check_case(stride_attention, make_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("version,layout", ((3, 1), (4, 2), (5, 1), (5, 2)))
@pytest.mark.parametrize("scale", (0.0, -0.125, 0.23))
def test_scale_and_view_reuse(stride_attention, stride_producer, version, layout, scale):
    config = spec(version, layout)
    first = make_case(stride_attention, stride_producer, config, scale=scale)
    check_case(stride_attention, first)
    other = make_case(
        stride_attention, stride_producer, {**config, "profile": "ntd"}, scale=scale + 0.125
    )
    assert other["expected"] == first["expected"]
    other["metadata"], other["metadata_storage"] = first["metadata"], first["metadata_storage"]
    check_case(stride_attention, other)


@pytest.mark.parametrize("version,layout", ((3, 1), (4, 2), (5, 1), (5, 2)))
@pytest.mark.parametrize("field", ("stride", "last_stride", "storage", "shape", "offset"))
def test_invalid_view_rejected_before_launch(
    stride_attention, stride_producer, version, layout, field
):
    case = make_case(stride_attention, stride_producer, spec(version, layout))
    for index in range(4):
        views = (View * 4).from_buffer_copy(case["views"]["migrated"])
        if field == "stride":
            views[index].strides[1] = -1
        elif field == "last_stride":
            views[index].strides[3] = 2
        elif field == "storage":
            views[index].storage_bytes = 0
        elif field == "shape":
            views[index].sizes[1] += 1
        else:
            views[index].offset = I64_MAX
        assert launch(stride_attention, case, views=views) == -3
    assert (case["outputs"]["migrated"][0].cpu() == 12345).all()
