"""Independent BSND stride attention, with v1/v2/v5 and fixed-source/math checks.

Source inputs are materialized once before timing; arbitrary FA3 strides have
no matching source ABI. This suite does not open production stride support.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.components.bsnd_stride import (
    BOUNDARY_CASES as BOUNDARY_CASES,
    I64_MAX as I64_MAX,
    LAYOUT_CASES as LAYOUT_CASES,
    NAMES as NAMES,
    PERFORMANCE_CASES as PERFORMANCE_CASES,
    View as View,
    WIDE_CASES as WIDE_CASES,
    check_case as check_case,
    configure_producer as configure_producer,
    describe as describe,
    launch as launch,
    load_attention as load_attention,
    make_case as make_case,
    make_view as make_view,
    spec as spec,
    stride_attention as stride_attention,
    stride_producer as stride_producer,
    strides_for as strides_for,
)


@pytest.mark.parametrize("config", LAYOUT_CASES + BOUNDARY_CASES + PERFORMANCE_CASES)
def test_stride_attention(stride_attention, stride_producer, config):
    check_case(stride_attention, make_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("config", WIDE_CASES)
def test_wide_active_stride(stride_attention, stride_producer, config):
    torch.npu.empty_cache()
    case = make_case(stride_attention, stride_producer, config)
    check_case(stride_attention, case)


@pytest.mark.parametrize(
    "config",
    [spec(5, f"combined_{stride}", (2, 5, 5, 17), [5, 3], [17, 7]) for stride in (65536, 65537)]
    + [spec(5, "independent", uq=[3, 1])],
)
def test_combined_fallback_and_default_kv(stride_attention, stride_producer, config):
    check_case(stride_attention, make_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("scale", (0.0, -0.125, 0.23))
def test_scale_and_input_reuse(stride_attention, stride_producer, scale):
    case = make_case(stride_attention, stride_producer, spec(uq=[3, 1], uk=[17, 0]), scale=scale)
    check_case(stride_attention, case)
    # Reuse the exact record with a different scale and fresh four-way views.
    other = make_case(
        stride_attention,
        stride_producer,
        spec(profile="bnsd", uq=[3, 1], uk=[17, 0]),
        scale=scale + 0.125,
    )
    assert other["expected"] == case["expected"]
    other["metadata"], other["metadata_storage"] = case["metadata"], case["metadata_storage"]
    check_case(stride_attention, other)


@pytest.mark.parametrize("field", ("stride", "storage", "shape", "offset"))
def test_invalid_view_rejected_before_launch(stride_attention, stride_producer, field):
    case = make_case(stride_attention, stride_producer, spec(uq=[3, 1], uk=[17, 0]))
    for index in range(4):
        views = (View * 4).from_buffer_copy(case["views"]["migrated"])
        if field == "stride":
            views[index].strides[1] = -1
        elif field == "storage":
            views[index].storage_bytes = 0
        elif field == "shape":
            views[index].sizes[0] += 1
        else:
            views[index].offset = I64_MAX
        assert launch(stride_attention, case, views=views) == -3
    assert (case["outputs"]["migrated"][0].cpu() == 12345).all()
