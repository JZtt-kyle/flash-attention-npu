"""910B PA Copy: exact L1 NZ data, fixed source and independent scalar indexing.

The source receives the same physical views when its Copy ABI can express
them. Zero or >uint32 table strides use independently materialized source
views; this conversion is outside component timing and is recorded explicitly.
No PA attention or public forward is invoked.
"""

import itertools
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_paged_copy_build import build_probes, load_probes

from tests.mla.support.components.paged_copy import (
    GUARD_ELEMENTS as GUARD_ELEMENTS,
    SENTINEL as SENTINEL,
    SLOT_ELEMENTS as SLOT_ELEMENTS,
    cache_view as cache_view,
    check_case as check_case,
    launch as launch,
    make_case as make_case,
    page_tasks as page_tasks,
    scalar_expected as scalar_expected,
)


@pytest.fixture(scope="module")
def copy_build(tmp_path_factory):
    directory = Path(os.environ.get("MLA_PAGED_COPY_DIR", tmp_path_factory.mktemp("paged-copy")))
    if os.environ.get("MLA_PAGED_COPY_REUSE") != "1":
        build_probes(directory)
    torch.npu.set_device(0)
    assert "910B" in torch.npu.get_device_name(0)
    torch.set_num_threads(2)
    return directory


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def copy_probes(copy_build, request):
    dtype = torch.float16 if request.param == "fp16" else torch.bfloat16
    return load_probes(copy_build, request.param), copy_build, dtype


@pytest.mark.parametrize("page", range(16, 1025, 16))
@pytest.mark.parametrize("profile", ("contiguous", "independent"))
def test_all_page_sizes(copy_probes, page, profile):
    check_case(copy_probes, make_case(copy_probes[2], page, profile))


@pytest.mark.parametrize("profile", ("overlap", "broadcast", "table_overlap", "table_broadcast"))
def test_readonly_aliases(copy_probes, profile):
    check_case(copy_probes, make_case(copy_probes[2], 16, profile))


@pytest.mark.parametrize("target,stride", itertools.product(("k", "v"), (65535, 65536, 65537)))
def test_nd2nz_stride_limit(copy_probes, target, stride):
    check_case(
        copy_probes, make_case(copy_probes[2], 16, f"{target}_row_{stride}", physical_pages=2)
    )


@pytest.mark.parametrize(
    "profile", ("wide_k_page", "wide_v_page", "wide_k_row", "wide_v_row", "wide_table")
)
def test_real_wide_addresses(copy_probes, profile):
    tasks = [[b, 9, 7, 32, mode] for b in (0, 1) for mode in range(6)]
    pages = 1 if profile.endswith("row") else 2
    case = make_case(copy_probes[2], 16, profile, tasks, physical_pages=pages)
    check_case(copy_probes, case)
    del case
    torch.npu.empty_cache()


@pytest.mark.parametrize("start,count", ((15, 17), (1, 1), (31, 31), (16, 256), (0, 0)))
def test_tail_and_zero_count(copy_probes, start, count):
    rows = min(256, ((count + 30) // 16 + 1) * 16)
    tasks = [[0, start, count, rows, mode] for mode in range(6)]
    check_case(copy_probes, make_case(copy_probes[2], 16, tasks=tasks), rounds=5)


def test_empty_pool_table(copy_probes):
    tasks = [[0, 0, 0, 16, mode] for mode in range(6)]
    check_case(copy_probes, make_case(copy_probes[2], 528, tasks=tasks, empty=True))


def test_stream_events_and_buffer_reuse(copy_probes):
    cases = [make_case(copy_probes[2], p) for p in (16, 112, 528)]
    workers = [torch.npu.Stream() for _ in range(2)]
    ready = torch.npu.Event()
    ready.record()
    done = []
    for worker in workers:
        worker.wait_event(ready)
        with torch.npu.stream(worker):
            for case in cases:
                for name in copy_probes[0]:
                    launch(copy_probes[0], case, name, rounds=11)
            event = torch.npu.Event()
            event.record()
            done.append(event)
        # The two workers reuse outputs with an explicit dependency.
        ready = event
    for event in done:
        torch.npu.current_stream().wait_event(event)
    for case in cases:
        check_case(copy_probes, case, rounds=7)
