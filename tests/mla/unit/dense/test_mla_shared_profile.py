"""Representative no-mask performance for the shared task ring and both Vector layouts.

Dense retains the four existing matrices. BSND/TND PA retains each existing
PERFORMANCE_CASES list. Other layout address/Copy implementations are unchanged;
all their numerical call paths have separate shared-nonpaged acceptance.
"""

import os
from pathlib import Path

import pytest
import torch

from tests.mla.support.mla_dense_profile import profile as dense_profile
from tests.mla.support.mla_profile_cases import CASES, S1G_CASES, KV_CASES, LONG_KV_CASES
from tests.mla.support.mla_bsnd_stride_profile import profile as paged_profile
from tests.mla.unit.paged_bsnd import test_flash_attn_npu_v3_mla_paged_attention as bsnd
from tests.mla.unit.paged_tnd import test_flash_attn_npu_v3_mla_tnd_paged_attention as tnd


@pytest.fixture(autouse=True)
def isolated_cpu_affinity():
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    torch.npu.set_device(0)
    assert torch.npu.device_count() == 1 and "910B" in torch.npu.get_device_name(0)
    try:
        yield
    finally:
        os.sched_setaffinity(0, affinity)


@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
def test_dense_component_profile(dtype):
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "dense-profile" / dtype
    dense_profile(
        directory,
        dtype_name=dtype,
        cases=CASES + S1G_CASES + KV_CASES + LONG_KV_CASES,
        source_only=True,
    )


@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
@pytest.mark.parametrize("layout", ("bsnd", "tnd"))
def test_paged_component_profile(dtype, layout):
    directory = Path(os.environ["MLA_VALIDATION_RESULT"])
    # The unified entry validates and copies the current PA build manifest;
    # load_attention rechecks its sources, transitive dependencies and DSO hashes.
    probes = directory / "probes" / dtype
    producer = Path(os.environ["MLA_PAGED_METADATA_LIBRARY"]).with_name("build-manifest.json")
    paged_profile(
        probes,
        producer,
        directory / f"paged-{layout}-profile" / dtype,
        suite=bsnd if layout == "bsnd" else tnd,
        kernel_prefix="paged",
        kernel_suffix="attention",
        seed=20261011,
        scope="Representative unmasked shared task-ring/Vector performance using the complete existing BSND/TND PA matrices. Fixed CANN non-FD and migration use identical schedules; layout materialization and metadata generation are outside timing. Preallocated probe wall is not public E2E.",
    )
