"""Shared helpers extracted from tests/mla/e2e/bsnd/test_flash_attn_npu_v3_mla_bsnd_stride_production.py.

Production BSND strides: public/raw dispatch, source/math and view lifetime.
"""

from contextlib import contextmanager
import ctypes as ct
from pathlib import Path
import subprocess
import pytest
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_dense_build import ROOT
from tests.mla.support.components.bsnd_stride import (
    PERFORMANCE_CASES as COMPONENT_CASES,
    make_case as make_probe_case,
)
from tests.mla.support.public_cases.bsnd_used import check_result as check_result
from tests.mla.support.public_cases.contract import _forward

# Fix the production component matrix before collection. Production uses every
# platform core for length-dependent launches; the source uses the same schedule.
PERFORMANCE_CASES = [dict(c, aic=None) for c in COMPONENT_CASES]


def make_case(probes, producer, config, **kwargs):
    return make_probe_case(probes, producer, config, full_launch=True, **kwargs)


def forward_args(case):
    args = dict(case["devices"]["migrated"], softmax_scale=case["scale"])
    if case["config"]["version"] == 5:
        args["seqused_q"] = case["inputs"][2]
    if case["config"]["version"] != 1:
        args["seqused_k"] = case["inputs"][3]
    return args


def cached(args, metadata=None):
    return api.flash_attn_with_kvcache(
        args["q"],
        args["k"],
        args["v"],
        qv=args["qv"],
        cache_seqlens=args.get("seqused_k"),
        scheduler_metadata=metadata,
        softmax_scale=args.get("softmax_scale"),
        num_splits=1,
        return_softmax_lse=True,
    )


def forward(case, entry="public", metadata=None):
    args = forward_args(case)
    if entry == "cpp" or case["config"]["version"] == 5:
        return _forward("cpp" if entry == "cpp" else "eager", args)
    if case["config"]["version"] == 2:
        return cached(args, metadata)
    return api.flash_attn_func(**args, num_splits=1, return_attn_probs=True)


@pytest.fixture(scope="module")
def storage_fault(tmp_path_factory):
    directory = tmp_path_factory.mktemp("storage-fault")
    library = directory / "storage.so"
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-I" + str(torch_root / "include"),
            str(ROOT / "tests/mla/support/mla_tensor_storage_probe.cpp"),
            "-L" + str(torch_root / "lib"),
            "-Wl,-rpath," + str(torch_root / "lib"),
            "-lc10",
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ct.CDLL(str(library))
    lib.mla_test_storage_bytes.argtypes = [ct.c_uint64, ct.c_uint64]
    lib.mla_test_storage_bytes.restype = ct.c_uint64

    @contextmanager
    def inject(tensor):
        # Change only host byte-size metadata; preserve allocation and streams.
        # NPU Storage.resize_ queues copies and is unsuitable for this fixture.
        # Compact logical bytes still fit: rejection must use the actual outer
        # strides and the full allocation, even when all used lengths are zero.
        truncated = (tensor.storage_offset() + tensor.numel()) * tensor.element_size()
        assert truncated < tensor.untyped_storage().size()
        previous = lib.mla_test_storage_bytes(tensor._cdata, truncated)
        try:
            assert tensor.untyped_storage().size() == truncated
            yield
        finally:
            assert lib.mla_test_storage_bytes(tensor._cdata, previous) == truncated

    return inject
