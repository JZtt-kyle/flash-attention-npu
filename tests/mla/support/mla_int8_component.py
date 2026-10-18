"""Buffer ownership and native tiling replay for the isolated INT8 component."""

import ctypes as ct
import itertools
import os
from functools import lru_cache

import torch

from tests.mla.support.mla_int8_build import build_probes
from tests.mla.support.mla_int8_cases import NativeCall, SOURCE_CASES
from tests.mla.support.mla_int8_native import native_vendor
from tests.mla.support.mla_int8_tiling import Params, parse_tiling
from tests.mla.support.mla_int8_layout import CacheLayout, checked_layout, preserve_view


@lru_cache(maxsize=1)
def fixtures():
    return parse_tiling(
        os.environ["MLA_INT8_SOURCE_LOG"], len(SOURCE_CASES), native_vendor(), SOURCE_CASES
    )["cases"]


@lru_cache(maxsize=1)
def libraries():
    manifest = build_probes()
    result = {}
    for name, info in manifest["libraries"].items():
        lib = ct.CDLL(info["path"])
        assert lib.int8_params_size() == ct.sizeof(Params)
        lib.launch_int8.argtypes = [ct.c_void_p] * 15 + [
            ct.POINTER(Params),
            ct.c_int,
            ct.POINTER(CacheLayout),
        ]
        lib.launch_int8.restype = ct.c_int
        result[name] = lib
    return result


class ComponentCall:
    def __init__(self, spec, inputs, *, oracle=False, lse=False, legacy_fd=False):
        self.fixture = fixtures()[SOURCE_CASES.index(spec)]
        self.lib = libraries()["oracle" if oracle else "migrated"]
        self.native = NativeCall(spec, inputs, nz=oracle)
        self.params = Params.from_buffer_copy(
            bytes((ct.c_uint32 * len(self.fixture["words"]))(*self.fixture["words"]))
        )
        p = self.params
        p.baseParams.softmaxLseFlag = int(lse)
        if legacy_fd:
            assert oracle and lse and p.tndSplitCoreParams.FdBalanceFlag == 1
            p.tndSplitCoreParams.FdBalanceFlag = 0
        assert p.increFlashAttentionSingleCoreParams.usedCoreNum == self.fixture["cores"]
        _, layout, qlens, _, klens, _ = spec
        self.layout = checked_layout(inputs, klens)
        n = self.native
        self.out_storage = torch.full(
            (n.out.numel() + 64,), torch.nan, dtype=torch.bfloat16, device="npu:0"
        )
        n.out = self.out_storage[32:-32].reshape(n.out.shape)
        self.q_lengths = torch.tensor(
            list(itertools.accumulate(qlens)) if layout == "TND" else qlens,
            dtype=torch.int64,
            device="npu:0",
        )
        self.k_lengths = torch.tensor(klens, dtype=torch.int64, device="npu:0")
        self.lse = torch.full(
            (max(n.out.numel() // 512, 1),), torch.nan, dtype=torch.float32, device="npu:0"
        )
        t = p.increFlashAttentionSingleCoreTensorSize
        self.workspace_size = self.fixture["cores"] * 2 * (t.mmResUbSize * 5 + t.bmm2ResUbSize * 4)
        self.partial_offset = self.workspace_size
        self.workspace_size += 4 * (p.splitKVParams.accumOutSize + p.splitKVParams.logSumExpSize)
        self.workspace = torch.zeros(self.workspace_size + 512, dtype=torch.uint8, device="npu:0")
        self.workspace[self.workspace_size :].fill_(123)
        self.key = n.key if oracle else n.inputs["v"]
        self.rope = n.rope if oracle else n.inputs["k"]
        self.table = n.inputs["table"]
        if not oracle:
            self.key, self.rope, self.table = (
                preserve_view(inputs[k]) for k in ("v", "k", "table")
            )
        # A single-element CANN TensorList, used only by the untouched oracle.
        # shapeInfo=(count << 32), dim=0, pointer-only shape sentinel.
        self.list_desc = torch.tensor(
            [24, 1 << 32, 0xFFFFFFFF, self.key.data_ptr()], dtype=torch.int64, device="npu:0"
        )
        buffers = [
            n.inputs["qv"],
            n.inputs["q"],
            self.list_desc if oracle else self.key,
            self.rope,
            n.inputs["qs"],
            n.inputs["ks"],
            n.inputs["vs"],
            self.table,
            self.q_lengths,
            self.k_lengths,
            n.kwargs.get("atten_mask"),
            n.out,
            self.lse,
            self.workspace,
        ]
        self.args = [None if v is None else v.data_ptr() for v in buffers]

    def __call__(self):
        result = self.lib.launch_int8(
            torch.npu.current_stream().npu_stream,
            *self.args,
            ct.byref(self.params),
            self.fixture["mode"],
            ct.byref(self.layout),
        )
        if result:
            raise RuntimeError(f"INT8 component launch failed: {result}")
        return self.native.out
