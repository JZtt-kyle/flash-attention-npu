"""Replay native INT8 launch geometry for isolated numerical comparisons.

The fixed source's full-quant scheduling is not reimplemented by this harness.
Raw descriptors come from a validated native dispatch, with split/core geometry
unchanged. This is test input, never a production metadata protocol.
"""

import ctypes as ct
from pathlib import Path
import re

from tests.mla.paths import ROOT
from tests.mla.support.mla_native_dispatch import parse_dispatch


def params_type():
    classes = {"uint32_t": ct.c_uint32, "float": ct.c_float}
    header = ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_int8_params.h"
    for name, body in re.findall(r"struct alignas\(8\) (\w+) \{(.*?)\};", header.read_text(), re.S):
        fields = []
        for kind, member, count in re.findall(r"(\w+) (\w+)(?:\[([^]]+)\])?;", body):
            typ = classes[kind]
            if count:
                typ *= {"MAX_AIC_CORE_NUM": 26, "MAX_AIC_CORE_NUM * 2": 52}[count]
            fields.append((member, typ))
        classes[name] = type(name, (ct.Structure,), {"_fields_": fields})
        assert ct.sizeof(classes[name]) % 8 == 0
    return classes["Int8TilingParams"]


Params = params_type()


def parse_tiling(log, count, vendor, specs=None):
    evidence = parse_dispatch(log, "int8", count, vendor, representation="int8")
    lines = Path(log).read_text().splitlines()
    result = []
    for case in evidence["cases"]:
        if specs is not None:
            from tests.mla.support.mla_int8_cases import case_identity

            signature = case_identity(specs[case["case"]])
            signatures = [
                line
                for line in lines[case["begin_line"] : case["end_line"] - 1]
                if line.startswith("MLA_INT8_CASE_SIGNATURE ")
            ]
            if signatures != ["MLA_INT8_CASE_SIGNATURE " + signature]:
                raise RuntimeError(
                    "INT8 source fixture geometry/producer changed; rerun int8-source-check"
                )
        chunks, op_name, cores = {}, None, None
        for line in lines[case["begin_line"] : case["end_line"] - 1]:
            match = re.search(
                r"OpName:\[([^]]*_FusedInferAttentionScore)\] PrintTilingData\[(\d+)\]: (.*)", line
            )
            if match:
                if op_name is not None and op_name != match[1]:
                    raise RuntimeError("Ambiguous INT8 tiling data")
                op_name = match[1]
                index = int(match[2])
                if index in chunks:
                    raise RuntimeError("Repeated INT8 tiling chunk")
                payload = match[3].split("data(uint32_t):", 1)[-1]
                chunks[index] = [int(v) for v in payload.split()]
            if (
                "LaunchKernel: kernel info" in line
                and "kernel_name=FusedInferAttentionScore_" in line
            ):
                cores = int(re.search(r"coreDim=(\d+)", line)[1])
        if not chunks or sorted(chunks) != list(range(len(chunks))) or cores is None:
            raise RuntimeError("Incomplete INT8 tiling data")
        words = [v for i in range(len(chunks)) for v in chunks[i]]
        size = ct.sizeof(Params) // 4
        # Native runtime appends an overflow-detection trailer to raw tiling.
        if len(words) != size + 2 or words[-1] != 0xA5A5A5A5:
            raise RuntimeError("INT8 tiling size/trailer mismatch")
        raw = bytes((ct.c_uint32 * size)(*words[:size]))
        p = Params.from_buffer_copy(raw)
        if p.increFlashAttentionSingleCoreParams.usedCoreNum > cores or not 1 <= cores <= 20:
            raise RuntimeError("INT8 core count mismatch")
        key = case["native_tiling_key"]
        mode = (
            int(key % 10 == 2) | (int(case["native_fd"]) << 1) | (int(key // 10**9 % 10 == 1) << 2)
        )
        result.append(dict(case=case["case"], key=key, mode=mode, cores=cores, words=words[:size]))
    return dict(cases=result, artifacts=evidence["artifacts"])
