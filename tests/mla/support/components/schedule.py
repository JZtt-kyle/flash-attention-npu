"""Shared helpers extracted from tests/mla/unit/dense/test_flash_attn_npu_v3_mla_schedule.py.

CPU differential checks against compiled functions from the pinned CANN Git blobs.

MLA_CANN_SOURCE_ROOT selects a read-only ops-transformer checkout; only git show is
used. The oracle cannot include/link the migrated schedule. No NPU or torch needed.
Compare SplitCore vectors directly: upstream SetSplitOutput reads usedCoreNum*2
FD endpoints even for C1V1, where the vectors only have aicNum entries.
MLA_SCHEDULE_ARTIFACT_DIR keeps binaries, source manifests and case digests.
MLA_SCHEDULE_SANITIZE=1 additionally enables ASan/UBSan for both executables.
"""

import subprocess
from tests.mla.paths import ROOT

MLA = ROOT / "csrc/ascend910/flash_attn_npu_3/mla"

CANN_COMMIT = "ca18aee610729ca5270a44e2d84c751bc687fbf0"

FIA = "attention/fused_infer_attention_score/op_host/arch22/"

SOURCE_FILES = [
    "attention/common/op_host/split_core.h",
    "attention/common/op_host/split_core.cpp",
    FIA + "fia_tiling_nonquant_mla.cpp",
    FIA + "fia_tiling_nonquant_mla_arch22.h",
]

FUNCTIONS = [
    "CalcInnerSize",
    "CalcMBaseSize",
    "CalcMmResSize",
    "FillTilingWorkspaceParams",
    "CalcNormalWorkspaceSize",
    "CalcFlashDecodeParamNums",
    "CalcFlashDecodeWorkspace",
    "CalcWorkspaceSize",
    "CalcNumBlocks",
    "CalcScheduleMode",
]


def source_function(source, token):
    """Extract a complete function definition, preserving its body byte for byte."""
    start = source.rfind("\n", 0, source.index(token)) + 1
    brace = source.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + "\n"


def run(command, **kwargs):
    return subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=120, **kwargs
    )


def case(batch=1, sq=1, heads=128, sk=512, aic=20, ratio=2, **overrides):
    values = dict(
        batch=batch,
        sq=sq,
        heads=heads,
        sk=sk,
        aic=aic,
        aiv=aic * ratio,
        l1=516096,
        ub=178176 if ratio == 1 else 174080,
        l0a=65536,
        l0b=65536,
        l0c=131072,
        libapi=16777216,
        block_dim=0,
    )
    values.update(overrides)
    return list(values.values())
