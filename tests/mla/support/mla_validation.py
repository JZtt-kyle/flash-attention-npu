"""Reproducible MLA checks: python -m tests.mla.support.mla_validation --help.

Daily E2E runs its explicit small-input cases; --full-matrix expands diagnostic combinations.
Builds, golden tensors and successful test evidence have separate identities.
Performance remains in tests.mla.support.mla_paged_profile and must always be measured.
"""

from tests.mla.paths import work_dir

import argparse
from collections import Counter
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from tests.mla.paths import relocated
from tests.mla.catalog import DIAGNOSTIC_E2E, E2E, UNITS, combined, is_e2e
from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.mla_validation_build import matching_files
from tests.mla.support.mla_validation_dependencies import suite_sources
from tests.mla.support.mla_validation_selection import INCREMENTAL_SUITES
from tests.mla.support.mla_split_stage_cases import SPLIT_STAGE_POLICY, SPLIT_STAGE_SUITES
from tests.mla.support.mla_validation_policy import (
    DEFAULT_MAX_CASES,
    POLICY_VERSION,
    CAUSAL_STAGE_POLICY,
    CAUSAL_STAGE_SUITES,
    WINDOW_STAGE_SUITES,
)


PUBLIC = "tests/mla/e2e/paged_tnd/test_flash_attn_npu_v3_mla_paged_tnd_public.py"
WINDOW_CPU = "tests/mla/unit/window_bsnd/test_flash_attn_npu_v3_mla_window_schedule.py"
WINDOW_ATTENTION = "tests/mla/unit/window_bsnd/test_flash_attn_npu_v3_mla_window_attention.py"
CAUSAL_CPU = "tests/mla/unit/causal_bsnd/test_flash_attn_npu_v3_mla_causal_schedule.py"
USED_SHARED_CPU = [
    f"tests/test_flash_attn_npu_v3_mla_{name}.py"
    for name in (
        "schedule",
        "bsnd_schedule",
        "tnd_schedule",
        "tnd_bsnd_schedule",
        "used_schedule",
        "paged_schedule",
        "used_metadata",
    )
]
SPLIT_KV_ATTENTION = "tests/mla/unit/split_kv_attention/test_attention.py"
SPLIT_KV_MERGE = "tests/mla/unit/split_kv_merge/test_merge.py"
SPLIT_KV_MERGE_SUITES = {
    "unit-auto_split_paged_tnd",
    "unit-auto_split_paged_bsnd",
    "unit-split_paged_bsnd",
    "unit-auto_split_mixed",
    "unit-split_mixed",
    "unit-auto_split_tnd",
    "split-tnd",
    "unit-split_tnd",
    "auto-split-bsnd",
    "unit-auto_split_bsnd",
    "split-bsnd",
    "split-bsnd-source",
    "unit-split_kv_bsnd",
    "unit-split_kv_merge",
    "split-kv-merge-check",
    "split-kv-merge",
    "split-kv-merge-profile",
}
SPLIT_KV_DEVICE_SUITES = {
    "unit-split_kv_attention",
    "split-kv-attention-check",
    "split-kv-attention",
    "split-kv-attention-profile",
}
SUITES = {
    "prefill-bsnd-tools": [
        "tests/mla/unit/tools/test_mla_prefill_bsnd_tools.py",
        "tests/mla/unit/tools/test_flash_attn_npu_v3_mla_build.py::test_aicpu_sources_by_backend",
        "tests/mla/unit/tools/test_organization.py",
    ],
    "prefill-bsnd-profile-check": ["tests/mla/e2e/prefill_bsnd/test_profile.py"],
    "prefill-bsnd-profile": ["tests/mla/e2e/prefill_bsnd/test_profile.py"],
    "prefill-bsnd-trap": ["tests/mla/e2e/prefill_bsnd/test_trap.py"],
    "prefill-bsnd-sanitize": ["tests/mla/unit/prefill_bsnd/test_metadata.py"],
    "prefill-bsnd-check": [
        "tests/mla/e2e/prefill_bsnd/test_public.py",
        "-k",
        "representative or cost_boundary",
    ],
    "prefill-bsnd-public": [
        "tests/mla/e2e/prefill_bsnd/test_public.py",
        "tests/mla/e2e/prefill_bsnd/test_source.py",
        "tests/mla/e2e/dense/test_flash_attn_npu_v3_mla_contract.py",
        "-k",
        "prefill or metadata_auto_default",
    ],
    "prefill-tools": ["tests/mla/unit/tools/test_mla_prefill_tools.py"],
    "prefill-check": ["tests/mla/unit/prefill_attention/test_attention.py", "-k", "representative"],
    "prefill-profile": ["tests/mla/unit/prefill_attention/component_profile.py"],
    "prefill-profile-check": ["tests/mla/unit/prefill_attention/component_profile.py"],
    "split-bsnd-public": ["tests/mla/e2e/split_kv_bsnd/test_public.py"],
    "auto-split-bsnd-public": ["tests/mla/e2e/auto_split_bsnd/test_public.py"],
    "split-tnd-public": ["tests/mla/e2e/split_tnd/test_public.py"],
    "split-stage-tools": [
        "tests/mla/unit/tools/test_mla_split_stage_performance.py",
        "tests/mla/unit/tools/test_mla_native_dispatch.py",
        "tests/mla/unit/tools/test_mla_validation.py",
        "tests/mla/unit/tools/test_mla_validation_policy.py",
        "tests/mla/unit/tools/test_mla_causal_stage_performance.py",
        "tests/mla/unit/tools/test_organization.py",
    ],
    "auto-split-paged-tnd-tools": [
        "tests/mla/unit/tools/test_mla_auto_split_paged_tnd_production_tools.py",
        "tests/mla/unit/tools/test_flash_attn_npu_v3_mla_build.py::test_aicpu_sources_by_backend",
        "tests/mla/unit/tools/test_organization.py",
    ],
    "auto-split-paged-tnd-check": [
        "tests/mla/e2e/auto_split_paged_tnd/test_public.py",
        "-k",
        "representative",
    ],
    "auto-split-paged-tnd-public": [
        "tests/mla/e2e/auto_split_paged_tnd/test_public.py",
        "tests/mla/e2e/split_paged_bsnd/test_public.py::test_invalid_tnd_paged_split",
        "tests/mla/e2e/auto_split_paged_bsnd/test_public.py::test_invalid_tnd_paged_split",
        "tests/mla/e2e/split_paged_tnd/test_public.py::test_invalid_paged_split",
    ],
    "auto-split-paged-tnd-trap": ["tests/mla/e2e/auto_split_paged_tnd/test_trap.py"],
    "auto-split-paged-tnd-profile": ["tests/mla/e2e/auto_split_paged_tnd/test_profile.py"],
    "auto-split-paged-tnd-protocol-tools": [
        "tests/mla/unit/tools/test_mla_auto_split_paged_tnd_tools.py"
    ],
    "auto-split-paged-tnd-protocol-sanitize": [
        "tests/mla/unit/auto_split_paged_tnd_protocol/test_plan.py",
        "tests/mla/unit/auto_split_paged_tnd_protocol/test_metadata.py",
    ],
    "auto-split-paged-tnd-protocol-check": [
        "tests/mla/unit/auto_split_paged_tnd_protocol/test_plan.py",
        "tests/mla/unit/auto_split_paged_tnd_protocol/test_metadata.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "auto-split-paged-bsnd-tools": [
        "tests/mla/unit/tools/test_mla_auto_split_paged_bsnd_production_tools.py"
    ],
    "auto-split-paged-bsnd-check": [
        "tests/mla/e2e/auto_split_paged_bsnd/test_public.py",
        "-k",
        "representative",
    ],
    "auto-split-paged-bsnd-public": ["tests/mla/e2e/auto_split_paged_bsnd/test_public.py"],
    "auto-split-paged-bsnd-trap": ["tests/mla/e2e/auto_split_paged_bsnd/test_trap.py"],
    "auto-split-paged-bsnd-profile": ["tests/mla/e2e/auto_split_paged_bsnd/test_profile.py"],
    "auto-split-paged-bsnd-protocol-tools": [
        "tests/mla/unit/tools/test_mla_auto_split_paged_bsnd_tools.py"
    ],
    "auto-split-paged-bsnd-protocol-sanitize": [
        "tests/mla/unit/auto_split_paged_bsnd_protocol/test_plan.py",
        "tests/mla/unit/auto_split_paged_bsnd_protocol/test_metadata.py",
    ],
    "auto-split-paged-bsnd-protocol-check": [
        "tests/mla/unit/auto_split_paged_bsnd_protocol/test_plan.py",
        "tests/mla/unit/auto_split_paged_bsnd_protocol/test_metadata.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "auto-split-mixed-tools": [
        "tests/mla/unit/tools/test_mla_auto_split_mixed_production_tools.py",
        "tests/mla/unit/tools/test_mla_auto_split_mixed_tools.py",
    ],
    "auto-split-mixed-check": [
        "tests/mla/e2e/auto_split_mixed/test_public.py",
        "-k",
        "representative",
    ],
    "auto-split-mixed-public": ["tests/mla/e2e/auto_split_mixed/test_public.py"],
    "auto-split-mixed-trap": ["tests/mla/e2e/auto_split_mixed/test_trap.py"],
    "auto-split-mixed-profile": ["tests/mla/e2e/auto_split_mixed/test_auto_split_mixed_profile.py"],
    "auto-split-mixed-protocol-tools": ["tests/mla/unit/tools/test_mla_auto_split_mixed_tools.py"],
    "auto-split-mixed-protocol-sanitize": [
        "tests/mla/unit/auto_split_mixed_protocol/test_plan.py",
        "tests/mla/unit/auto_split_mixed_protocol/test_metadata.py",
    ],
    "auto-split-mixed-protocol-check": [
        "tests/mla/unit/auto_split_mixed_protocol/test_plan.py",
        "tests/mla/unit/auto_split_mixed_protocol/test_metadata.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "split-paged-tnd-tools": ["tests/mla/unit/tools/test_mla_split_paged_tnd_production_tools.py"],
    "split-paged-tnd-check": [
        "tests/mla/e2e/split_paged_tnd/test_public.py",
        "-k",
        "representative",
    ],
    "split-paged-tnd-public": ["tests/mla/e2e/split_paged_tnd/test_public.py"],
    "split-paged-tnd-trap": ["tests/mla/e2e/split_paged_tnd/test_trap.py"],
    "split-paged-tnd-profile": ["tests/mla/e2e/split_paged_tnd/test_profile.py"],
    "split-paged-bsnd-tools": [
        "tests/mla/unit/tools/test_mla_split_paged_bsnd_production_tools.py"
    ],
    "split-paged-bsnd-check": [
        "tests/mla/e2e/split_paged_bsnd/test_public.py",
        "-k",
        "representative",
    ],
    "split-paged-bsnd-public": ["tests/mla/e2e/split_paged_bsnd/test_public.py"],
    "split-paged-bsnd-trap": ["tests/mla/e2e/split_paged_bsnd/test_trap.py"],
    "split-paged-bsnd-profile": ["tests/mla/e2e/split_paged_bsnd/test_profile.py"],
    "split-paged-tnd-protocol-tools": ["tests/mla/unit/tools/test_mla_split_paged_tnd_tools.py"],
    "split-paged-tnd-protocol-sanitize": [
        "tests/mla/unit/split_paged_tnd_protocol/test_plan.py",
        "tests/mla/unit/split_paged_tnd_protocol/test_metadata.py",
    ],
    "split-paged-tnd-protocol-check": [
        "tests/mla/unit/split_paged_tnd_protocol/test_plan.py",
        "tests/mla/unit/split_paged_tnd_protocol/test_metadata.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "split-paged-bsnd-protocol-tools": ["tests/mla/unit/tools/test_mla_split_paged_bsnd_tools.py"],
    "split-paged-bsnd-protocol-sanitize": [
        "tests/mla/unit/split_paged_bsnd_protocol/test_plan.py",
        "tests/mla/unit/split_paged_bsnd_protocol/test_metadata.py",
    ],
    "split-paged-bsnd-protocol-check": [
        "tests/mla/unit/split_paged_bsnd_protocol/test_plan.py",
        "tests/mla/unit/split_paged_bsnd_protocol/test_metadata.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "split-mixed-protocol-tools": ["tests/mla/unit/tools/test_mla_split_mixed_tools.py"],
    "split-mixed-protocol-sanitize": [
        "tests/mla/unit/split_mixed_protocol/test_plan.py",
        "tests/mla/unit/split_mixed_protocol/test_metadata.py",
    ],
    "split-mixed-protocol-check": [
        "tests/mla/unit/split_mixed_protocol/test_plan.py",
        "tests/mla/unit/split_mixed_protocol/test_metadata.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "auto-split-tnd-tools": ["tests/mla/unit/tools/test_mla_auto_split_tnd_tools.py"],
    "auto-split-tnd-check": ["tests/mla/e2e/auto_split_tnd/test_public.py", "-k", "representative"],
    "auto-split-tnd-public": ["tests/mla/e2e/auto_split_tnd/test_public.py"],
    "auto-split-tnd-trap": ["tests/mla/e2e/auto_split_tnd/test_trap.py"],
    "auto-split-tnd-profile": ["tests/mla/e2e/auto_split_tnd/test_auto_split_tnd_profile.py"],
    "split-mixed-tools": ["tests/mla/unit/tools/test_mla_split_mixed_production_tools.py"],
    "split-mixed-check": ["tests/mla/e2e/split_mixed/test_public.py", "-k", "representative"],
    "split-mixed-public": ["tests/mla/e2e/split_mixed/test_public.py"],
    "split-mixed-trap": ["tests/mla/e2e/split_mixed/test_trap.py"],
    "split-mixed-profile": ["tests/mla/e2e/split_mixed/test_split_mixed_profile.py"],
    "split-tnd-tools": ["tests/mla/unit/tools/test_mla_split_tnd_tools.py"],
    "split-tnd-cpu": ["tests/mla/unit/split_tnd/test_metadata.py"],
    "split-tnd-sanitize": ["tests/mla/unit/split_tnd/test_metadata.py"],
    "split-tnd-check": ["tests/mla/e2e/split_tnd/test_public.py", "-k", "representative"],
    "split-tnd": [
        "tests/mla/unit/split_tnd/test_source.py",
        "tests/mla/e2e/split_tnd/test_public.py",
    ],
    "split-tnd-trap": ["tests/mla/e2e/split_tnd/test_trap.py"],
    "split-tnd-profile": ["tests/mla/e2e/split_tnd/test_split_tnd_profile.py"],
    "auto-split-bsnd-check": [
        "tests/mla/e2e/auto_split_bsnd/test_public.py",
        "-k",
        "representative",
    ],
    "auto-split-bsnd": [
        "tests/mla/e2e/auto_split_bsnd/test_public.py",
        "tests/mla/unit/auto_split_bsnd/test_source.py",
    ],
    "auto-split-bsnd-tools": ["tests/mla/unit/tools/test_mla_auto_split_bsnd_tools.py"],
    "auto-split-bsnd-trap": ["tests/mla/e2e/auto_split_bsnd/test_trap.py"],
    "auto-split-bsnd-profile": ["tests/mla/e2e/auto_split_bsnd/test_auto_split_bsnd_profile.py"],
    "auto-split-metadata-sanitize": ["tests/mla/unit/auto_split_bsnd_metadata/test_metadata.py"],
    "auto-split-metadata-tools": ["tests/mla/unit/tools/test_mla_auto_split_metadata_tools.py"],
    "auto-split-metadata-check": [
        "tests/mla/unit/auto_split_bsnd_metadata/test_metadata.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "auto-split-tnd-metadata-sanitize": ["tests/mla/unit/auto_split_tnd_metadata/test_metadata.py"],
    "auto-split-tnd-metadata-tools": [
        "tests/mla/unit/tools/test_mla_auto_split_tnd_metadata_tools.py"
    ],
    "auto-split-tnd-metadata-check": [
        "tests/mla/unit/auto_split_tnd_metadata/test_metadata.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "auto-split-tnd-plan-sanitize": ["tests/mla/unit/auto_split_tnd_plan/test_plan.py"],
    "auto-split-tnd-plan-tools": ["tests/mla/unit/tools/test_mla_auto_split_tnd_plan_tools.py"],
    "auto-split-tnd-plan-check": [
        "tests/mla/unit/auto_split_tnd_plan/test_plan.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "auto-split-kv-plan-sanitize": ["tests/mla/unit/auto_split_kv_plan/test_plan.py"],
    "auto-split-kv-plan-tools": ["tests/mla/unit/tools/test_mla_auto_split_kv_plan_tools.py"],
    "auto-split-kv-plan-check": [
        "tests/mla/unit/auto_split_kv_plan/test_plan.py",
        "-k",
        "representative or capacity_cost_boundary",
    ],
    "split-bsnd-cpu": ["tests/mla/unit/split_kv_bsnd/test_metadata.py"],
    "split-bsnd-sanitize": ["tests/mla/unit/split_kv_bsnd/test_metadata.py"],
    "split-bsnd-tools": ["tests/mla/unit/tools/test_mla_split_bsnd_tools.py"],
    "split-bsnd-check": ["tests/mla/e2e/split_kv_bsnd/test_public.py", "-k", "representative"],
    "split-bsnd": [
        "tests/mla/e2e/split_kv_bsnd/test_public.py",
        "tests/mla/unit/split_kv_bsnd/test_source.py",
    ],
    "split-bsnd-profile": ["tests/mla/e2e/split_kv_bsnd/test_split_bsnd_profile.py"],
    "split-bsnd-trap": ["tests/mla/e2e/split_kv_bsnd/test_trap.py"],
    "split-bsnd-source": ["tests/mla/unit/split_kv_bsnd/test_source.py"],
    "split-tnd-plan-sanitize": ["tests/mla/unit/split_tnd_plan/test_plan.py"],
    "split-tnd-plan-tools": ["tests/mla/unit/tools/test_mla_split_tnd_plan_tools.py"],
    "split-tnd-plan-check": [
        "tests/mla/unit/split_tnd_plan/test_plan.py::test_source_ranges_ownership_and_allocation_rows[window-used-gaps]",
        "tests/mla/unit/split_tnd_plan/test_plan.py::test_u64_task_ids_and_last_row_without_enumeration",
        "tests/mla/unit/split_tnd_plan/test_plan.py::test_capacity_cost_boundary_and_byte_overflow",
    ],
    "split-kv-plan-sanitize": ["tests/mla/unit/split_kv_plan/test_plan.py"],
    "split-kv-plan-tools": ["tests/mla/unit/tools/test_mla_split_kv_plan_tools.py"],
    "split-kv-merge-check": [SPLIT_KV_MERGE, "-k", "representative or cost_boundary"],
    "split-kv-merge": [SPLIT_KV_MERGE, "-k", "not representative and not cost_boundary"],
    "split-kv-merge-profile": ["tests/mla/unit/split_kv_merge/test_component_profile.py"],
    "split-kv-merge-tools": ["tests/mla/unit/tools/test_mla_split_kv_merge_tools.py"],
    "split-kv-attention-check": [SPLIT_KV_ATTENTION, "-k", "representative or cost_boundary"],
    "split-kv-attention": [SPLIT_KV_ATTENTION, "-k", "not representative and not cost_boundary"],
    "split-kv-attention-profile": ["tests/mla/unit/split_kv_attention/test_component_profile.py"],
    "split-kv-attention-tools": ["tests/mla/unit/tools/test_mla_split_kv_attention_tools.py"],
    "window-paged-tnd-cpu": [
        "tests/mla/unit/window_paged_tnd/test_flash_attn_npu_v3_mla_window_paged_tnd_metadata.py"
    ],
    "window-paged-tnd-sanitize": [
        "tests/mla/unit/window_paged_tnd/test_flash_attn_npu_v3_mla_window_paged_tnd_metadata.py"
    ],
    "window-paged-tnd-check": [
        "tests/mla/e2e/window_paged_tnd/test_flash_attn_npu_v3_mla_window_paged_tnd_public.py",
        "-k",
        "representative or cost_boundary",
    ],
    "window-paged-tnd-public": [
        "tests/mla/e2e/window_paged_tnd/test_flash_attn_npu_v3_mla_window_paged_tnd_public.py",
        "-k",
        "not representative and not cost_boundary and not release_trap",
    ],
    "window-paged-tnd-trap": [
        "tests/mla/e2e/window_paged_tnd/test_flash_attn_npu_v3_mla_window_paged_tnd_public.py",
        "-k",
        "release_trap",
    ],
    "window-paged-tnd-profile": [
        "tests/mla/e2e/window_paged_tnd/test_mla_window_paged_tnd_profile.py"
    ],
    "window-paged-cpu": [
        "tests/mla/unit/window_paged/test_flash_attn_npu_v3_mla_window_paged_metadata.py"
    ],
    "window-paged-sanitize": [
        "tests/mla/unit/window_paged/test_flash_attn_npu_v3_mla_window_paged_metadata.py"
    ],
    "window-paged-check": [
        "tests/mla/e2e/window_paged/test_flash_attn_npu_v3_mla_window_paged_public.py",
        "-k",
        "representative or cost_boundary",
    ],
    "window-paged-public": [
        "tests/mla/e2e/window_paged/test_flash_attn_npu_v3_mla_window_paged_public.py",
        "-k",
        "not representative and not cost_boundary and not release_trap",
    ],
    "window-paged-trap": [
        "tests/mla/e2e/window_paged/test_flash_attn_npu_v3_mla_window_paged_public.py",
        "-k",
        "release_trap",
    ],
    "window-paged-profile": ["tests/mla/e2e/window_paged/test_mla_window_paged_profile.py"],
    "window-paged-boundary": [
        "tests/mla/e2e/causal_paged/test_flash_attn_npu_v3_mla_causal_paged_public.py::test_static_and_unopened_boundaries",
        "tests/mla/e2e/causal_paged_tnd/test_flash_attn_npu_v3_mla_causal_paged_tnd_public.py::test_static_and_unopened_boundaries",
        "tests/mla/e2e/paged_bsnd/test_flash_attn_npu_v3_mla_paged_metadata_public.py::test_unsupported_options",
        "tests/mla/e2e/paged_bsnd/test_flash_attn_npu_v3_mla_paged_metadata_public.py::test_schema_compile_and_length_updates",
        "tests/mla/e2e/paged_bsnd/test_flash_attn_npu_v3_mla_paged_public.py::test_compile_and_length_updates",
        "tests/mla/e2e/paged_tnd/test_flash_attn_npu_v3_mla_paged_tnd_public.py::test_compile_and_length_updates",
        "-k",
        "static_and_unopened_boundaries or window_size or schema_compile_and_length_updates or test_compile_and_length_updates",
    ],
    "window-paged-tnd-tools": [
        "tests/mla/unit/tools/test_mla_window_paged_tnd_tools.py",
        "tests/mla/unit/tools/test_mla_causal_stage_performance.py",
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_success_failure_interruption_and_resume[{suite}]"
            for suite in (
                "window-paged-tnd-tools",
                "window-paged-tnd-cpu",
                "window-paged-tnd-sanitize",
            )
        ],
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_dependencies_changing_during_execution_fail[{suite}]"
            for suite in ("window-paged-tnd-cpu", "window-paged-tnd-sanitize")
        ],
        "tests/mla/unit/tools/test_mla_validation.py::test_performance_always_executes_on_resume[window-paged-tnd-profile]",
    ],
    "window-stage-correctness": [
        "tests/mla/e2e/window_stage/test_window_stage.py",
    ],
    "window-paged-tools": [
        "tests/mla/unit/tools/test_mla_window_paged_tools.py",
        "tests/mla/unit/tools/test_mla_validation_policy.py::test_mixed_average_weights_entries_and_keeps_regressions",
        "tests/mla/unit/tools/test_mla_validation_policy.py::test_mixed_average_rejects_incomplete_evidence",
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_success_failure_interruption_and_resume[{suite}]"
            for suite in ("window-paged-tools", "window-paged-cpu", "window-paged-sanitize")
        ],
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_dependencies_changing_during_execution_fail[{suite}]"
            for suite in ("window-paged-cpu", "window-paged-sanitize")
        ],
        "tests/mla/unit/tools/test_mla_validation.py::test_performance_always_executes_on_resume[window-paged-profile]",
        "tests/mla/unit/tools/test_mla_validation.py::test_evidence_missing_changed_failed_or_different_identity_cannot_resume",
    ],
    "window-mixed-cpu": [
        "tests/mla/unit/window_mixed/test_flash_attn_npu_v3_mla_window_mixed_metadata.py"
    ],
    "window-mixed-sanitize": [
        "tests/mla/unit/window_mixed/test_flash_attn_npu_v3_mla_window_mixed_metadata.py"
    ],
    "window-mixed-check": [
        "tests/mla/e2e/window_mixed/test_flash_attn_npu_v3_mla_window_mixed_public.py",
        "-k",
        "representative or cost_boundary",
    ],
    "window-mixed-public": [
        "tests/mla/e2e/window_mixed/test_flash_attn_npu_v3_mla_window_mixed_public.py",
        "-k",
        "not representative and not cost_boundary and not release_trap",
    ],
    "window-mixed-trap": [
        "tests/mla/e2e/window_mixed/test_flash_attn_npu_v3_mla_window_mixed_public.py",
        "-k",
        "release_trap",
    ],
    "window-mixed-profile": ["tests/mla/e2e/window_mixed/test_mla_window_mixed_profile.py"],
    "window-mixed-tools": [
        "tests/mla/unit/tools/test_mla_window_mixed_tools.py",
        "tests/mla/unit/tools/test_mla_validation_policy.py::test_mixed_average_weights_entries_and_keeps_regressions",
        "tests/mla/unit/tools/test_mla_validation_policy.py::test_mixed_average_rejects_incomplete_evidence",
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_success_failure_interruption_and_resume[{suite}]"
            for suite in ("window-mixed-tools", "window-mixed-cpu", "window-mixed-sanitize")
        ],
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_dependencies_changing_during_execution_fail[{suite}]"
            for suite in ("window-mixed-cpu", "window-mixed-sanitize")
        ],
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_device_groups_use_new_probe_and_device_identity[{suite}]"
            for suite in ("window-mixed-check", "window-mixed-public")
        ],
        "tests/mla/unit/tools/test_mla_validation.py::test_performance_always_executes_on_resume[window-mixed-profile]",
        "tests/mla/unit/tools/test_mla_validation.py::test_evidence_missing_changed_failed_or_different_identity_cannot_resume",
    ],
    "window-tnd-cpu": [
        "tests/mla/unit/window_tnd/test_flash_attn_npu_v3_mla_window_tnd_metadata.py"
    ],
    "window-tnd-sanitize": [
        "tests/mla/unit/window_tnd/test_flash_attn_npu_v3_mla_window_tnd_metadata.py"
    ],
    "window-tnd-check": [
        "tests/mla/e2e/window_tnd/test_flash_attn_npu_v3_mla_window_tnd_public.py",
        "-k",
        "representative or cost_boundary",
    ],
    "window-tnd-public": [
        "tests/mla/e2e/window_tnd/test_flash_attn_npu_v3_mla_window_tnd_public.py",
        "-k",
        "not representative and not cost_boundary and not release_trap",
    ],
    "window-tnd-trap": [
        "tests/mla/e2e/window_tnd/test_flash_attn_npu_v3_mla_window_tnd_public.py",
        "-k",
        "release_trap",
    ],
    "window-tnd-profile": ["tests/mla/e2e/window_tnd/test_mla_window_tnd_profile.py"],
    "window-tnd-tools": [
        "tests/mla/unit/tools/test_mla_window_tnd_tools.py",
        "tests/mla/unit/tools/test_mla_validation_policy.py::test_varlen_average_retains_regressions_and_failure",
        "tests/mla/unit/tools/test_mla_validation_policy.py::test_varlen_average_rejects_incomplete_evidence",
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_success_failure_interruption_and_resume[{suite}]"
            for suite in ("window-tnd-tools", "window-tnd-cpu", "window-tnd-sanitize")
        ],
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_dependencies_changing_during_execution_fail[{suite}]"
            for suite in ("window-tnd-cpu", "window-tnd-sanitize")
        ],
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_device_groups_use_new_probe_and_device_identity[{suite}]"
            for suite in ("window-tnd-check", "window-tnd-public")
        ],
        "tests/mla/unit/tools/test_mla_validation.py::test_performance_always_executes_on_resume[window-tnd-profile]",
        "tests/mla/unit/tools/test_mla_validation.py::test_evidence_missing_changed_failed_or_different_identity_cannot_resume",
    ],
    "window-bsnd-cpu": ["tests/mla/unit/window_bsnd/test_flash_attn_npu_v3_mla_window_metadata.py"],
    "window-bsnd-sanitize": [
        "tests/mla/unit/window_bsnd/test_flash_attn_npu_v3_mla_window_metadata.py"
    ],
    "window-bsnd-check": [
        "tests/mla/e2e/window_bsnd/test_flash_attn_npu_v3_mla_window_public.py",
        "-k",
        "representative or cost_boundary",
    ],
    "window-bsnd-public": [
        "tests/mla/e2e/window_bsnd/test_flash_attn_npu_v3_mla_window_public.py",
        "-k",
        "not representative and not cost_boundary and not release_trap",
    ],
    "window-bsnd-trap": [
        "tests/mla/e2e/window_bsnd/test_flash_attn_npu_v3_mla_window_public.py",
        "-k",
        "release_trap",
    ],
    "window-bsnd-profile": ["tests/mla/e2e/window_bsnd/test_mla_window_bsnd_profile.py"],
    "window-bsnd-tools": [
        "tests/mla/unit/tools/test_mla_window_bsnd_tools.py",
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_success_failure_interruption_and_resume[{suite}]"
            for suite in ("window-bsnd-tools", "window-bsnd-cpu", "window-bsnd-sanitize")
        ],
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_dependencies_changing_during_execution_fail[{suite}]"
            for suite in ("window-bsnd-cpu", "window-bsnd-sanitize")
        ],
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_device_groups_use_new_probe_and_device_identity[{suite}]"
            for suite in ("window-bsnd-check", "window-bsnd-public")
        ],
        "tests/mla/unit/tools/test_mla_validation.py::test_performance_always_executes_on_resume[window-bsnd-profile]",
        "tests/mla/unit/tools/test_mla_validation.py::test_evidence_missing_changed_failed_or_different_identity_cannot_resume",
    ],
    "window-check": [WINDOW_ATTENTION, "-k", "representative or cost_boundary"],
    "window-attention": [WINDOW_ATTENTION, "-k", "not representative and not cost_boundary"],
    "window-profile": ["tests/mla/unit/window_bsnd/test_mla_window_profile.py"],
    "window-device-tools": [
        "tests/mla/unit/tools/test_mla_window_device_tools.py",
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_device_groups_use_new_probe_and_device_identity[{suite}]"
            for suite in ("window-check", "window-attention", "window-profile")
        ],
        "tests/mla/unit/tools/test_mla_validation.py::test_standalone_causal_smoke_never_requires_old_production_artifacts[window-check]",
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::test_causal_probe_identity_change_is_not_accepted[window-{fault}]"
            for fault in ("source", "dependency", "library", "loaded", "missing")
        ],
        "tests/mla/unit/tools/test_mla_validation.py::test_performance_always_executes_on_resume[window-profile]",
        "tests/mla/unit/tools/test_mla_validation.py::test_success_failure_interruption_and_resume[window-device-tools]",
        "tests/mla/unit/tools/test_mla_reference_cache.py::test_changed_identity_recomputes[window]",
        *[
            f"tests/mla/unit/tools/test_mla_reference_cache.py::test_cold_and_hit_equal_uncached_equations[options2-3-5-False-{dtype}]"
            for dtype in ("dtype0", "dtype1")
        ],
    ],
    "window-cpu": [WINDOW_CPU],
    "window-sanitize": [WINDOW_CPU],
    "window-tools": [
        "tests/mla/unit/tools/test_mla_window_build.py",
        *[
            f"tests/mla/unit/tools/test_mla_validation.py::{name}[{suite}]"
            for name in (
                "test_success_failure_interruption_and_resume",
                "test_causal_dependencies_changing_during_execution_fail",
            )
            for suite in ("window-cpu", "window-sanitize")
        ],
        "tests/mla/unit/tools/test_mla_validation.py::test_causal_selection_and_isolated_cpu_environment[window-cpu-0]",
        "tests/mla/unit/tools/test_mla_validation.py::test_causal_selection_and_isolated_cpu_environment[window-sanitize-1]",
        "tests/mla/unit/tools/test_mla_validation.py::test_causal_driver_checks_protocol_and_final_sanitizer_exit",
    ],
    "causal-stage-tools": ["tests/mla/unit/tools/test_mla_causal_stage_performance.py"],
    "causal-paged-tnd-cpu": [
        "tests/mla/unit/causal_paged_tnd/test_flash_attn_npu_v3_mla_causal_paged_tnd_metadata.py"
    ],
    "causal-paged-tnd-sanitize": [
        "tests/mla/unit/causal_paged_tnd/test_flash_attn_npu_v3_mla_causal_paged_tnd_metadata.py"
    ],
    "causal-paged-tnd-check": [
        "tests/mla/e2e/causal_paged_tnd/test_flash_attn_npu_v3_mla_causal_paged_tnd_public.py",
        "-k",
        "representative",
    ],
    "causal-paged-tnd-public": [
        "tests/mla/e2e/causal_paged_tnd/test_flash_attn_npu_v3_mla_causal_paged_tnd_public.py",
        "-k",
        "not representative and not release_trap",
    ],
    "causal-paged-tnd-trap": [
        "tests/mla/e2e/causal_paged_tnd/test_flash_attn_npu_v3_mla_causal_paged_tnd_public.py",
        "-k",
        "release_trap",
    ],
    "causal-paged-tnd-profile": [
        "tests/mla/e2e/causal_paged_tnd/test_mla_causal_paged_tnd_profile.py"
    ],
    "causal-paged-cpu": [
        "tests/mla/unit/causal_paged/test_flash_attn_npu_v3_mla_causal_paged_metadata.py"
    ],
    "causal-paged-sanitize": [
        "tests/mla/unit/causal_paged/test_flash_attn_npu_v3_mla_causal_paged_metadata.py"
    ],
    "paged-shared-cpu": [
        f"tests/test_flash_attn_npu_v3_mla_{name}.py"
        for name in ("paged_schedule", "paged_metadata", "paged_metadata_consumer")
    ],
    "paged-shared-sanitize": [
        f"tests/test_flash_attn_npu_v3_mla_{name}.py"
        for name in ("paged_schedule", "paged_metadata", "paged_metadata_consumer")
    ],
    "causal-paged-tools": [
        "tests/mla/unit/tools/test_mla_causal_paged_profile_driver.py",
        "tests/mla/unit/tools/test_mla_paged_reference_cache.py",
        "tests/mla/unit/tools/test_mla_native_dispatch.py",
        "tests/mla/unit/tools/test_mla_validation_policy.py::test_mixed_average_weights_entries_and_keeps_regressions",
        "tests/mla/unit/tools/test_mla_validation_policy.py::test_mixed_average_rejects_incomplete_evidence",
    ],
    "causal-paged-check": [
        "tests/mla/e2e/causal_paged/test_flash_attn_npu_v3_mla_causal_paged_public.py",
        "-k",
        "representative",
    ],
    "causal-paged-public": [
        "tests/mla/e2e/causal_paged/test_flash_attn_npu_v3_mla_causal_paged_public.py",
        "-k",
        "not representative and not release_trap",
    ],
    "causal-paged-trap": [
        "tests/mla/e2e/causal_paged/test_flash_attn_npu_v3_mla_causal_paged_public.py",
        "-k",
        "release_trap",
    ],
    "causal-paged-profile": ["tests/mla/e2e/causal_paged/test_mla_causal_paged_profile.py"],
    "causal-mixed-boundary": [
        "tests/mla/e2e/causal_paged/test_flash_attn_npu_v3_mla_causal_paged_public.py",
        "-k",
        "static_and_unopened_boundaries",
    ],
    "causal-mixed-tools": [
        "tests/mla/unit/tools/test_mla_validation_policy.py",
        "-k",
        "mixed_average",
    ],
    "causal-mixed-cpu": [
        "tests/mla/unit/causal_mixed/test_flash_attn_npu_v3_mla_causal_mixed_metadata.py"
    ],
    "causal-mixed-sanitize": [
        "tests/mla/unit/causal_mixed/test_flash_attn_npu_v3_mla_causal_mixed_metadata.py"
    ],
    "causal-mixed-check": [
        "tests/mla/e2e/causal_mixed/test_flash_attn_npu_v3_mla_causal_mixed_public.py",
        "-k",
        "representative",
    ],
    "causal-mixed-public": [
        "tests/mla/e2e/causal_mixed/test_flash_attn_npu_v3_mla_causal_mixed_public.py",
        "-k",
        "not representative and not release_trap",
    ],
    "causal-mixed-trap": [
        "tests/mla/e2e/causal_mixed/test_flash_attn_npu_v3_mla_causal_mixed_public.py",
        "-k",
        "release_trap",
    ],
    "causal-mixed-profile": ["tests/mla/e2e/causal_mixed/test_mla_causal_mixed_profile.py"],
    "causal-tnd-cpu": [
        "tests/mla/unit/causal_tnd/test_flash_attn_npu_v3_mla_causal_tnd_metadata.py"
    ],
    "causal-tnd-sanitize": [
        "tests/mla/unit/causal_tnd/test_flash_attn_npu_v3_mla_causal_tnd_metadata.py"
    ],
    "causal-tnd-check": [
        "tests/mla/e2e/causal_tnd/test_flash_attn_npu_v3_mla_causal_tnd_public.py",
        "-k",
        "representative",
    ],
    "causal-tnd-public": [
        "tests/mla/e2e/causal_tnd/test_flash_attn_npu_v3_mla_causal_tnd_public.py",
        "-k",
        "not representative and not release_trap",
    ],
    "causal-tnd-trap": [
        "tests/mla/e2e/causal_tnd/test_flash_attn_npu_v3_mla_causal_tnd_public.py",
        "-k",
        "release_trap",
    ],
    "causal-tnd-profile": ["tests/mla/e2e/causal_tnd/test_mla_causal_tnd_profile.py"],
    "causal-tnd-boundary": [
        "tests/mla/e2e/causal_paged/test_flash_attn_npu_v3_mla_causal_paged_public.py",
        "-k",
        "static_and_unopened_boundaries",
    ],
    "used-shared-cpu": USED_SHARED_CPU,
    "used-shared-sanitize": USED_SHARED_CPU,
    "causal-bsnd-cpu": [
        "tests/mla/unit/causal_bsnd/test_flash_attn_npu_v3_mla_causal_metadata.py",
        "tests/mla/unit/used/test_flash_attn_npu_v3_mla_used_metadata_consumer.py",
    ],
    "causal-bsnd-sanitize": [
        "tests/mla/unit/causal_bsnd/test_flash_attn_npu_v3_mla_causal_metadata.py",
        "tests/mla/unit/used/test_flash_attn_npu_v3_mla_used_metadata_consumer.py",
    ],
    "causal-bsnd-public": [
        "tests/mla/e2e/causal_bsnd/test_flash_attn_npu_v3_mla_causal_public.py",
        "-k",
        "not release_trap",
    ],
    "causal-bsnd-check": [
        "tests/mla/e2e/causal_bsnd/test_flash_attn_npu_v3_mla_causal_public.py",
        "-k",
        "representative",
    ],
    "causal-bsnd-trap": [
        "tests/mla/e2e/causal_bsnd/test_flash_attn_npu_v3_mla_causal_public.py",
        "-k",
        "release_trap",
    ],
    "causal-bsnd-profile": [
        "tests/mla/e2e/causal_bsnd/test_mla_causal_bsnd_profile.py",
        "-k",
        "not native_preflight",
    ],
    "causal-bsnd-native-check": [
        "tests/mla/e2e/causal_bsnd/test_mla_causal_bsnd_profile.py",
        "-k",
        "native_preflight",
    ],
    "causal-bsnd-legacy": [
        "tests/mla/e2e/bsnd/test_flash_attn_npu_v3_mla_bsnd_used.py",
        "tests/mla/e2e/dense/test_flash_attn_npu_v3_mla_kvcache.py",
        "-k",
        "descriptor_rejections or tensor_rejections or explicit_metadata_rejected or word_overflow or fullgraph or cross_stream_allocator_lifetime or metadata_rejects_unsupported or metadata_length_tensor_boundary",
    ],
    "public": [PUBLIC, "-k", "not public_release_trap"],
    "trap": [PUBLIC, "-k", "public_release_trap"],
    "device": [
        "tests/mla/e2e/paged_tnd/test_flash_attn_npu_v3_mla_paged_tnd_production.py",
        "-k",
        "not release_trap",
    ],
    "bsnd": [
        "tests/mla/e2e/paged_bsnd/test_flash_attn_npu_v3_mla_paged_public.py",
        "-k",
        "not public_release_trap",
    ],
    "metadata": ["tests/mla/e2e/paged_bsnd/test_flash_attn_npu_v3_mla_paged_metadata_public.py"],
    "ordinary": [
        "tests/mla/e2e/dense/test_flash_attn_npu_v3_mla_contract.py",
        "tests/test_flash_attn_npu_v3_metadata_signature.py",
        "tests/mla/e2e/dense/test_flash_attn_npu_v3_mla_metadata_dispatch.py",
        "tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_func_metadata_bsnd",
        "tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_varlen_func_metadata_tnd[data_type0-1-1-1-512-1024-128-True]",
        "tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_kvcache_metadata_bsnd[data_type0-1-1-1-1024-1024-128-128-False]",
        "tests/test_flash_attn_npu_v3_metadata.py::test_flash_attn_kvcache_metadata_flash_decode[data_type1-8-1-64-2-False-0.0-0]",
        "tests/test_torch_compile_fa3.py::test_fa3_910_scheduler_metadata_torch_compile_correctness",
        "tests/test_torch_compile_fa3.py::test_fa3_910_varlen_torch_compile_correctness",
        "tests/test_torch_compile_fa3.py::test_fa3_910_fixed_torch_compile_correctness",
    ],
    "nonpaged": [
        "tests/mla/e2e/mixed/test_flash_attn_npu_v3_mla_mixed_metadata_public.py",
        "tests/mla/e2e/dense/test_flash_attn_npu_v3_mla_kvcache.py",
        "-k",
        "not public_source_accuracy and not public_rounding and not reuse_scales_streams_and_allocator and not shared_metadata_host_threads and not public_runtime_error",
        *[
            f"tests/mla/unit/paged_tnd/test_flash_attn_npu_v3_mla_tnd_paged_attention.py::{name}"
            for name in (
                "test_strides",
                "test_empty_cache",
                "test_stale_lengths_and_regeneration",
                "test_corruption_precedes_attention",
            )
        ],
    ],
    "cpu": [
        "tests/mla/unit/tools/test_flash_attn_npu_v3_mla_build.py::test_generated_mla_launch",
        "tests/mla/unit/tools/test_flash_attn_npu_v3_mla_build.py::test_device_dependency_staleness",
    ],
    "tools": [
        "tests/mla/unit/tools/test_matrix.py",
        "tests/mla/unit/tools/test_organization.py",
        "tests/mla/unit/tools/test_mla_validation.py",
        "tests/mla/unit/tools/test_mla_source_compare.py",
        "tests/mla/unit/tools/test_mla_codex_loop.py",
        "tests/mla/unit/tools/test_mla_reference_cache.py",
        "tests/mla/unit/tools/test_mla_validation_dependencies.py",
        "tests/mla/unit/tools/test_mla_paged_reference_cache.py",
        "tests/mla/unit/tools/test_mla_native_dispatch.py",
        "tests/mla/unit/tools/test_mla_validation_policy.py",
        "tests/common/test_golden_cache.py",
    ],
    "reference": ["tests/mla/unit/reference/test_flash_attn_npu_v3_mla_reference.py"],
    "causal-smoke": [
        "tests/mla/unit/causal_bsnd/test_flash_attn_npu_v3_mla_causal_attention.py",
        "-k",
        "cost_and_mask_smoke or amla_and_masked_max_changes",
    ],
    "causal-attention": [
        "tests/mla/unit/causal_bsnd/test_flash_attn_npu_v3_mla_causal_attention.py"
    ],
    "causal-profile": ["tests/mla/unit/causal_bsnd/test_mla_causal_profile.py"],
    "shared-profile": ["tests/mla/unit/dense/test_mla_shared_profile.py"],
    "shared-dense": [
        f"tests/test_flash_attn_npu_v3_mla_{name}.py"
        for name in ("dense", "multiblock", "kvblocks", "longkv")
    ],
    "shared-nonpaged": [
        *[
            f"tests/test_flash_attn_npu_v3_mla_{name}.py"
            for name in (
                "bsnd_attention",
                "tnd_attention",
                "tnd_bsnd_attention",
                "used_attention",
                "bsnd_stride",
                "tnd_stride",
            )
        ],
        "-k",
        "not runtime and not trap",
    ],
    "shared-paged": [
        *[
            f"tests/test_flash_attn_npu_v3_mla_{name}.py"
            for name in ("paged_attention", "tnd_paged_attention")
        ],
        "-k",
        "task_ring or strides or empty or zero or scale_halves or reuse_with_new or stream_and_allocator",
    ],
    "causal-cpu": [CAUSAL_CPU],
    "causal-sanitize": [CAUSAL_CPU],
}
SUITES.update(
    {
        "split-kv-sanitize": ["tests/mla/unit/split_kv/test_split_kv.py"],
        "split-kv-tools": ["tests/mla/unit/tools/test_mla_split_kv_tools.py"],
    }
)
SUITES.update(INCREMENTAL_SUITES)
USED_SHARED_CPU = [relocated(path) for path in USED_SHARED_CPU]

SUITES = {name: [relocated(arg) for arg in args] for name, args in SUITES.items()}
SUITES["e2e"] = combined(E2E, E2E)
SUITES.update({f"unit-{name}": paths for name, paths in UNITS.items()})
SUITES.update({f"detail-{name}": paths for name, paths in DIAGNOSTIC_E2E.items()})

PERFORMANCE_SUITES = {
    "prefill-bsnd-profile",
    "prefill-bsnd-profile-check",
    "prefill-profile",
    "auto-split-paged-tnd-profile",
    "auto-split-paged-bsnd-profile",
    "split-paged-bsnd-profile",
    "split-paged-tnd-profile",
    "auto-split-mixed-profile",
    "split-mixed-profile",
    "auto-split-tnd-profile",
    "auto-split-bsnd-profile",
    "split-bsnd-profile",
    "split-tnd-profile",
    "split-kv-merge-profile",
    "split-kv-attention-profile",
    "window-paged-tnd-profile",
    "window-paged-profile",
    "window-mixed-profile",
    "window-tnd-profile",
    "window-bsnd-profile",
    "window-profile",
    "causal-paged-tnd-profile",
    "causal-paged-profile",
    "causal-tnd-profile",
    "causal-mixed-profile",
    "causal-profile",
    "shared-profile",
    "causal-bsnd-profile",
    "causal-bsnd-native-check",
}
WINDOW_DEVICE_SUITES = {
    "window-mixed-check",
    "window-tnd-check",
    "window-mixed-public",
    "window-tnd-public",
    "window-check",
    "window-attention",
    "window-profile",
    "window-bsnd-check",
    "window-bsnd-public",
}
STANDALONE_CHECK_SUITES = {
    "prefill-check",
    "prefill-profile-check",
    "unit-prefill_attention",
    "prefill-profile",
    "split-kv-merge-check",
    "causal-smoke",
    "causal-check",
    "window-check",
    "split-kv-attention-check",
}
CAUSAL_DEVICE_SUITES = {
    "causal-tnd-check",
    "causal-mixed-check",
    "causal-mixed-public",
    "causal-tnd-public",
    "causal-tnd-profile",
    "causal-mixed-profile",
    "causal-smoke",
    "causal-attention",
    "causal-profile",
    "causal-check",
    "causal-bsnd-public",
    "causal-bsnd-check",
    "causal-bsnd-profile",
}
CPU_SUITES = {
    "prefill-bsnd-tools",
    "unit-prefill_bsnd",
    "prefill-bsnd-sanitize",
    "prefill-tools",
    "split-stage-tools",
    "auto-split-paged-tnd-tools",
    "auto-split-paged-bsnd-tools",
    "auto-split-paged-tnd-protocol-tools",
    "auto-split-paged-bsnd-protocol-tools",
    "split-paged-bsnd-tools",
    "split-paged-tnd-tools",
    "split-paged-tnd-protocol-tools",
    "split-paged-bsnd-protocol-tools",
    "auto-split-mixed-tools",
    "auto-split-mixed-protocol-tools",
    "split-mixed-tools",
    "auto-split-tnd-tools",
    "auto-split-bsnd-tools",
    "unit-auto_split_bsnd_metadata",
    "auto-split-metadata-sanitize",
    "auto-split-metadata-tools",
    "auto-split-metadata-check",
    "unit-auto_split_mixed_protocol",
    "auto-split-mixed-protocol-sanitize",
    "auto-split-mixed-protocol-check",
    "unit-auto_split_paged_tnd_protocol",
    "unit-auto_split_paged_bsnd_protocol",
    "auto-split-paged-tnd-protocol-sanitize",
    "auto-split-paged-bsnd-protocol-sanitize",
    "auto-split-paged-tnd-protocol-check",
    "auto-split-paged-bsnd-protocol-check",
    "unit-split_paged_tnd_protocol",
    "split-paged-tnd-protocol-sanitize",
    "split-paged-tnd-protocol-check",
    "unit-split_paged_bsnd_protocol",
    "split-paged-bsnd-protocol-sanitize",
    "split-paged-bsnd-protocol-check",
    "unit-split_mixed_protocol",
    "split-mixed-protocol-sanitize",
    "split-mixed-protocol-check",
    "unit-auto_split_tnd_metadata",
    "auto-split-tnd-metadata-sanitize",
    "auto-split-tnd-metadata-check",
    "unit-auto_split_tnd_plan",
    "auto-split-tnd-plan-sanitize",
    "auto-split-tnd-plan-check",
    "unit-auto_split_kv_plan",
    "auto-split-kv-plan-sanitize",
    "split-mixed-protocol-tools",
    "auto-split-tnd-metadata-tools",
    "auto-split-tnd-plan-tools",
    "auto-split-kv-plan-tools",
    "auto-split-kv-plan-check",
    "split-bsnd-cpu",
    "split-tnd-cpu",
    "split-tnd-sanitize",
    "split-bsnd-sanitize",
    "split-bsnd-tools",
    "split-tnd-tools",
    "unit-split_tnd_plan",
    "split-tnd-plan-sanitize",
    "split-tnd-plan-check",
    "unit-split_kv_plan",
    "split-kv-plan-sanitize",
    "split-tnd-plan-tools",
    "split-kv-plan-tools",
    "split-kv-merge-tools",
    "split-kv-attention-tools",
    "unit-split_kv",
    "split-kv-sanitize",
    "split-kv-tools",
    "unit-tools",
    "unit-window_paged_tnd",
    "window-paged-tnd-cpu",
    "window-paged-cpu",
    "window-mixed-cpu",
    "window-tnd-cpu",
    "window-paged-tnd-sanitize",
    "window-paged-sanitize",
    "window-mixed-sanitize",
    "window-tnd-sanitize",
    "window-paged-tnd-tools",
    "window-paged-tools",
    "window-mixed-tools",
    "window-tnd-tools",
    "window-bsnd-cpu",
    "window-bsnd-sanitize",
    "window-bsnd-tools",
    "window-device-tools",
    "window-cpu",
    "window-sanitize",
    "window-tools",
    "causal-stage-tools",
    "causal-paged-tools",
    "causal-paged-tnd-cpu",
    "causal-paged-tnd-sanitize",
    "causal-paged-cpu",
    "causal-paged-sanitize",
    "paged-shared-cpu",
    "paged-shared-sanitize",
    "causal-mixed-tools",
    "causal-tnd-cpu",
    "causal-mixed-cpu",
    "causal-mixed-sanitize",
    "causal-tnd-sanitize",
    "used-shared-cpu",
    "used-shared-sanitize",
    "cpu",
    "tools",
    "unit-tools",
    "reference",
    "causal-cpu",
    "causal-sanitize",
    "causal-bsnd-cpu",
    "causal-bsnd-sanitize",
}


def suite_dependencies(suite):
    if suite in {"unit-prefill_bsnd", "prefill-bsnd-sanitize"}:
        from tests.mla.support.mla_prefill_bsnd_build import toolchain_identity

        return toolchain_identity(suite.endswith("sanitize"))
    if suite in {
        "unit-auto_split_paged_tnd_protocol",
        "unit-auto_split_paged_bsnd_protocol",
        "auto-split-paged-tnd-protocol-sanitize",
        "auto-split-paged-bsnd-protocol-sanitize",
        "auto-split-paged-tnd-protocol-check",
        "auto-split-paged-bsnd-protocol-check",
        "unit-split_paged_tnd_protocol",
        "split-paged-tnd-protocol-sanitize",
        "split-paged-tnd-protocol-check",
        "unit-split_paged_bsnd_protocol",
        "split-paged-bsnd-protocol-sanitize",
        "split-paged-bsnd-protocol-check",
    }:
        from tests.mla.support.mla_split_paged_bsnd_build import paged_split_toolchain_identity

        return paged_split_toolchain_identity(suite.endswith("sanitize"))
    if suite in {
        "unit-split_kv",
        "split-kv-sanitize",
        "split-bsnd-cpu",
        "split-tnd-cpu",
        "split-tnd-sanitize",
        "split-bsnd-sanitize",
        "unit-auto_split_mixed_protocol",
        "auto-split-mixed-protocol-sanitize",
        "auto-split-mixed-protocol-check",
        "unit-split_mixed_protocol",
        "split-mixed-protocol-sanitize",
        "split-mixed-protocol-check",
        "unit-auto_split_tnd_metadata",
        "auto-split-tnd-metadata-sanitize",
        "auto-split-tnd-metadata-check",
        "unit-auto_split_tnd_plan",
        "auto-split-tnd-plan-sanitize",
        "auto-split-tnd-plan-check",
        "unit-auto_split_kv_plan",
        "unit-auto_split_bsnd_metadata",
        "auto-split-metadata-sanitize",
        "auto-split-metadata-check",
        "auto-split-kv-plan-sanitize",
        "auto-split-kv-plan-check",
        "unit-split_tnd_plan",
        "split-tnd-plan-sanitize",
        "split-tnd-plan-check",
        "unit-split_kv_plan",
        "split-kv-plan-sanitize",
    }:
        from tests.mla.support.mla_split_kv_build import split_toolchain_identity

        return split_toolchain_identity(suite.endswith("sanitize"))
    if suite in {
        "window-cpu",
        "window-sanitize",
        "window-bsnd-cpu",
        "window-bsnd-sanitize",
        "unit-window_paged_tnd",
        "window-paged-tnd-cpu",
        "window-paged-cpu",
        "window-mixed-cpu",
        "window-tnd-cpu",
        "window-paged-tnd-sanitize",
        "window-paged-sanitize",
        "window-mixed-sanitize",
        "window-tnd-sanitize",
    }:
        from tests.mla.support.mla_window_build import window_toolchain_identity

        return window_toolchain_identity(suite.endswith("sanitize"))
    if suite not in {
        "causal-paged-tnd-cpu",
        "causal-paged-tnd-sanitize",
        "causal-paged-cpu",
        "causal-paged-sanitize",
        "paged-shared-cpu",
        "paged-shared-sanitize",
        "causal-cpu",
        "causal-sanitize",
        "causal-tnd-cpu",
        "causal-mixed-cpu",
        "causal-mixed-sanitize",
        "causal-tnd-sanitize",
        "causal-bsnd-cpu",
        "causal-bsnd-sanitize",
        "used-shared-cpu",
        "used-shared-sanitize",
    }:
        return {}
    from tests.mla.support.mla_causal_build import toolchain_identity

    return toolchain_identity(suite.endswith("sanitize"))


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sources():
    # Conservative closure, deliberately excluding docs and evidence directories.
    # Shared helpers, setup flags, fixtures and untracked source changes count.
    paths = [ROOT / "setup.py", ROOT / "pyproject.toml", ROOT / "scripts/mla_codex_loop.py"]
    for directory in ("tests", "csrc/ascend910", "csrc/catlass/include", "flash_attn_npu_3"):
        paths.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.suffix in {".py", ".cpp", ".h", ".hpp", ".aicpu", ".asc", ".json"}
        )
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another validation process holds {path}") from exc
        yield stream.fileno()


def execute(command, env, directory, lock_fds=()):
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = dict(command=command, cwd=str(ROOT), status="running", pid=os.getpid())
    record["environment"] = {
        key: value
        for key, value in env.items()
        if key.startswith(("MLA_", "GOLDEN_CACHE_", "FLASH_ATTN_"))
        or key
        in {
            "ASCEND_HOME_PATH",
            "ASCEND_TOOLKIT_HOME",
            "ASCEND_OPP_PATH",
            "ASCEND_AICPU_PATH",
            "ASCEND_RT_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "TORCH_DEVICE_BACKEND_AUTOLOAD",
            "PYTEST_ADDOPTS",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "CI_TORCH_SEED",
        }
    }
    save(directory / "command.json", record)
    with (directory / "output.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=lock_fds,
        )
        record["child_pid"] = process.pid
        save(directory / "command.json", record)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 130
    record.update(
        returncode=code,
        seconds=time.monotonic() - started,
        status="passed" if code == 0 else "interrupted" if code == 130 else "failed",
    )
    save(directory / "command.json", record)
    return record


def pytest_command(suite, directory, collect=False, *, split_stage=False):
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-x",
        "-q",
        *SUITES[suite],
        "-p",
        "tests.mla.support.mla_validation_plugin",
        "-o",
        "addopts=",
        "--random-sample=0",
    ]
    if split_stage:
        command += ["-k", "native_public_profile"]
    if collect:
        return command + ["--collect-only"]
    return command + [f"--basetemp={directory / 'tmp'}", f"--junitxml={directory / 'results.xml'}"]


def environment(args, directory, suite):
    env = dict(os.environ)
    # Inherited pytest/plugin or old step-directory settings must not silently
    # change coverage or redirect fixtures to unrelated artifacts.
    for key in list(env):
        if key.startswith(("MLA_", "GOLDEN_CACHE_", "PYTEST_")) and key not in {
            "MLA_CANN_SOURCE_ROOT",
            "MLA_FA3_SOURCE_ROOT",
            "MLA_WORK_DIR",
            "MLA_NATIVE_ROOT",
            "MLA_NATIVE_DEPS",
        }:
            env.pop(key)
    env.update(
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        ASCEND_RT_VISIBLE_DEVICES=args.device,
        FLASH_ATTN_BUILD_VERSION="v3",
        FLASH_ATTN_BUILD_NPU="910",
        FLASH_ATTN_SKIP_SUBMODULE_INIT="1",
        PYTEST_ADDOPTS="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        MLA_VALIDATION_RESULT=str(directory),
        MLA_VALIDATION_NPU="0" if suite in CPU_SUITES else "1",
        GOLDEN_CACHE_MODE="off" if args.golden == "off" else "cache",
        GOLDEN_CACHE_REFRESH="1" if args.golden == "refresh" else "0",
        GOLDEN_CACHE_DIR=str(args.artifacts / "golden"),
        GOLDEN_CACHE_STATS_FILE=str(directory / "golden.tsv"),
        MLA_PAGED_ATTENTION_DIR=str(directory / "probes"),
        MLA_PAGED_ATTENTION_REUSE="1",
        MLA_PAGED_METADATA_LIBRARY=str(args.artifacts / "producer/metadata.so"),
    )
    if suite in {"unit-prefill_bsnd", "prefill-bsnd-sanitize"}:
        env["MLA_SCHEDULE_ARTIFACT_DIR"] = str(directory / "schedule")
        env["MLA_SCHEDULE_SANITIZE"] = str(int(suite.endswith("sanitize")))
    if suite in {"prefill-bsnd-profile", "prefill-bsnd-profile-check"}:
        env["MLA_PREFILL_BSND_PROFILE_CHECK"] = str(int(suite.endswith("check")))
    for variable, child in {
        "MLA_PAGED_METADATA_DEVICE_DIR": "producer",
        "MLA_PAGED_SCHEDULE_DIR": "source",
        "MLA_PAGED_PUBLIC_DIR": "public",
        "MLA_PAGED_CONSUMER_CPU_DIR": "consumer",
        "MLA_PAGED_PRODUCTION_DIR": "production",
        "MLA_TND_PAGED_KVCACHE_DIR": "kvcache",
    }.items():
        env[variable] = str(directory / child)
    if getattr(args, "split_stage_performance", False) and suite in SPLIT_STAGE_SUITES:
        env["MLA_SPLIT_STAGE_PERFORMANCE"] = "1"
        env["MLA_SPLIT_STAGE_SUITE"] = suite
    if suite == "e2e":
        env["MLA_E2E_SELECTION"] = "1"
    if getattr(args, "causal_stage_performance", False) and suite in CAUSAL_STAGE_SUITES:
        env["MLA_CAUSAL_STAGE_PERFORMANCE"] = "1"
    if getattr(args, "window_stage_performance", False) and suite in WINDOW_STAGE_SUITES:
        env["MLA_WINDOW_STAGE_PERFORMANCE"] = "1"
    env["MLA_VALIDATION_MAX_CASES"] = str(getattr(args, "max_cases", DEFAULT_MAX_CASES))
    env["MLA_FULL_MATRIX"] = "1" if getattr(args, "full_matrix", False) else "0"
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0" if suite in CPU_SUITES else "1"
    if suite in {
        "prefill-check",
        "unit-prefill_attention",
        "prefill-profile",
        "prefill-profile-check",
    }:
        env["MLA_PREFILL_DIR"] = str(directory / "probes")
        env["MLA_PREFILL_PROFILE_CHECK"] = str(int(suite == "prefill-profile-check"))
    if suite in CAUSAL_DEVICE_SUITES:
        env["MLA_CAUSAL_ATTENTION_DIR"] = str(directory / "probes")
    if suite in WINDOW_DEVICE_SUITES:
        env["MLA_WINDOW_ATTENTION_DIR"] = str(directory / "probes")
    if suite in SPLIT_KV_MERGE_SUITES:
        env["MLA_SPLIT_KV_MERGE_DIR"] = str(directory / "probes")
    if suite in SPLIT_KV_DEVICE_SUITES:
        env["MLA_SPLIT_KV_ATTENTION_DIR"] = str(directory / "probes")
    if suite in {
        "split-bsnd-cpu",
        "split-tnd-cpu",
        "split-tnd-sanitize",
        "split-bsnd-sanitize",
        "unit-auto_split_mixed_protocol",
        "auto-split-mixed-protocol-sanitize",
        "auto-split-mixed-protocol-check",
        "unit-auto_split_paged_tnd_protocol",
        "unit-auto_split_paged_bsnd_protocol",
        "auto-split-paged-tnd-protocol-sanitize",
        "auto-split-paged-bsnd-protocol-sanitize",
        "auto-split-paged-tnd-protocol-check",
        "auto-split-paged-bsnd-protocol-check",
        "unit-split_paged_tnd_protocol",
        "split-paged-tnd-protocol-sanitize",
        "split-paged-tnd-protocol-check",
        "unit-split_paged_bsnd_protocol",
        "split-paged-bsnd-protocol-sanitize",
        "split-paged-bsnd-protocol-check",
        "unit-split_mixed_protocol",
        "split-mixed-protocol-sanitize",
        "split-mixed-protocol-check",
        "unit-auto_split_tnd_metadata",
        "auto-split-tnd-metadata-sanitize",
        "auto-split-tnd-metadata-check",
        "unit-auto_split_tnd_plan",
        "auto-split-tnd-plan-sanitize",
        "auto-split-tnd-plan-check",
        "unit-auto_split_kv_plan",
        "unit-auto_split_bsnd_metadata",
        "auto-split-metadata-sanitize",
        "auto-split-metadata-check",
        "auto-split-kv-plan-sanitize",
        "auto-split-kv-plan-check",
        "unit-split_tnd_plan",
        "split-tnd-plan-sanitize",
        "split-tnd-plan-check",
        "unit-split_kv_plan",
        "split-kv-plan-sanitize",
        "unit-split_kv",
        "split-kv-sanitize",
        "unit-window_paged_tnd",
        "window-paged-tnd-cpu",
        "window-paged-cpu",
        "window-mixed-cpu",
        "window-tnd-cpu",
        "window-paged-tnd-sanitize",
        "window-paged-sanitize",
        "window-mixed-sanitize",
        "window-tnd-sanitize",
        "window-bsnd-cpu",
        "window-bsnd-sanitize",
        "window-cpu",
        "window-sanitize",
        "causal-paged-tnd-cpu",
        "causal-paged-tnd-sanitize",
        "causal-paged-cpu",
        "causal-paged-sanitize",
        "paged-shared-cpu",
        "paged-shared-sanitize",
        "causal-cpu",
        "causal-sanitize",
        "causal-tnd-cpu",
        "causal-mixed-cpu",
        "causal-mixed-sanitize",
        "causal-tnd-sanitize",
        "causal-bsnd-cpu",
        "causal-bsnd-sanitize",
        "used-shared-cpu",
        "used-shared-sanitize",
    }:
        env["MLA_SCHEDULE_ARTIFACT_DIR"] = str(directory / "schedule")
        env["MLA_SCHEDULE_SANITIZE"] = "1" if suite.endswith("sanitize") else "0"
        if suite in {"causal-bsnd-sanitize", "paged-shared-sanitize"}:
            env["MLA_CONSUMER_SANITIZE"] = "1"
    return env


def validate_result(directory, returncode, runtime_info=None):
    cases = json.loads((directory / "cases.json").read_text())
    root = ET.parse(directory / "results.xml").getroot()
    tests = list(root.iter("testcase"))
    failed = sum(any(t.find(kind) is not None for kind in ("failure", "error")) for t in tests)
    skipped = sum(t.find("skipped") is not None for t in tests)
    passed = len(tests) - failed - skipped
    if runtime_info:
        loaded = json.loads((directory / "loaded.json").read_text())
        if loaded != dict(path=runtime_info["extension"], sha256=runtime_info["extension_sha256"]):
            raise RuntimeError("pytest loaded a different extension")
    # Exit zero, xfail/skip or an empty selection are not acceptance evidence.
    ok = (
        returncode == 0
        and bool(cases)
        and len(cases) == len(set(cases)) == len(tests)
        and not failed
        and not skipped
    )
    return dict(
        status="passed" if ok else "failed",
        passed=passed,
        failed=failed,
        skipped=skipped,
        collected=len(cases),
        cases_sha256=digest(cases),
    )


def validate_causal_builds(directory, kind="causal"):
    """Check both fresh probes and their actual loaded identities after execution."""
    for dtype in ("fp16", "bf16"):
        base = directory / "probes" / dtype
        manifest = json.loads((base / "build-manifest.json").read_text())
        if not manifest.get(kind) or manifest.get("dtype") != dtype:
            raise RuntimeError(f"Expected a {kind} probe build for " + dtype)
        sources = {str(ROOT / p): h for p, h in manifest["sources"].items()}
        libraries = {r["path"]: r["sha256"] for r in manifest["libraries"].values()}
        loaded = json.loads((base / "loaded.json").read_text())
        if (
            not manifest["dependencies"]
            or not libraries
            or loaded != manifest["libraries"]
            or not all(matching_files(x) for x in (sources, manifest["dependencies"], libraries))
        ):
            raise RuntimeError(
                f"{kind.title()} probe sources, dependencies, outputs or loaded identities changed"
            )


def reusable(record, identity):
    if not isinstance(record, dict):
        return False
    if record.get("status") != "passed" or record.get("identity") != identity:
        return False
    try:
        evidence = record["evidence"]
        return bool(evidence) and all(
            sha256(Path(path)) == value for path, value in evidence.items()
        )
    except (KeyError, OSError):
        return False


def cache_stats(path):
    if not path.exists():
        return {}
    return dict(Counter(line.split("\t")[0] for line in path.read_text().splitlines()))


def run(args):
    args.split_stage_performance = getattr(args, "split_stage_performance", False)
    if args.split_stage_performance and (
        getattr(args, "window_stage_performance", False)
        or getattr(args, "causal_stage_performance", False)
        or not set(args.suite) <= set(SPLIT_STAGE_SUITES)
    ):
        raise ValueError("--split-stage-performance requires only M10 performance suites")
    args.window_stage_performance = getattr(args, "window_stage_performance", False)
    args.causal_stage_performance = getattr(args, "causal_stage_performance", False)
    if args.window_stage_performance and (
        args.causal_stage_performance or not set(args.suite) <= set(WINDOW_STAGE_SUITES)
    ):
        raise ValueError("--window-stage-performance requires window performance suites")
    if args.causal_stage_performance and not set(args.suite) <= set(CAUSAL_STAGE_SUITES) | {
        "causal-profile"
    }:
        raise ValueError("--causal-stage-performance requires causal performance suites")
    args.max_cases = getattr(args, "max_cases", DEFAULT_MAX_CASES)
    args.full_matrix = getattr(args, "full_matrix", False)
    if args.max_cases < 0:
        raise ValueError("--max-cases must be positive")
    args.output = args.output.resolve()
    args.artifacts = args.artifacts.resolve()
    args.suite = list(dict.fromkeys(args.suite))
    for path in (args.output, args.artifacts):
        if path.resolve().is_relative_to(ROOT / "build"):
            raise ValueError(
                "build/ is reserved for headless logs; use an external --output/--artifacts directory"
            )
    if args.resume and args.golden == "refresh":
        raise ValueError("--resume cannot be combined with --golden refresh")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    max_cases_per_suite=args.max_cases,
                    full_matrix=args.full_matrix,
                    selection_policy=POLICY_VERSION,
                    split_stage_performance=args.split_stage_performance,
                    split_stage_policy=SPLIT_STAGE_POLICY,
                    window_stage_performance=args.window_stage_performance,
                    causal_stage_performance=args.causal_stage_performance,
                    causal_stage_policy=CAUSAL_STAGE_POLICY,
                    prepare=[
                        sys.executable,
                        "-m",
                        "tests.mla.support.mla_validation_build",
                        "prepare",
                        str(args.artifacts),
                    ]
                    if set(args.suite) - CPU_SUITES
                    and not args.collect_only
                    and not set(args.suite) <= STANDALONE_CHECK_SUITES
                    else [],
                    suites={
                        name: pytest_command(
                            name,
                            args.output / "attempt-N" / name,
                            args.collect_only,
                            split_stage=args.split_stage_performance,
                        )
                        for name in args.suite
                    },
                    golden=args.golden,
                    device=args.device,
                    artifacts=str(args.artifacts),
                    resume=args.resume,
                ),
                indent=2,
            )
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        lock_fds = [stack.enter_context(locked(args.output / ".lock"))]
        needs_npu = bool(set(args.suite) - CPU_SUITES) and not args.collect_only
        # The representative component probe builds itself from source. It
        # does not call the production extension or PA producer. Full device
        # acceptance still prepares production and checks its actual runtime.
        needs_build = needs_npu and not set(args.suite) <= STANDALONE_CHECK_SUITES
        if needs_npu:
            lock_fds += [
                stack.enter_context(
                    locked(Path("/tmp") / f"flash-attn-mla-device-{args.device}.lock")
                ),
                stack.enter_context(locked(args.artifacts / ".lock")),
            ]
        attempt = args.output / f"attempt-{time.time_ns()}"
        attempt.mkdir()
        state_path = args.output / "state.json"
        try:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            if not isinstance(state, dict):
                raise ValueError("Expected a group index")
        except ValueError:
            # Preserve the damaged index for diagnosis; it cannot authorize a
            # skipped test. Existing attempt directories remain untouched.
            shutil.copyfile(state_path, attempt / "invalid-state.json")
            state = {}
        summary = dict(
            max_cases_per_suite=args.max_cases,
            full_matrix=args.full_matrix,
            selection_policy=POLICY_VERSION,
            split_stage_performance=args.split_stage_performance,
            split_stage_policy=SPLIT_STAGE_POLICY,
            window_stage_performance=args.window_stage_performance,
            causal_stage_performance=args.causal_stage_performance,
            causal_stage_policy=CAUSAL_STAGE_POLICY,
            attempt=str(attempt),
            groups={name: dict(status="not_run") for name in args.suite},
        )
        summary_path = args.output / "summary.json"
        save(summary_path, summary)
        save(
            attempt / "baseline.json",
            dict(
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            ),
        )
        runtime_info = None
        baseline = sources()
        save(attempt / "sources.json", baseline)
        if needs_npu:
            env = environment(args, attempt, "public")
            # Do not initialize the runtime in the compiler subprocess.
            build_env = dict(env, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
            for name, command, step_env in (
                (
                    "build",
                    [
                        sys.executable,
                        "-m",
                        "tests.mla.support.mla_validation_build",
                        "prepare",
                        str(args.artifacts),
                    ],
                    build_env,
                ),
                (
                    "runtime",
                    [
                        sys.executable,
                        "-m",
                        "tests.mla.support.mla_validation_build",
                        "runtime",
                        str(attempt / "runtime.json"),
                    ],
                    env,
                ),
            ):
                if name == "build" and not needs_build:
                    continue
                print(f"{name}: running; log {attempt / name / 'output.log'}", flush=True)
                result = execute(command, step_env, attempt / name, lock_fds)
                if sources() != baseline:
                    result.update(
                        status="failed", returncode=1, error="Source changed during preparation"
                    )
                summary[name] = result
                save(summary_path, summary)
                print(f"{name}: {result['status']} ({result['seconds']:.1f}s)", flush=True)
                if result["returncode"]:
                    return result["returncode"]
            runtime_info = json.loads((attempt / "runtime.json").read_text())
            driver_info = subprocess.check_output(["npu-smi", "info"], text=True, timeout=30)
            (attempt / "npu-smi.log").write_text(driver_info)
            runtime_info["driver_header"] = next(
                line.strip() for line in driver_info.splitlines() if "npu-smi" in line
            )
            runtime_info["host_kernel"] = os.uname().release
            if needs_build:
                shutil.copyfile(args.artifacts / "prepare.json", attempt / "build-manifest.json")
            else:
                summary["build"] = dict(
                    status="not_required",
                    scope="Fresh independent mask probes; no production execution",
                )
            summary["runtime_identity"] = runtime_info
        packages = {
            n: importlib.metadata.version(n) for n in ("pytest", "torch", "torch_npu", "numpy")
        }
        for suite in args.suite:
            directory = attempt / suite
            directory.mkdir()
            env = environment(args, directory, suite)
            dependencies = suite_dependencies(suite)
            save(directory / "suite-dependencies.json", dependencies)
            group_sources = suite_sources(ROOT, SUITES[suite], baseline, headless=suite == "tools")
            save(directory / "sources.json", group_sources)
            # Omit only output locations from the identity. Runtime and all
            # inherited environment values participate without exposing secrets.
            semantic_env = {
                k: v for k, v in env.items() if not k.startswith(("MLA_", "GOLDEN_CACHE_"))
            }
            identity = digest(
                dict(
                    source_schema=2,
                    sources=group_sources,
                    packages=packages,
                    runtime=runtime_info if suite not in CPU_SUITES else sys.version,
                    environment=semantic_env,
                    selection=SUITES[suite],
                    case_limit=args.max_cases,
                    full_matrix=args.full_matrix,
                    selection_policy=POLICY_VERSION,
                    split_stage_performance=args.split_stage_performance,
                    split_stage_policy=SPLIT_STAGE_POLICY,
                    window_stage_performance=args.window_stage_performance,
                    causal_stage_performance=args.causal_stage_performance,
                    causal_stage_policy=CAUSAL_STAGE_POLICY,
                    dependencies=dependencies,
                    golden=args.golden,
                    build=sha256(args.artifacts / "prepare.json") if needs_build else None,
                )
            )
            if (
                args.resume
                and not args.collect_only
                and suite not in PERFORMANCE_SUITES
                and is_e2e(SUITES[suite])
                and reusable(state.get(suite, {}), identity)
            ):
                summary["groups"][suite] = dict(
                    status="inherited",
                    from_directory=state[suite]["directory"],
                    passed=state[suite]["passed"],
                    seconds=0,
                    identity=identity,
                )
                save(summary_path, summary)
                print(f"{suite}: inherited {state[suite]['passed']} cases", flush=True)
                continue
            # Invalidate before launching, so interruption cannot resurrect an
            # older successful run after a newer failure.
            if not args.collect_only:
                state.pop(suite, None)
                save(state_path, state)
            command = pytest_command(
                suite, directory, args.collect_only, split_stage=args.split_stage_performance
            )
            if needs_build and suite not in CPU_SUITES:
                # Probe libraries are shared; numerical outputs and loaded
                # manifests belong to this attempt, not the mutable cache.
                for relative in ("probes/fp16", "probes/bf16", "producer"):
                    target = directory / relative
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(
                        args.artifacts / relative / "build-manifest.json",
                        target / "build-manifest.json",
                    )
            print(f"{suite}: running; log {directory / 'output.log'}", flush=True)
            result = execute(command, env, directory, lock_fds)
            result.update(
                directory=str(directory),
                identity=identity,
                golden=cache_stats(directory / "golden.tsv"),
            )
            if not args.collect_only:
                try:
                    result.update(
                        validate_result(
                            directory,
                            result["returncode"],
                            runtime_info if suite not in CPU_SUITES else None,
                        )
                    )
                except (OSError, ValueError, ET.ParseError, RuntimeError) as exc:
                    result.update(status="failed", error=str(exc))
            elif result["returncode"] == 0:
                cases = json.loads((directory / "cases.json").read_text())
                result.update(status="collected" if cases else "failed", collected=len(cases))
            if result["returncode"] == 130:
                result["status"] = "interrupted"
            if (
                suite_sources(ROOT, SUITES[suite], sources(), headless=suite == "tools")
                != group_sources
            ):
                result.update(status="failed", error="Source changed during validation")
            if suite_dependencies(suite) != dependencies:
                result.update(
                    status="failed", error="CPU toolchain or fixed source changed during validation"
                )
            if runtime_info and not matching_files(runtime_info["loaded_libraries"]):
                result.update(status="failed", error="Loaded libraries changed during validation")
            if needs_build:
                prepared = json.loads((args.artifacts / "prepare.json").read_text())
                if not matching_files(prepared["outputs"]) or not matching_files(
                    prepared["dependencies"]
                ):
                    result.update(
                        status="failed",
                        error="Build artifacts or dependencies changed during validation",
                    )
            if (
                suite
                in CAUSAL_DEVICE_SUITES
                | WINDOW_DEVICE_SUITES
                | SPLIT_KV_DEVICE_SUITES
                | SPLIT_KV_MERGE_SUITES
                | {
                    "prefill-check",
                    "unit-prefill_attention",
                    "prefill-profile",
                    "prefill-profile-check",
                }
                and not args.collect_only
                and result["returncode"] == 0
            ):
                try:
                    if suite in {
                        "prefill-check",
                        "unit-prefill_attention",
                        "prefill-profile",
                        "prefill-profile-check",
                    }:
                        validate_causal_builds(directory, kind="prefill_merge")
                    elif suite in SPLIT_KV_MERGE_SUITES:
                        validate_causal_builds(directory, kind="split_kv_merge")
                    elif suite in SPLIT_KV_DEVICE_SUITES:
                        validate_causal_builds(directory, kind="split_kv")
                    elif suite in WINDOW_DEVICE_SUITES:
                        validate_causal_builds(directory, kind="window")
                    else:
                        validate_causal_builds(directory)
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    result.update(status="failed", error=str(exc))
            summary["groups"][suite] = result
            if result["status"] == "passed":
                result["evidence"] = {
                    str(p): sha256(p)
                    for p in directory.rglob("*")
                    if p.is_file() and "tmp" not in p.relative_to(directory).parts
                }
                state[suite] = result
                save(state_path, state)
            save(summary_path, summary)
            save(attempt / "summary.json", summary)
            print(f"{suite}: {result['status']} ({result['seconds']:.1f}s)", flush=True)
            if result["status"] in {"failed", "interrupted"}:
                return result["returncode"] or 1
        save(attempt / "summary.json", summary)
    return 0


def case_limit(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("expected a positive diagnostic limit")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=work_dir() / "results",
        help="Evidence directory outside build/; attempts are never overwritten",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--suite", nargs="+", choices=SUITES, help="Explicit diagnostic or performance groups"
    )
    selection.add_argument(
        "--e2e", nargs="+", choices=E2E, help="Public regression for completed modules"
    )
    selection.add_argument(
        "--unit", nargs="+", choices=UNITS, help="Only units owned by these modules"
    )
    parser.add_argument(
        "--list", action="store_true", help="List module ownership and explicit suites"
    )
    parser.add_argument(
        "--causal-stage-performance",
        action="store_true",
        help="Limit five causal layouts to 100 total dtype/input configurations; requires stage aggregate acceptance",
    )
    parser.add_argument(
        "--window-stage-performance",
        action="store_true",
        help="Measure five window layouts within 100 total dtype/input configurations",
    )
    parser.add_argument(
        "--split-stage-performance",
        action="store_true",
        help="Measure M10 five layouts and three split modes using 92 fixed configurations",
    )
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument(
        "--max-cases",
        type=case_limit,
        default=DEFAULT_MAX_CASES,
        help="Optional diagnostic truncation; default runs the complete representative set",
    )
    limits.add_argument(
        "--full-matrix",
        action="store_true",
        help="Explicit exhaustive diagnostic combinations; performance inputs stay fixed",
    )
    parser.add_argument("--artifacts", type=Path, default=work_dir() / "mla-validation-artifacts")
    parser.add_argument("--device", default="6", choices=[str(i) for i in range(8)])
    parser.add_argument("--golden", choices=("off", "cache", "refresh"), default="cache")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without building, importing torch or writing files",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Record exact nodeids without building or executing tests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Inherit successful E2E only when identities and evidence match; units and performance execute",
    )
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(
                dict(e2e=E2E, diagnostic_e2e=DIAGNOSTIC_E2E, unit=UNITS, suites=list(SUITES)),
                indent=2,
            )
        )
        return 0
    if args.e2e:
        SUITES["e2e"] = combined(args.e2e, E2E)
        args.suite = ["e2e"]
    elif args.unit:
        args.suite = list(dict.fromkeys(f"unit-{name}" for name in args.unit))
    elif not args.suite:
        args.suite = ["e2e"]
    if args.full_matrix and args.suite == ["e2e"]:
        modules = args.e2e or E2E
        SUITES["e2e"] = combined(modules, E2E) + combined(modules, DIAGNOSTIC_E2E)
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
