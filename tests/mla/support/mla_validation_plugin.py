"""Record the exact collected case IDs and the extension loaded by pytest."""

import importlib
import json
import os
from pathlib import Path

import pytest

from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_validation_policy import (
    POLICY_VERSION,
    select_nodeids,
    select_e2e_nodeids,
)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    limit = int(os.environ.get("MLA_VALIDATION_MAX_CASES", "0"))
    candidates = [item.nodeid for item in items]
    selector = select_e2e_nodeids if os.environ.get("MLA_E2E_SELECTION") == "1" else select_nodeids
    chosen = selector(candidates, limit)
    selected = set(chosen)
    deselected = [item for item in items if item.nodeid not in selected]
    items[:] = [item for item in items if item.nodeid in selected]
    directory = Path(os.environ["MLA_VALIDATION_RESULT"])
    (directory / "selection.json").write_text(
        json.dumps(
            dict(
                policy=POLICY_VERSION,
                full_matrix=os.environ.get("MLA_FULL_MATRIX") == "1",
                selection_mode="diagnostic-cap-v2"
                if limit
                else (
                    "full-matrix-v1"
                    if os.environ.get("MLA_FULL_MATRIX") == "1"
                    else "explicit-daily-v1"
                    if os.environ.get("MLA_E2E_SELECTION") == "1"
                    else "representative-pairs-v1"
                ),
                limit=limit,
                candidates=candidates,
                selected=chosen,
                deselected=[item.nodeid for item in deselected],
            ),
            indent=2,
        )
        + "\n"
    )
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def pytest_collection_finish(session):
    directory = Path(os.environ["MLA_VALIDATION_RESULT"])
    (directory / "cases.json").write_text(
        json.dumps([item.nodeid for item in session.items], indent=2) + "\n"
    )
    if os.environ.get("MLA_VALIDATION_NPU") == "1" and not session.config.option.collectonly:
        extension = importlib.import_module("flash_attn_npu_3.flash_attn_npu_3")
        path = Path(extension.__file__).resolve()
        (directory / "loaded.json").write_text(
            json.dumps(dict(path=str(path), sha256=sha256(path)), indent=2) + "\n"
        )
