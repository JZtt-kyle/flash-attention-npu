"""Record the exact collected case IDs and the extension loaded by pytest."""

import importlib
import json
import os
from pathlib import Path

import pytest

from tests.mla_dense_build import sha256
from tests.mla_validation_policy import POLICY_VERSION, select_nodeids


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    limit = int(os.environ.get("MLA_VALIDATION_MAX_CASES", "100"))
    candidates = [item.nodeid for item in items]
    chosen = select_nodeids(candidates, limit)
    selected = set(chosen)
    deselected = [item for item in items if item.nodeid not in selected]
    items[:] = [item for item in items if item.nodeid in selected]
    directory = Path(os.environ["MLA_VALIDATION_RESULT"])
    (directory / "selection.json").write_text(
        json.dumps(
            dict(
                policy=POLICY_VERSION,
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
