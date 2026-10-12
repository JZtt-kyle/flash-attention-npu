"""Record the exact collected case IDs and the extension loaded by pytest."""

import importlib
import json
import os
from pathlib import Path

from tests.mla_dense_build import sha256


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
