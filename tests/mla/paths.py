"""Repository sources, external scratch space and stable pre-move case identities."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "tests/mla/support"
LEGACY_PATHS = json.loads((Path(__file__).with_name("legacy_paths.json")).read_text())
ORIGINAL_PATHS = {new: old for old, new in LEGACY_PATHS.items()}


def work_dir():
    """Build products and evidence are disposable; build/ holds headless logs only."""
    key = hashlib.sha256(str(ROOT).encode()).hexdigest()[:12]
    return Path(os.environ.get("MLA_WORK_DIR", f"/tmp/flash-attention-npu-mla-{key}")).resolve()


def relocated(nodeid):
    path, separator, case = nodeid.partition("::")
    return LEGACY_PATHS.get(path, path) + separator + case


def original_nodeid(nodeid):
    path, separator, case = nodeid.partition("::")
    return ORIGINAL_PATHS.get(path, path) + separator + case
