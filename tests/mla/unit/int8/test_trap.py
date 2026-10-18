"""Device-side nonfinite scale and mutated page rejection, isolated contexts."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("field", ["qs", "ks", "vs", "page"])
def test_invalid_device_values_trap(field):
    path = Path(os.environ["MLA_VALIDATION_RESULT"]) / ("trap-" + field + ".log")
    with path.open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "tests.mla.support.mla_int8_trap", field],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    assert result.returncode == 0, path.read_text()[-6000:]
    assert '"rejected": "' + field + '"' in path.read_text()
