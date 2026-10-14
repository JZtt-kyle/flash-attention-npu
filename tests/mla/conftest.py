"""Compact module diagnostics before pytest creates test items."""

import json
import os
from pathlib import Path

import pytest

from tests.mla.matrix import compact_calls, full_matrix


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_generate_tests(metafunc):
    yield
    path = Path(metafunc.definition.path)
    # Tool fault-injection cases and performance input lists are already explicit.
    if "/unit/tools/" in str(path) or path.name.endswith("_profile.py"):
        return
    if full_matrix() or metafunc.definition.get_closest_marker("mla_full_combinations"):
        return
    calls = metafunc._calls
    chosen = compact_calls(calls, public="/e2e/" in str(path))
    # Preserve CallSpec2 objects and scope/indirect fixture semantics. Pruning here
    # avoids constructing the exhaustive collection of pytest Function items.
    metafunc._calls = chosen
    output = os.environ.get("MLA_VALIDATION_RESULT")
    if output and calls:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "parameter-coverage.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        function=metafunc.definition.nodeid,
                        expanded=len(calls),
                        retained=len(chosen),
                        selected=[call.id for call in chosen],
                        coverage="protocol pairs and shape boundaries/corners per dtype; representative pages and public adapters",
                    )
                )
                + "\n"
            )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "mla_full_combinations: preserve a documented multi-parameter regression matrix"
    )
