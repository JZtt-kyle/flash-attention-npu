"""Module ownership: unit tests are opt-in, completed behaviors accumulate in E2E."""

from tests.mla.paths import ROOT
from tests.mla.regression_cases import daily_suites


def module_suites(layer, *, performance=False):
    result = {}
    for directory in sorted((ROOT / "tests/mla" / layer).iterdir()):
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        paths = [
            str(p.relative_to(ROOT))
            for p in sorted(directory.glob("test_*.py"))
            if layer == "unit" or p.name.endswith("_profile.py") == performance
        ]
        if paths:
            result[directory.name] = paths
    return result


UNITS = module_suites("unit")
DIAGNOSTIC_E2E = module_suites("e2e")
E2E = daily_suites()
PERFORMANCE = module_suites("e2e", performance=True)


def combined(modules, catalog):
    """One pytest process, with no duplicate selections."""
    return list(dict.fromkeys(path for module in modules for path in catalog[module]))


def is_e2e(selection):
    paths = [arg.split("::", 1)[0] for arg in selection if ".py" in arg]
    return bool(paths) and all("/unit/" not in p for p in paths)
