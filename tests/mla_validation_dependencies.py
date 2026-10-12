"""Conservative native dependencies plus the selected tests' Python imports.

All native sources and test drivers remain in the closure. Unrelated test
modules and the headless launcher no longer invalidate a numerical suite.
"""

import ast
from pathlib import Path


def suite_sources(root, selection, baseline, *, headless=False):
    root = Path(root)
    selected = {
        path
        for path in baseline
        if not (
            path.startswith("tests/")
            and Path(path).name.startswith("test_")
            and path.endswith(".py")
        )
        and path != "scripts/mla_codex_loop.py"
    }
    if headless and "scripts/mla_codex_loop.py" in baseline:
        selected.add("scripts/mla_codex_loop.py")
    pending = [arg.split("::", 1)[0] for arg in selection if arg.split("::", 1)[0].endswith(".py")]
    pending += ["tests/mla_validation.py"]
    pending += [path for path in selected if Path(path).name in {"__init__.py", "conftest.py"}]
    visited = set()

    def module_paths(name):
        parts = name.split(".")
        paths = ["/".join(parts) + ".py"]
        paths += ["/".join(parts[:i]) + "/__init__.py" for i in range(1, len(parts) + 1)]
        return [path for path in paths if path in baseline]

    while pending:
        path = pending.pop()
        if path in visited or path not in baseline:
            continue
        visited.add(path)
        selected.add(path)
        tree = ast.parse((root / path).read_text(), filename=path)
        package = path.removesuffix(".py").split("/")[:-1]
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                name = ".".join([*prefix, *([node.module] if node.module else [])])
                names = [name, *(name + "." + alias.name for alias in node.names)]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Subprocess '-m tests.module' and literal import_module calls.
                if node.value.startswith("tests.") and all(
                    part.isidentifier() for part in node.value.split(".")
                ):
                    names = [node.value]
            elif isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name == "exec" and path == "tests/mla_metadata_build.py":
                    # This reviewed builder executes an AST subset extracted
                    # from setup.py; both files are always in the native closure.
                    compiled = node.args[0] if node.args else None
                    if (
                        isinstance(compiled, ast.Call)
                        and isinstance(compiled.func, ast.Name)
                        and compiled.func.id == "compile"
                        and len(compiled.args) == 3
                        and isinstance(compiled.args[1], ast.Constant)
                        and compiled.args[1].value == "setup.py"
                        and "setup.py" in baseline
                    ):
                        continue
                if name in {
                    "exec",
                    "eval",
                    "run_path",
                    "spec_from_file_location",
                    "SourceFileLoader",
                }:
                    return dict(baseline)
                if name in {"__import__", "import_module", "run_module"} and (
                    not node.args or not isinstance(node.args[0], ast.Constant)
                ):
                    # Unknown dynamic imports require the original full closure.
                    return dict(baseline)
            for name in names:
                pending.extend(module_paths(name))
    return {path: baseline[path] for path in sorted(selected)}
