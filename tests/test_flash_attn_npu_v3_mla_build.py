"""CPU checks for the scoped ACL launch bridge and generated build inputs."""

import ast
import hashlib
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_scoped_batch_launch(tmp_path):
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    executable = tmp_path / "launch-test"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-pthread",
            "-fsanitize=address,undefined",
            "-g",
            f"-I{cann}/include",
            f"-I{ROOT}/csrc/ascend910/flash_attn_npu_3",
            str(ROOT / "tests/mla_launch_driver.cpp"),
            str(ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_launch.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


def test_generated_mla_launch(tmp_path):
    autogen = ROOT / "csrc/ascend910/flash_attn_npu_3/autogen"
    subprocess.run(
        ["python3", str(autogen / "generate_kernels.py"), "-o", str(tmp_path)], check=True
    )
    generated = list(tmp_path.glob("*.cpp"))
    assert len(generated) == 10
    for path in generated:
        assert path.read_bytes() == (autogen / path.name).read_bytes()


def test_device_dependency_staleness(tmp_path):
    # Execute only the pure freshness helpers, avoiding setup's build side effects.
    tree = ast.parse((ROOT / "setup.py").read_text())
    helpers = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in ("_cmdhash", "_obj_is_fresh")
        ],
        type_ignores=[],
    )
    scope = {"os": os, "hashlib": hashlib}
    exec(compile(helpers, "setup.py", "exec"), scope)
    src, obj, header, deps, command = [
        tmp_path / n for n in ("launch.cpp", "launch.o", "kernel.hpp", "device.d", "cmdhash")
    ]
    for path in (src, obj, header):
        path.touch()
        os.utime(path, (1000, 1000))
    deps.write_text(f"{obj}: {src} \\\n {header}\n")
    command.write_text(scope["_cmdhash"](["bisheng", "-O2"]))

    def fresh():
        return scope["_obj_is_fresh"](
            str(obj), str(src), str(deps), str(command), ["bisheng", "-O2"]
        )

    assert fresh()
    os.utime(header, (1001, 1001))
    assert not fresh()
