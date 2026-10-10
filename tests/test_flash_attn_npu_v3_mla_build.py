"""CPU checks for the scoped ACL launch bridge and generated build inputs."""

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import shutil

import pytest

from tests.mla_metadata_build import MLA, metadata_builder


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
    assert len(generated) == 18
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


def test_aicpu_sources_by_backend():
    builder = metadata_builder()
    builder._build_aicpu_object = lambda _output, source: str(Path(source).relative_to(ROOT))
    for name, directory in (
        ("flash_attn_npu.flash_attn_npu", "ascend910/flash_attn_npu"),
        ("flash_attn_npu_3.flash_attn_npu_3", "ascend910/flash_attn_npu_3"),
        ("flash_attn_npu_3_950", "ascend950/flash_attn_npu_3"),
        ("flash_attn_npu_4.flash_attn_npu_4", "ascend910/flash_attn_npu_4"),
    ):
        expected = [f"csrc/{directory}/fa_metadata.aicpu"]
        if name == "flash_attn_npu_3.flash_attn_npu_3":
            expected.append(f"csrc/{directory}/mla/mla_metadata.aicpu")
            expected.append(f"csrc/{directory}/mla/mla_bsnd_metadata.aicpu")
            expected.append(f"csrc/{directory}/mla/mla_tnd_metadata.aicpu")
            expected.append(f"csrc/{directory}/mla/mla_tnd_bsnd_metadata.aicpu")
            expected.append(f"csrc/{directory}/mla/mla_used_metadata.aicpu")
        assert builder._build_aicpu_metadata("unused.so", name) == expected
    assert builder._build_aicpu_metadata("unused.so", "unknown") == []


@pytest.mark.parametrize(
    "stem,header_name",
    [
        ("mla_metadata", "mla_schedule.h"),
        ("mla_bsnd_metadata", "mla_bsnd_schedule.h"),
        ("mla_tnd_metadata", "mla_tnd_schedule.h"),
        ("mla_tnd_metadata", "mla_tnd_bsnd_schedule.h"),
        ("mla_tnd_metadata", "mla_used_schedule.h"),
        ("mla_tnd_metadata", "mla_tnd_metadata_args.h"),
        ("mla_tnd_bsnd_metadata", "mla_tnd_bsnd_schedule.h"),
        ("mla_tnd_bsnd_metadata", "mla_used_schedule.h"),
        ("mla_tnd_bsnd_metadata", "mla_tnd_bsnd_metadata.h"),
        ("mla_tnd_bsnd_metadata", "mla_tnd_bsnd_metadata_args.h"),
        ("mla_used_metadata", "mla_used_schedule.h"),
        ("mla_used_metadata", "mla_used_metadata.h"),
        ("mla_used_metadata", "mla_used_metadata_args.h"),
        ("mla_used_metadata", "mla_tnd_schedule.cpp"),
    ],
)
def test_aicpu_incremental_dependencies(tmp_path, monkeypatch, capsys, stem, header_name):
    monkeypatch.delenv("FLASH_ATTN_FORCE_REBUILD", raising=False)
    source = tmp_path / "mla"
    shutil.copytree(MLA, source)
    src, out = str(source / (stem + ".aicpu")), str(tmp_path / "test.so")
    builder = metadata_builder()
    obj = Path(builder._build_aicpu_object(out, src))
    dependencies = Path(str(obj) + ".d").read_text()
    for name in (
        "mla_metadata.h",
        stem + "_args.h"
        if stem in ("mla_tnd_metadata", "mla_tnd_bsnd_metadata", "mla_used_metadata")
        else "mla_metadata_args.h",
        "mla_schedule.cpp",
        "cann_split_core.cpp",
    ):
        assert name in dependencies
    if stem == "mla_bsnd_metadata":
        for name in (
            "mla_bsnd_metadata.h",
            "mla_bsnd_metadata.cpp",
            "mla_bsnd_schedule.h",
            "mla_bsnd_schedule.cpp",
        ):
            assert name in dependencies
    if stem == "mla_tnd_metadata":
        for name in (
            "mla_tnd_metadata.h",
            "mla_tnd_metadata.cpp",
            "mla_tnd_schedule.h",
            "mla_tnd_bsnd_schedule.h",
            "mla_used_schedule.h",
            "mla_tnd_schedule.cpp",
        ):
            assert name in dependencies
    if stem == "mla_tnd_bsnd_metadata":
        for name in (
            "mla_tnd_bsnd_metadata.h",
            "mla_tnd_bsnd_metadata.cpp",
            "mla_tnd_bsnd_schedule.h",
            "mla_used_schedule.h",
            "mla_tnd_schedule.cpp",
        ):
            assert name in dependencies
    if stem == "mla_used_metadata":
        for name in (
            "mla_used_metadata.h",
            "mla_used_metadata.cpp",
            "mla_used_schedule.h",
            "mla_tnd_schedule.cpp",
        ):
            assert name in dependencies
    capsys.readouterr()
    assert builder._build_aicpu_object(out, src) == str(obj)
    assert "[compile-aicpu-skip]" in capsys.readouterr().out
    # Change a transitive header while keeping the .aicpu source untouched.
    header = source / header_name
    header.write_text(header.read_text() + "// dependency invalidation probe\n")
    before = obj.stat().st_mtime_ns
    builder._build_aicpu_object(out, src)
    assert "[compile-aicpu-skip]" not in capsys.readouterr().out
    assert obj.stat().st_mtime_ns > before
    # Missing dependency evidence and changed options must also force rebuilding.
    for evidence in (Path(str(obj) + ".d"), Path(str(obj) + ".cmdhash")):
        evidence.unlink()
        builder._build_aicpu_object(out, src)
        assert "[compile-aicpu-skip]" not in capsys.readouterr().out


def test_metadata_entry_bounds_and_serialization(tmp_path):
    # The barrier itself is tested on 910B; this shim allows sanitizers to check
    # the real entry's pointer/size guards and serializer without an NPU runtime.
    (tmp_path / "aicpu_api.h").write_text(
        "#define __global__\n#define __aicpu__\n"
        "namespace AscendC { inline void DataStoreBarrier() {} }\n"
    )
    executable = tmp_path / "metadata-test"
    command = [
        "g++",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-g",
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
        "-fno-omit-frame-pointer",
        f"-I{tmp_path}",
        f"-I{MLA}",
        "-x",
        "c++",
        str(MLA / "mla_metadata.aicpu"),
        str(ROOT / "tests/mla_metadata_driver.cpp"),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True)
    subprocess.run([str(executable)], check=True)
