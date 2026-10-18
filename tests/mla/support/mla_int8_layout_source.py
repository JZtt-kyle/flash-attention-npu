"""Pinned PA address oracle and existing CPU driver, including sanitizers.

The source trace uses element offsets, independent of element byte width.
Only the migrated descriptor validates INT8/BF16 storage sizes.
"""

import json
import os
from pathlib import Path
import subprocess

from tests.mla.paths import ROOT
from tests.mla.support.mla_dense_build import COMMIT, sha256


def source_class(source, token, templated):
    start = source.rfind("\n", 0, source.index(token)) + 1
    if templated:
        start = source.rfind("template <", 0, start)
    end, depth = source.index("{", start) + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    assert source[end] == ";"
    return source[start : end + 1] + "\n"


def build_drivers(directory):
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    selections = [
        ("parser.h", "class BlockTableParser", False),
        (
            "offset_calculator_v2.h",
            "struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD",
            True,
        ),
        ("attn_copy_gm_to_l1.h", "class CopyKvGmToL1", True),
        ("attn_copy_gm_to_l1.h", "class CopyKKropePAGmToL1", True),
    ]
    blobs = {}
    for name, _, _ in selections:
        path = "attention/common/op_kernel/memcopy/" + name
        blobs[name] = subprocess.check_output(
            ["git", "-C", str(root), "show", COMMIT + ":" + path], timeout=30
        )
        (directory / name).write_bytes(blobs[name])
    copyright = blobs["parser.h"].decode().split("*/", 1)[0] + "*/\n"
    (directory / "source_helpers.inc").write_text(
        copyright + "\n".join(source_class(blobs[n].decode(), t, p) for n, t, p in selections)
    )
    mla = ROOT / "csrc/ascend910/flash_attn_npu_3/mla"
    deps = [
        Path(__file__),
        mla / "mla_bsnd_layout.cpp",
        mla / "mla_bsnd_layout.h",
        mla / "mla_paged_layout.cpp",
        mla / "mla_paged_layout.h",
        Path("/usr/bin/g++").resolve(),
    ]
    binaries = []
    for name in ("layout", "source"):
        driver = ROOT / f"tests/mla/support/mla_paged_{name}_driver.cpp"
        deps.append(driver)
        binary = directory / name
        cmd = [
            "g++",
            "-std=c++17",
            "-O1",
            "-g",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-D_GLIBCXX_ASSERTIONS",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer",
            "-DMLA_INT8_LAYOUT",
            f"-I{mla}",
            f"-I{directory}",
        ]
        if name == "layout":
            cmd += [str(mla / "mla_bsnd_layout.cpp"), str(mla / "mla_paged_layout.cpp")]
        cmd += [str(driver), "-o", str(binary)]
        (directory / (name + "-command.json")).write_text(json.dumps(cmd, indent=2) + "\n")
        with (directory / (name + "-compile.log")).open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        binaries.append(binary)
    deps += [*binaries, *(directory / n for n in blobs), directory / "source_helpers.inc"]
    (directory / "manifest.json").write_text(
        json.dumps(dict(commit=COMMIT, files={str(p): sha256(p) for p in deps}), indent=2) + "\n"
    )
    return binaries


def run_rows(binary, rows):
    data = "".join(" ".join(map(str, row)) + "\n" for row in rows)
    import hashlib

    label = binary.name + "-" + hashlib.sha256(data.encode()).hexdigest()[:20]
    (binary.parent / (label + ".input")).write_text(data)
    result = subprocess.run([str(binary)], input=data, capture_output=True, text=True, timeout=60)
    (binary.parent / (label + ".output")).write_text(result.stdout)
    (binary.parent / (label + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    actual = [list(map(int, row.split(","))) for row in result.stdout.splitlines()]
    assert len(actual) == len(rows)
    return actual
