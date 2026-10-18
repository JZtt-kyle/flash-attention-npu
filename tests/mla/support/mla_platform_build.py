"""Compile the production platform query against CANN's real host platform database."""

import os
from pathlib import Path
import platform
import subprocess

from tests.mla.paths import ROOT
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_causal_build import toolchain_identity


def platform_inputs():
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    configs = cann / f"{platform.machine()}-linux/data/platform_config"
    if not configs.is_dir():
        raise FileNotFoundError(configs)
    return cann, configs


def platform_identity(sanitize=False):
    cann, configs = platform_inputs()
    files = list(configs.glob("Ascend910*.ini"))
    files += list((cann / "include/platform").glob("*.h"))
    files += list(
        (cann / f"{platform.machine()}-linux/asc/include/utils/tiling/platform").glob("*.h")
    )
    files += [
        cann / "lib64" / name
        for name in ("libplatform.so", "libtiling_api.a", "libunified_dlog.so")
    ]
    identity = toolchain_identity(sanitize)
    identity["platform_files"] = {str(p.resolve()): sha256(p) for p in sorted(files)}
    return identity


def compile_platform(directory, *, sanitize=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mla = ROOT / "csrc/ascend910/flash_attn_npu_3/mla"
    source = (mla / "mla_forward.cpp").read_text()
    # Compile the exact production function, not a second implementation of
    # the allowlist or resource checks. Only TORCH_CHECK's exception transport
    # and the runtime SoC name are replaced in this CPU-only executable.
    start = source.index("MlaPlatformInfo DensePlatform(")
    end = source.index("\nMlaPlatformInfo CausalPlatform(", start)
    (directory / "platform_query.h").write_text(source[start:end])
    cann, _ = platform_inputs()
    executable = directory / "platform"
    command = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O1",
        "-g",
        f"-I{directory}",
        f"-I{mla}",
        f"-I{cann}/include",
        f"-I{cann}/{platform.machine()}-linux/asc/include/utils",
        str(ROOT / "tests/mla/support/mla_platform_driver.cpp"),
        str(mla / "mla_schedule.cpp"),
        str(mla / "detail/cann_split_core.cpp"),
        f"-L{cann}/lib64",
        f"-Wl,-rpath,{cann}/lib64",
        "-ltiling_api",
        "-lplatform",
        "-lunified_dlog",
        "-o",
        str(executable),
    ]
    if sanitize:
        command += [
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
        ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return executable
