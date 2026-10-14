"""Content-checked, source-only CPU oracles for the independent split-KV protocol."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from tests.mla.support.mla_causal_build import source_blobs, toolchain_identity
from tests.mla.support.mla_window_build import FA3_COMMIT, fa3_blob
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_schedule import (
    CANN_COMMIT,
    FIA,
    MLA,
    ROOT,
    source_function,
)

BLOCK_SHA256 = "4c9b132fdbff869716ca673f087cf2ede20f041fa8923c2f181394b716981083"
WORKSPACE_FUNCTIONS = (
    "CalcMmResSize",
    "FillTilingWorkspaceParams",
    "CalcFlashDecodeParamNums",
    "CalcNormalWorkspaceSize",
    "CalcFlashDecodeWorkspace",
    "CalcWorkspaceSize",
)


def blobs():
    root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    result = source_blobs()
    for path in (FIA + "fia_tiling_nonquant_mla.cpp",):
        result[path] = subprocess.check_output(
            ["git", "-C", str(root), "show", CANN_COMMIT + ":" + path], timeout=30
        )
    result["flash_api.cpp"] = fa3_blob()
    directory = Path(os.environ.get("MLA_FA3_SOURCE_ROOT", f"/tmp/mla-fa3-reference/{FA3_COMMIT}"))
    block = (directory / "block.h").read_bytes()
    if hashlib.sha256(block).hexdigest() != BLOCK_SHA256:
        raise RuntimeError("FA3 split oracle must match the pinned block.h SHA256")
    result["block.h"] = block
    return result


def split_toolchain_identity(sanitize):
    return dict(
        **toolchain_identity(sanitize),
        fa3_commit=FA3_COMMIT,
        split_source_blobs={p: hashlib.sha256(b).hexdigest() for p, b in blobs().items()},
    )


def fa3_functions(api, block):
    request = next(
        line for line in api.splitlines() if "params.num_splits = num_splits <= 0" in line
    )
    check = next(line for line in api.splitlines() if '"num_splits > 256 not supported"' in line)
    lower = next(line for line in api.splitlines() if "TORCH_CHECK(params.num_splits >= 1)" in line)
    alloc = api[api.index("    at::Tensor out_accum, softmax_lse_accum;") :]
    alloc = alloc[: alloc.index("        params.is_fp32 = false;")]
    # Use the unmodified split-only fragment with an independently enumerated
    # mask interval; FA3's CUDA tile choice/overfetch is not the CANN M geometry.
    start = block.index("            int num_n_blocks_per_split =")
    end = block.index("            // if (threadIdx.x", start)
    return (
        "int Request(int64_t num_splits) { Params params{};\n"
        + request
        + "\n"
        + lower
        + "\n"
        + check
        + "\nreturn params.num_splits; }\n"
        + "std::vector<Tensor> Accumulators(bool is_varlen_q, int64_t batch_size, int64_t seqlen_q, "
        "int64_t total_q, int64_t num_heads, int splits) {\nParams params{}; params.num_splits=splits;\n"
        "int head_size_v=512; Options opts;\n"
        + alloc
        + "}\nreturn {out_accum,softmax_lse_accum};\n}\n"
        + "namespace cute { int ceil_div(int a,int b) { return (a+b-1)/b; } }\n"
        "std::pair<int,int> ExplicitRange(int n_block_min,int n_block_max,"
        "int num_splits_actual,int split_idx_actual) {\n"
        + block[start:end]
        + "return {n_block_min,n_block_max};\n}\n"
    )


def compile_drivers(directory):
    directory.mkdir(parents=True, exist_ok=True)
    sanitize = os.environ.get("MLA_SCHEDULE_SANITIZE") == "1"
    identity = split_toolchain_identity(sanitize)
    (directory / "toolchain.json").write_text(json.dumps(identity, indent=2) + "\n")
    sources = blobs()
    for name, blob in sources.items():
        (directory / Path(name).name).write_bytes(blob)
    for name in ("LICENSE.cann", "LICENSE.fa3"):
        (directory / name).write_bytes((MLA / name).read_bytes())
    host = sources[FIA + "fia_tiling_nonquant_mla.cpp"].decode()
    functions = "template <typename T>\n" + source_function(host, "inline auto Align(")
    functions += "\n".join(
        source_function(host, "FiaTilingNonQuantMla::" + name + "(") for name in WORKSPACE_FUNCTIONS
    )
    (directory / "workspace_functions.inc").write_text(functions)
    (directory / "fa3_split_functions.inc").write_text(
        fa3_functions(sources["flash_api.cpp"].decode(), sources["block.h"].decode())
    )
    (directory / "log").mkdir(exist_ok=True)
    (directory / "log/log.h").write_text(
        "#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n"
    )
    binaries = []
    for kind in ("adapter", "native", "contracts"):
        binary = directory / kind
        command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-g", "-D_GLIBCXX_ASSERTIONS"]
        if sanitize:
            command += [
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
                "-fno-sanitize-recover=all",
            ]
        if kind == "adapter":
            command += ["-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(MLA)]
            command += [
                str(MLA / name)
                for name in (
                    "mla_split_kv.cpp",
                    "mla_window.cpp",
                    "mla_causal.cpp",
                    "mla_schedule.cpp",
                    "mla_tnd_schedule.cpp",
                    "detail/cann_split_core.cpp",
                )
            ]
            driver = "mla_split_kv_driver.cpp"
        else:
            command += ["-I" + str(directory)]
            if kind == "native":
                command += [str(directory / "split_core.cpp")]
                driver = "mla_window_source_driver.cpp"
            else:
                driver = "mla_split_kv_source_driver.cpp"
        command += [str(ROOT / "tests/mla/support" / driver), "-o", str(binary)]
        (directory / f"{kind}-compile.json").write_text(json.dumps(command, indent=2) + "\n")
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (directory / f"{kind}-compile.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        binaries.append(binary)
    (directory / "binaries.json").write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries}, indent=2)
        + "\n"
    )
    return binaries
