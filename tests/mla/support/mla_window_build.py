"""CPU window probes with independent, content-checked FA3 and CANN sources."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from tests.mla.support.mla_causal_build import source_blobs, toolchain_identity
from tests.mla.support.components.schedule import MLA, ROOT

FA3_COMMIT = "d15f1531a460ba456f41b01a774f33ab2db8febf"
FA3_SHA256 = "012547af36b3a780b2849b2ac26bda75d4aa26b0c83921449b81109128782fab"


def fa3_blob():
    directory = Path(os.environ.get("MLA_FA3_SOURCE_ROOT", f"/tmp/mla-fa3-reference/{FA3_COMMIT}"))
    blob = (directory / "flash_api.cpp").read_bytes()
    if hashlib.sha256(blob).hexdigest() != FA3_SHA256:
        raise RuntimeError("FA3 window oracle must match the pinned flash_api.cpp SHA256")
    return blob


def window_toolchain_identity(sanitize):
    return dict(
        **toolchain_identity(sanitize),
        fa3_commit=FA3_COMMIT,
        fa3_source_sha256=hashlib.sha256(fa3_blob()).hexdigest(),
    )


def normalization_oracle(blob):
    """Compile the exact forward and metadata snippets, without torch/CUDA."""
    text = blob.decode()
    metadata = text[text.index("    if (window_size_left >= max_seqlen_k - 1)") :]
    metadata = metadata[: metadata.index("    params.attention_chunk = attention_chunk;")]
    forward = text[text.index("    if (window_size_left >= seqlen_k - 1)") :]
    forward = forward[: forward.index("    if (!is_varlen_q)")]
    params = text[text.index("    // Causal is the special case") :]
    params = params[: params.index("    params.attention_chunk = attention_chunk;")]
    functions = [
        "Params SetParams(int seqlen_q, int seqlen_k, int window_size_left, "
        "int window_size_right, int attention_chunk) {\nParams params{};\n"
        + params
        + "return params;\n}\n"
    ]
    for name, fragment in (("Metadata", metadata), ("Forward", forward)):
        functions.append(
            "Params " + name + "(int64_t q, int64_t k, int64_t window_size_left, "
            "int64_t window_size_right, bool is_causal) {\n"
            "Params params{};\n"
            "int64_t max_seqlen_q=q, max_seqlen_k=k, seqlen_q=q, seqlen_k=k;\n"
            "int headdim=64, head_size=64; int64_t attention_chunk=0;\n"
            "bool paged_KV=true; std::optional<int> page_size=16;\n"
            + fragment
            + (
                "return params;\n}\n"
                if name == "Metadata"
                else "return SetParams(seqlen_q, seqlen_k, window_size_left, window_size_right, attention_chunk);\n}\n"
            )
        )
    return (
        "// Verbatim snippets from pinned FA3 flash_api.cpp; see copied source and LICENSE.fa3.\n"
        "#include <algorithm>\n#include <cstdint>\n#include <iostream>\n#include <optional>\n"
        "struct Params { bool is_causal, is_local; int64_t window_size_left, window_size_right; };\n"
        + "\n".join(functions)
        + """
int main() {
    int64_t q,k,left,right; bool causal;
    while (std::cin >> q >> k >> left >> right >> causal) {
        auto a=Forward(q,k,left,right,causal), b=Metadata(q,k,left,right,causal);
        std::cout << '[' << (a.is_causal ? 1 : a.is_local ? 2 : 0) << ','
                  << a.window_size_left << ',' << a.window_size_right << ','
                  << (b.is_causal ? 1 : b.is_local ? 2 : 0) << ','
                  << b.window_size_left << ',' << b.window_size_right << "]\\n" << std::flush;
    }
}
"""
    )


def compile_drivers(directory):
    directory.mkdir(parents=True, exist_ok=True)
    sanitize = os.environ.get("MLA_SCHEDULE_SANITIZE") == "1"
    (directory / "toolchain.json").write_text(
        json.dumps(window_toolchain_identity(sanitize), indent=2) + "\n"
    )
    for name, blob in source_blobs().items():
        (directory / Path(name).name).write_bytes(blob)
    blob = fa3_blob()
    (directory / "flash_api.cpp").write_bytes(blob)
    (directory / "LICENSE.fa3").write_bytes((MLA / "LICENSE.fa3").read_bytes())
    (directory / "fa3_normalization.cpp").write_text(normalization_oracle(blob))
    (directory / "log").mkdir(exist_ok=True)
    (directory / "log/log.h").write_text(
        "#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n"
    )
    binaries = []
    for kind in ("adapter", "source", "migrated-source", "fa3-normalize"):
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
                str(MLA / p)
                for p in (
                    "mla_window.cpp",
                    "mla_causal.cpp",
                    "mla_schedule.cpp",
                    "mla_tnd_schedule.cpp",
                    "detail/cann_split_core.cpp",
                )
            ]
            command += [str(ROOT / "tests/mla/support/mla_window_schedule_driver.cpp")]
        elif kind == "fa3-normalize":
            command += [str(directory / "fa3_normalization.cpp")]
        else:
            if kind == "source":
                command += ["-I" + str(directory), str(directory / "split_core.cpp")]
            else:
                command += [
                    "-DMIGRATED_SPLIT",
                    "-I" + str(MLA),
                    str(MLA / "detail/cann_split_core.cpp"),
                ]
            command += [str(ROOT / "tests/mla/support/mla_window_source_driver.cpp")]
        command += ["-o", str(binary)]
        (directory / f"{kind}-compile.json").write_text(json.dumps(command, indent=2) + "\n")
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (directory / f"{kind}-compile.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        binaries.append(binary)
    return binaries
