"""Build isolated dense probes from current headers and pinned CANN blobs."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import tarfile


from tests.mla.paths import ROOT

COMMIT = "ca18aee610729ca5270a44e2d84c751bc687fbf0"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_probes(
    artifacts,
    dtype="fp16",
    *,
    metadata=False,
    bsnd=False,
    tnd=False,
    mixed=False,
    used=False,
    stride=False,
    tnd_stride=False,
    paged=False,
    causal=False,
    window=False,
    split_kv=False,
    paged_causal=False,
    paged_window=False,
    prefill=False,
):
    assert dtype in ("fp16", "bf16")
    assert (
        sum(
            (
                metadata,
                bsnd,
                tnd,
                mixed,
                used,
                stride,
                tnd_stride,
                paged,
                causal,
                window,
                split_kv,
                prefill,
            )
        )
        <= 1
    )
    assert not paged_causal or paged
    assert not paged_window or (paged and not paged_causal)
    artifacts = Path(artifacts).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    source = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    paths = [
        "attention/common/op_kernel",
        "attention/fused_infer_attention_score/op_kernel",
        "attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp",
        "attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h",
    ]
    archive = subprocess.check_output(["git", "-C", str(source), "archive", COMMIT, *paths])
    oracle = artifacts / "oracle-source"
    oracle.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(oracle, filter="data")
    manifest = {
        "commit": COMMIT,
        "files": {
            str(p.relative_to(oracle)): sha256(p) for p in sorted(oracle.rglob("*")) if p.is_file()
        },
    }
    (artifacts / "oracle-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # The source build generates raw device structs from these host macros.
    # Recreate only the declarations (not getters/registration/gert) for the
    # standalone oracle; all device algorithm headers remain unmodified.
    host = (oracle / paths[-1]).read_text()
    structs = []
    for name, body in re.findall(
        r"BEGIN_TILING_DATA_DEF\((\w+)\)(.*?)END_TILING_DATA_DEF", host, re.S
    ):
        fields = []
        for macro, args in re.findall(r"(TILING_DATA_FIELD_DEF(?:_ARR|_STRUCT)?)\(([^)]*)\)", body):
            args = [x.strip() for x in args.split(",")]
            if macro.endswith("_ARR"):
                field_dtype, count, field = args
                fields.append(f"    {field_dtype} {field}[{count}];")
            else:
                field_dtype, field = args
                fields.append(f"    {field_dtype} {field};")
        # Generated CANN tiling substructures have 8-byte alignment. The source
        # copy_data_align64 endpoint loads require this, including after bools.
        structs.append(f"struct alignas(8) {name} {{\n" + "\n".join(fields) + "\n};")
        if name == "FusedInferAttentionScoreTilingData":
            break
    assert structs and structs[-1].startswith(
        "struct alignas(8) FusedInferAttentionScoreTilingData "
    )
    (artifacts / "oracle_tiling.h").write_text(
        "// Test-only raw declarations from the pinned tiling macro definitions.\n"
        "#include <cstdint>\nconstexpr uint32_t FIA_MAX_AIC_CORE_NUM = 26;\n"
        + "\n".join(structs)
        + "\n"
    )
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    common = [
        str(cann / "bin/bisheng"),
        "-x",
        "asc",
        "--npu-arch=dav-2201",
        "--cce-auto-infer-kernel-type=false",
        "-O2",
        "-g",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-DCATLASS_ARCH=2201",
        "-DASC_DEVKIT_MAJOR=9",
        f"-I{cann}/include",
        f"-I{cann}/include/experiment/runtime",
        f"-I{cann}/pkg_inc/runtime",
        f"-I{cann}/pkg_inc",
        f"-I{cann}/pkg_inc/profiling",
        f"-I{cann}/include/experiment/msprof",
        f"-L{cann}/lib64",
        "-lascendcl",
        "-lruntime",
        "-Wl,--wrap=aclrtLaunchKernelWithHostArgs",
    ]
    if paged_causal:
        common += ["-DMLA_PAGED_CAUSAL"]
    if dtype == "bf16":
        common.append("-DMLA_DENSE_BF16")
    if bsnd:
        common += ["-DMLA_BSND_ATTENTION", "-DNDEBUG"]
    if tnd or mixed or used or stride or tnd_stride or paged:
        common += ["-DNDEBUG"]
    layout_objects = []
    for layout in (
        ["bsnd", "tnd", "paged"]
        if paged
        else ["bsnd", "tnd"]
        if tnd_stride
        else ["bsnd"]
        if stride
        else []
    ):
        layout_object = artifacts / f"{layout}_layout.o"
        layout_command = [
            "g++",
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-c",
            str(ROOT / f"csrc/ascend910/flash_attn_npu_3/mla/mla_{layout}_layout.cpp"),
            "-o",
            str(layout_object),
        ]
        (artifacts / f"{layout}-layout-command.json").write_text(
            json.dumps(layout_command, indent=2) + "\n"
        )
        with (artifacts / f"{layout}-layout-compile.log").open("w") as log:
            subprocess.run(
                layout_command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60
            )
        layout_objects.append(str(layout_object))
    probe_source = (
        ROOT
        / "tests/mla/support"
        / (
            "mla_prefill_probe.cpp"
            if prefill
            else "mla_split_kv_attention_probe.cpp"
            if split_kv
            else "mla_window_attention_probe.cpp"
            if window
            else "mla_causal_attention_probe.cpp"
            if causal
            else "mla_window_paged_attention_probe.cpp"
            if paged_window
            else "mla_paged_attention_probe.cpp"
            if paged
            else "mla_tnd_stride_probe.cpp"
            if tnd_stride
            else "mla_bsnd_stride_probe.cpp"
            if stride
            else "mla_used_attention_probe.cpp"
            if used
            else "mla_tnd_bsnd_attention_probe.cpp"
            if mixed
            else "mla_tnd_attention_probe.cpp"
            if tnd
            else "mla_dense_probe.cpp"
        )
    )

    def compile_probe(name):
        output = artifacts / f"{name}.so"
        cmd = [*common, str(probe_source)]
        if name == "oracle":
            base = oracle / "attention/fused_infer_attention_score/op_kernel"
            cmd += [
                "-DMLA_DENSE_ORACLE",
                f"-I{base}/arch22",
                f"-I{base}",
                f"-I{artifacts}",
                f"-I{ROOT}/csrc/ascend910/flash_attn_npu_3",
            ]
        else:
            cmd += [f"-I{ROOT}/csrc/catlass/include", f"-I{ROOT}/csrc/ascend910/flash_attn_npu_3"]
            if name in ("metadata", "metadata_checked"):
                cmd.append("-DMLA_DENSE_METADATA")
            if name == "metadata_checked":
                cmd += ["-DMLA_DENSE_METADATA_CHECKED", "-DNDEBUG"]
        cmd += ["-MMD", "-MF", str(artifacts / f"{name}.d"), "-o", str(output)]
        if layout_objects:
            cmd += ["-x", "none", *layout_objects]
        (artifacts / f"{name}-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        with (artifacts / f"{name}-compile.log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        if result.returncode:
            raise RuntimeError(f"{name} build failed; see {artifacts / f'{name}-compile.log'}")
        # The final host pass overwrites -MMD with host-only dependencies. The
        # device guards require a separate device preprocessing pass. Every
        # invocation rebuilds both probes; no pre-existing .so is accepted.
        deps = [
            arg
            for arg in cmd[: cmd.index("-MMD")]
            if arg != "-shared" and not arg.startswith(("-l", "-L", "-Wl,"))
        ]
        depfile = artifacts / f"{name}-device.d"
        deps += ["--offload-device-only", "-M", "-MF", str(depfile)]
        (artifacts / f"{name}-deps-command.json").write_text(json.dumps(deps, indent=2) + "\n")
        with (artifacts / f"{name}-deps.log").open("w") as log:
            subprocess.run(deps, stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
        dependencies = depfile.read_text()
        expected = (
            [
                "fia_kernel_nonquant_mla.h",
                "fia_block_vec_nonquant_mla.h",
                "fia_block_cube_nonquant_mla.h",
            ]
            if name == "oracle"
            else [
                "mla_dense.hpp",
                "mla_vector.hpp",
                "mla_vector_ops.hpp",
                "mla_cube.hpp",
                "mla_copy.hpp",
                "mla_mmad.hpp",
            ]
        )
        if prefill:
            expected = (
                [
                    "fia_kernel_nonquant.h",
                    "fia_block_cube_nonquant_gqa.h",
                    "fia_block_vec_nonquant.h",
                ]
                if name == "oracle"
                else [
                    "mla_prefill.hpp",
                    "mla_prefill_cube.hpp",
                    "mla_prefill_vector.hpp",
                    "mla_copy.hpp",
                ]
            )
        assert all(header in dependencies for header in expected)
        if name in ("metadata", "metadata_checked"):
            assert "mla_metadata_consumer.h" in dependencies and "mla_metadata.hpp" in dependencies
        if tnd and name == "migrated":
            assert "mla_tnd.hpp" in dependencies and "mla_tnd_metadata.hpp" in dependencies
        if mixed and name == "migrated":
            assert (
                "mla_tnd_bsnd.hpp" in dependencies and "mla_tnd_bsnd_metadata.hpp" in dependencies
            )
        if used and name == "migrated":
            assert "mla_used.hpp" in dependencies and "mla_used_metadata.hpp" in dependencies
            assert "mla_used_metadata_consumer.h" in dependencies
        if stride and name == "migrated":
            assert all(
                header in dependencies
                for header in (
                    "mla_bsnd_layout.h",
                    "mla_used.hpp",
                    "mla_bsnd_metadata.hpp",
                    "mla_used_metadata.hpp",
                )
            )
        if bsnd and name == "migrated":
            assert "mla_bsnd_metadata.hpp" in dependencies
        if paged and name == "migrated":
            assert all(
                header in dependencies
                for header in (
                    "mla_paged.hpp",
                    "mla_paged_metadata.hpp",
                    "mla_paged_metadata_consumer.h",
                    "mla_paged_copy.hpp",
                    "mla_paged_layout.h",
                )
            )
        if tnd_stride and name == "migrated":
            assert all(
                header in dependencies
                for header in (
                    "mla_tnd_layout.h",
                    "mla_tnd.hpp",
                    "mla_tnd_bsnd.hpp",
                    "mla_used.hpp",
                    "mla_tnd_metadata.hpp",
                    "mla_tnd_bsnd_metadata.hpp",
                    "mla_used_metadata.hpp",
                )
            )
        assert ("oracle-source" in dependencies) == (name == "oracle")
        return str(output)

    names = (
        ("oracle", "migrated", "metadata", "metadata_checked")
        if metadata
        else ("oracle", "migrated")
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        libraries = dict(zip(names, pool.map(compile_probe, names)))
    host_output = artifacts / "schedule.so"
    host_cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        str(ROOT / "tests/mla/support/mla_dense_schedule.cpp"),
        str(ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_schedule.cpp"),
        str(ROOT / "csrc/ascend910/flash_attn_npu_3/mla/detail/cann_split_core.cpp"),
        f"-I{ROOT}/csrc/ascend910/flash_attn_npu_3",
        f"-I{cann}/include",
        f"-L{cann}/lib64",
        "-ltiling_api",
        "-lplatform",
        "-o",
        str(host_output),
    ]
    if causal or window or split_kv:
        host_cmd += [
            str(
                ROOT
                / (
                    "tests/mla/support/mla_window_attention_schedule.cpp"
                    if window or split_kv
                    else "tests/mla/support/mla_causal_attention_schedule.cpp"
                )
            ),
            str(ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_causal.cpp"),
            str(ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_schedule.cpp"),
        ]
    if window or split_kv:
        host_cmd += [str(ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_window.cpp")]
    if split_kv:
        host_cmd += [
            str(ROOT / "tests/mla/support/mla_split_kv_attention_schedule.cpp"),
            str(ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_split_kv.cpp"),
        ]
    (artifacts / "schedule-command.json").write_text(json.dumps(host_cmd, indent=2) + "\n")
    with (artifacts / "schedule-compile.log").open("w") as log:
        subprocess.run(host_cmd, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
    libraries["schedule"] = str(host_output)
    current = [
        ROOT / "tests/mla/support/mla_dense_build.py",
        ROOT / "tests/mla/support/mla_dense_schedule.cpp",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_dense_params.h",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_schedule.h",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_schedule.cpp",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/detail/cann_split_core.h",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/detail/cann_split_core.cpp",
        probe_source,
        ROOT / "tests/mla/support/mla_probe_batch_launch.h",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_layout.h",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_layout.cpp",
        *(
            [ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_layout.cpp"]
            if tnd_stride or paged
            else []
        ),
        *([ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_paged_layout.cpp"] if paged else []),
        *sorted((ROOT / "csrc/ascend910/flash_attn_npu_3/mla").glob("mla_paged*.h")),
        *sorted((ROOT / "csrc/ascend910/flash_attn_npu_3/mla").glob("mla_tnd*.h")),
        *sorted((ROOT / "csrc/ascend910/flash_attn_npu_3/mla").glob("mla_used*.h")),
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_metadata.h",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_metadata_consumer.h",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_metadata.h",
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_metadata_consumer.h",
        *sorted((ROOT / "csrc/ascend910/flash_attn_npu_3/mla").glob("*/*.hpp")),
    ]
    if causal or window or split_kv:
        current += [
            ROOT
            / (
                "tests/mla/support/mla_window_attention_schedule.cpp"
                if window or split_kv
                else "tests/mla/support/mla_causal_attention_schedule.cpp"
            ),
            ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_causal.h",
            ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_causal.cpp",
            ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_causal_metadata.h",
            ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_schedule.cpp",
        ]
    if window or split_kv:
        current += [
            ROOT / "csrc/ascend910/flash_attn_npu_3/mla" / p
            for p in ("mla_window.h", "mla_window.cpp", "mla_window_params.h")
        ]
    if split_kv:
        current += [
            ROOT / "tests/mla/support/mla_split_kv_attention_schedule.cpp",
            *[
                ROOT / "csrc/ascend910/flash_attn_npu_3/mla" / p
                for p in ("mla_split_kv.cpp", "mla_split_kv.h", "mla_split_kv_params.h")
            ],
        ]
    manifest = {
        "prefill": prefill,
        "split_kv": split_kv,
        "causal": causal,
        "window": window,
        "dtype": dtype,
        "bsnd": bsnd,
        "tnd": tnd,
        "mixed": mixed,
        "used": used,
        "stride": stride,
        "tnd_stride": tnd_stride,
        "paged": paged,
        "paged_causal": paged_causal,
        "paged_window": paged_window,
        "ndebug": bsnd or tnd or mixed or used or stride or tnd_stride or paged,
        "oracle_tiling_sha256": sha256(artifacts / "oracle_tiling.h"),
        "libraries": {
            name: {"path": path, "sha256": sha256(Path(path))} for name, path in libraries.items()
        },
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in current},
        "compiler": subprocess.check_output([str(cann / "bin/bisheng"), "--version"], text=True),
        "cann": str(cann),
        "catlass": subprocess.check_output(
            ["git", "-C", str(ROOT / "csrc/catlass"), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    if paged or causal or window or split_kv or prefill:
        dependencies = set()
        for depfile in artifacts.glob("*-device.d"):
            dependencies.update(
                Path(x).resolve() for x in depfile.read_text().replace("\\\n", " ").split()[1:]
            )
        assert all(p.is_file() for p in dependencies)
        manifest["dependencies"] = {str(p): sha256(p) for p in sorted(dependencies)}
    (artifacts / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return libraries


if __name__ == "__main__":
    print(json.dumps(build_probes(os.environ["MLA_DENSE_ARTIFACT_DIR"]), indent=2))
