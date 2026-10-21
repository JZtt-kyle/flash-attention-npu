# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by Minghua Shen, 2026.

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Mirrors flash-attention/csrc/flash_attn/src/generate_kernels.py, adapted to the
# v3 NPU dispatch layout. The GPU script's generation axis is
# <dtype, head_dim, is_causal> — one explicit instantiation per .cu so nvcc
# compiles them in parallel. On the NPU side head_dim is NOT a template axis:
# the forward (FAInfer) does not bin by head_dim at all (it is a runtime tiling
# value), and the backward (FAGGeneral) selects head_dim at runtime via a
# switch() inside the impl (bwd_dispatch_common.hpp). So head_dim is deliberately
# dropped here.
#
# The real compile-time axes are dtype and input layout (TND / BSND). Each
# generated .cpp holds exactly ONE explicit instantiation of the per-family
# dispatch template, so the heavy kernel templates land in separate,
# parallel-compiled translation units. The impl bodies live in the shared
# fwd_dispatch_impl.hpp / bwd_dispatch_common.hpp headers (one directory up);
# this script only emits the instantiation stubs.

# dtype key -> C++ type token
DTYPE_MAP = {
    "fp16": "half",
    "bf16": "bfloat16_t",
}

# layout key -> (display name, fwd IS_TND bool token, bwd layout constant token)
LAYOUTS = [
    ("bsnd", "BSND", "false", "BSND"),
    ("tnd", "TND", "true", "TND"),
]

PRELUDE = (
    "/**\n"
    " * Copyright (c) 2026 Huawei Technologies Co., Ltd.\n"
    " * Modified by Minghua Shen, 2026.\n"
    " */\n"
)


def _header(family_desc: str, layout_display: str) -> str:
    return (
        PRELUDE + f"// v3 {family_desc}, {layout_display} variant. One explicit instantiation per\n"
        "// translation unit so the kernel templates compile in parallel across\n"
        "// cores; head_dim is a runtime axis (switch/tiling inside the impl), not a\n"
        "// template parameter, so it is not a generation axis.\n\n"
    )


def fwd_kernel(dtype_key: str, layout: tuple) -> "Kernel":
    ctype = DTYPE_MAP[dtype_key]
    layout_key, display, is_tnd, _ = layout
    body = (
        _header("forward FAInfer dispatch", display)
        + '#include "../fwd_dispatch_impl.hpp"\n\n'
        + f"template void launch_fwd_dtype<{ctype}, {is_tnd}>(const FwdLaunchArgs &);"
        f"  // {display}\n"
    )
    return Kernel(
        family="fwd",
        dtype=dtype_key,
        layout=layout_key,
        filename=f"fwd_dispatch_{dtype_key}_{layout_key}.cpp",
        content=body,
    )


def bwd_kernel(dtype_key: str, layout: tuple) -> "Kernel":
    ctype = DTYPE_MAP[dtype_key]
    layout_key, display, _, enum = layout
    body = (
        _header("backward FAGGeneral dispatch", display)
        + '#include "../bwd_dispatch_common.hpp"\n\n'
        + f"template void bwd_dispatch_run<{ctype}, {enum}>(const BwdLaunchArgs &);"
        f"  // {display}\n"
    )
    return Kernel(
        family="bwd",
        dtype=dtype_key,
        layout=layout_key,
        filename=f"bwd_dispatch_{dtype_key}_{layout_key}.cpp",
        content=body,
    )


@dataclass
class Kernel:
    family: str
    dtype: str
    layout: str
    filename: str
    content: str


def mla_kernel(dtype: str, layout: str, implementation: str, defines: dict) -> Kernel:
    """One dtype/layout instantiation; numerical code stays in the shared layers."""
    values = dict(dtype=dtype, title=dtype.title(), ctype=DTYPE_MAP[dtype])
    body = PRELUDE + (
        "// Generated MLA instantiation. CANN source copyrights and licenses\n"
        "// are retained in the shared implementation and mla/LICENSE.cann.\n"
    )
    for name, value in defines.items():
        body += f"#define {name} {value.format(**values)}\n"
    body += f'#include "../mla/{implementation}"\n'
    return Kernel("mla", dtype, layout, f"mla_dispatch_{dtype}_{layout}.cpp", body)


def get_all_kernels() -> List[Kernel]:
    kernels: List[Kernel] = []
    for dtype in DTYPE_MAP:
        for layout in LAYOUTS:
            kernels.append(fwd_kernel(dtype, layout))
            kernels.append(bwd_kernel(dtype, layout))

    # Preserve separate compiler instantiations, with one declaration of each
    # supported family instead of maintaining two copies of every wrapper.
    direct = [
        ("bsnd", "DENSE", "dense", "Dense"),
        ("tnd", "TND", "tnd", "Tnd"),
        ("tnd_bsnd", "MIXED", "tnd_bsnd", "TndBsnd"),
        ("bsnd_used", "USED", "bsnd_used", "BsndUsed"),
        ("tnd_used", "USED", "tnd_used", "TndUsed"),
        ("paged_bsnd", "PAGED", "paged_bsnd", "PagedBsnd"),
        ("paged_tnd", "PAGED", "paged_tnd", "PagedTnd"),
    ]
    split_layouts = [
        ("bsnd", "Bsnd"),
        ("tnd", "Tnd"),
        ("mixed", "Mixed"),
        ("paged_bsnd", "PagedBsnd"),
        ("paged_tnd", "PagedTnd"),
    ]
    for dtype in DTYPE_MAP:
        for layout, prefix, entry, launch in direct:
            defines = {
                f"MLA_{prefix}_TYPE": "{ctype}",
                f"MLA_{prefix}_ENTRY": f"mla_{entry}_{{dtype}}",
                f"MLA_{prefix}_LAUNCH": f"Launch{launch}{{title}}",
            }
            if prefix in ("USED", "PAGED"):
                defines[f"MLA_{prefix}_TND"] = str(int("tnd" in layout))
            kernels.append(
                mla_kernel(
                    dtype,
                    layout,
                    "mla_dispatch_impl.hpp",
                    defines,
                )
            )
        for layout, launch in split_layouts:
            defines = {
                "MLA_SPLIT_TYPE": "{ctype}",
                "MLA_SPLIT_PARTIAL": f"mla_split_{layout}_partial_{{dtype}}",
                "MLA_SPLIT_MERGE": f"mla_split_{layout}_merge_{{dtype}}",
                "MLA_SPLIT_LAUNCH": f"LaunchSplit{launch}{{title}}",
                f"MLA_SPLIT_{layout.upper()}": "1",
            }
            if layout == "paged_tnd":
                defines["MLA_SPLIT_INIT"] = "mla_split_paged_tnd_init_{dtype}"
            kernels.append(
                mla_kernel(dtype, f"split_{layout}", "mla_split_dispatch_impl.hpp", defines)
            )

            defines = {
                "MLA_AUTO_TYPE": "{ctype}",
                "MLA_AUTO_INIT": f"mla_auto_split_{layout}_init_{{dtype}}",
                "MLA_AUTO_PARTIAL": f"mla_auto_split_{layout}_partial_{{dtype}}",
                "MLA_AUTO_MERGE": f"mla_auto_split_{layout}_merge_{{dtype}}",
                "MLA_AUTO_LAUNCH": f"LaunchAutoSplit{launch}{{title}}",
                "MLA_AUTO_PAGED": str(int(layout.startswith("paged_"))),
                "MLA_AUTO_TND": str(int(layout in ("tnd", "mixed", "paged_tnd"))),
                "MLA_AUTO_MIXED": str(int(layout == "mixed")),
            }
            kernels.append(
                mla_kernel(
                    dtype, f"auto_split_{layout}", "mla_auto_split_dispatch_impl.hpp", defines
                )
            )
        kernels.append(
            mla_kernel(
                dtype,
                "prefill_bsnd",
                "mla_prefill_dispatch_impl.hpp",
                {
                    "MLA_PREFILL_TYPE": "{ctype}",
                    "MLA_PREFILL_INIT": "MlaPrefillTnd{title}Init",
                    "MLA_PREFILL_ATTENTION": "MlaPrefillBsnd{title}Attention",
                    "MLA_PREFILL_MERGE": "MlaPrefillBsnd{title}Merge",
                    "MLA_PREFILL_LAUNCH": "LaunchPrefillBsnd{title}",
                },
            )
        )
        kernels.append(
            mla_kernel(
                dtype,
                "gqa",
                "mla_gqa_dispatch_impl.hpp",
                {
                    "MLA_GQA_TYPE": "{ctype}",
                    "MLA_GQA_ATTENTION": "MlaGqa{title}Attention",
                    "MLA_GQA_LAUNCH": "LaunchGqa{title}",
                },
            )
        )
    return kernels


def write_kernel(kernel: Kernel, output_dir: Path) -> None:
    (output_dir / kernel.filename).write_text(kernel.content)


def main(output_dir: Optional[str]) -> None:
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)
    print(f"Generated {len(get_all_kernels())} dispatch TUs in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the v3 flash_attn_npu dispatch template instantiations "
        "(per dtype x layout; head_dim is a runtime axis and is not generated).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the dispatch .cpp stubs; "
        "defaults to this script's directory (csrc/flash_attn_npu_3/autogen).",
    )
    args = parser.parse_args()
    main(args.output_dir)
