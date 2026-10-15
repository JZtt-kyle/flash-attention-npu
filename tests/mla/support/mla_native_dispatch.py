"""Validate actual fixed-package CANN launches, separately from timing."""

import hashlib
import json
from pathlib import Path
import re

from tests.mla.support.mla_split_stage_cases import check_split_stage_dispatch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def e2e_applicable(template, fd, *, representation="compressed", include_native_fd=False):
    if representation not in ("compressed", "prefill"):
        raise ValueError("Unknown MLA representation")
    return template in ((3, 4) if representation == "prefill" else (5,)) and (
        include_native_fd or not fd
    )


def parse_dispatch(log, dtype, count, vendor, *, representation="compressed"):
    vendor = Path(vendor).resolve()
    root = vendor / "op_impl/ai_core/tbe/kernel"
    configs = list(root.glob("config/ascend910b/**/fused_infer_attention_score.json"))
    if len(configs) != 1:
        raise RuntimeError("Expected one fixed CANN kernel configuration")
    dtype_name = {"fp16": "float16", "bf16": "bfloat16"}[dtype]
    bins = [
        b
        for b in json.loads(configs[0].read_text())["binList"]
        if b["inputs"][0]["dtype"] == dtype_name and b["outputs"][0]["dtype"] == dtype_name
    ]
    if len(bins) != 1:
        raise RuntimeError("Expected one CANN binary for the requested dtype")
    metadata = (root / bins[0]["binInfo"]["jsonFilePath"]).resolve()
    info = json.loads(metadata.read_text())
    binary = (metadata.parent / (info["binFileName"] + info["binFileSuffix"])).resolve()
    if not binary.is_relative_to(vendor) or not metadata.is_relative_to(vendor):
        raise RuntimeError("CANN dispatch binary escapes the requested package")
    if sha256(binary) != info["sha256"]:
        raise RuntimeError("CANN dispatch binary changed")
    keys = {int(row["tilingKey"]) for row in info["kernelList"]}
    current, cases, registrations = None, [], []
    with Path(log).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.startswith("MLA_STRIDE_NATIVE_BEGIN "):
                _, kind, index = line.split()
                if current is not None or kind != dtype or int(index) != len(cases):
                    raise RuntimeError("Out-of-order native dispatch boundary")
                current = dict(case=int(index), begin_line=line_number, launches=[])
            if "Register Kernel succ" in line and "bin path: " in line:
                path = Path(line.split("bin path: ", 1)[1].strip()).resolve()
                if path == binary:
                    registrations.append(
                        dict(line=line_number, path=str(path), sha256=sha256(path))
                    )
            if (
                current is not None
                and "LaunchKernel: kernel info" in line
                and "kernel_name=FusedInferAttentionScore_" in line
            ):
                match = re.search(r"kernel_name=(FusedInferAttentionScore_[0-9a-f]+)_(\d+),", line)
                if match is None:
                    raise RuntimeError("Unrecognized native launch record")
                current["launches"].append(
                    dict(line=line_number, binary=match[1] + ".o", key=int(match[2]))
                )
            if line.startswith("MLA_STRIDE_NATIVE_END "):
                _, kind, index = line.split()
                if current is None or kind != dtype or int(index) != current["case"]:
                    raise RuntimeError("Unmatched native dispatch boundary")
                launches = current["launches"]
                if (
                    len(launches) != 1
                    or launches[0]["binary"] != binary.name
                    or launches[0]["key"] not in keys
                ):
                    raise RuntimeError("Missing, ambiguous or foreign native FIA launch")
                key = launches[0]["key"]
                # Fixed-source GenTilingKey encodes this digit as PA * 2 + FD.
                template, fd = key // 10**15 % 100, bool((key // 10**5 % 10) & 1)
                current.update(
                    end_line=line_number,
                    native_tiling_key=key,
                    native_template=template,
                    native_fd=fd,
                    e2e_applicable=e2e_applicable(template, fd, representation=representation),
                )
                cases.append(current)
                current = None
    if current is not None or len(cases) != count or not registrations:
        raise RuntimeError("Incomplete native dispatch or missing package registration")
    check_split_stage_dispatch(cases)
    return dict(
        dtype=dtype,
        representation=representation,
        cases=cases,
        registrations=registrations,
        artifacts={str(p.resolve()): sha256(p) for p in (Path(log), configs[0], metadata, binary)},
    )
