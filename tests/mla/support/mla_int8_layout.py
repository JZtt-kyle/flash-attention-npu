"""Existing PA descriptor with independently validated INT8 content storage."""

import ctypes as ct
import json
from pathlib import Path
import subprocess

from tests.mla.paths import ROOT, work_dir
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_validation_build import matching_files
from tests.mla.support.components.paged_metadata_consumer import CacheLayout, TensorView, TableView


def layout_library():
    root = work_dir() / "int8-layout"
    root.mkdir(parents=True, exist_ok=True)
    mla = ROOT / "csrc/ascend910/flash_attn_npu_3/mla"
    cpp = [
        ROOT / "tests/mla/support/mla_int8_layout_probe.cpp",
        mla / "mla_bsnd_layout.cpp",
        mla / "mla_paged_layout.cpp",
    ]
    files = [
        *cpp,
        mla / "mla_bsnd_layout.h",
        mla / "mla_paged_layout.h",
        Path(__file__).resolve(),
        Path("/usr/bin/g++").resolve(),
    ]
    identity = {str(p): sha256(p) for p in files}
    binary, manifest = root / "layout.so", root / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if (
        previous.get("inputs") != identity
        or not previous.get("output")
        or not matching_files(previous["output"])
    ):
        cmd = [
            "g++",
            "-std=c++17",
            "-O2",
            "-g",
            "-shared",
            "-fPIC",
            *map(str, cpp),
            f"-I{mla.parent}",
            "-o",
            str(binary),
        ]
        (root / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        with (root / "compile.log").open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        manifest.write_text(
            json.dumps(dict(inputs=identity, output={str(binary): sha256(binary)}), indent=2) + "\n"
        )
    lib = ct.CDLL(str(binary))
    lib.int8_cache_layout.argtypes = [
        ct.POINTER(TableView),
        ct.POINTER(TensorView),
        ct.POINTER(TensorView),
        ct.POINTER(CacheLayout),
    ]
    lib.int8_pages.argtypes = [
        ct.POINTER(CacheLayout),
        ct.c_void_p,
        ct.c_uint64,
        ct.c_void_p,
        ct.c_uint64,
    ]
    return lib


def tensor_view(tensor, cls=TensorView):
    return cls(
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        tensor.untyped_storage().nbytes(),
    )


def checked_layout(inputs, lengths):
    lib = layout_library()
    layout = CacheLayout()
    views = [
        tensor_view(inputs["table"], TableView),
        tensor_view(inputs["k"]),
        tensor_view(inputs["v"]),
    ]
    status = lib.int8_cache_layout(*(ct.byref(v) for v in views), ct.byref(layout))
    if status:
        raise ValueError(f"Invalid INT8 ND cache layout: {status}")
    table = inputs["table"]
    available = table.untyped_storage().nbytes() // 4 - table.storage_offset()
    values = (ct.c_int32 * len(lengths))(*lengths)
    status = lib.int8_pages(ct.byref(layout), table.data_ptr(), available, values, len(lengths))
    if status:
        raise ValueError(f"Invalid INT8 ND pages: {status}")
    return layout


def preserve_view(tensor, device="npu:0"):
    import torch

    storage = torch.as_strided(
        tensor, (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,), 0
    ).to(device)
    return torch.as_strided(storage, tensor.shape, tensor.stride(), tensor.storage_offset())
