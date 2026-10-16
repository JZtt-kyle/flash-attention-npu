"""192/128 probe allocation and calls; test selection/assertions stay in unit/."""

import ctypes as ct
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_prefill_build import build_prefill
from tests.mla.support.components.window_attention import WindowParams


class Layout(ct.Structure):
    _fields_ = [(k, ct.c_uint64) for k in ("batch", "row", "head", "span")]


class Params(ct.Structure):
    _fields_ = (
        [(k, ct.c_uint32) for k in ("batch", "sq", "sk", "heads", "kv_heads", "m", "n", "cores")]
        + [("scale", ct.c_float), ("window", WindowParams)]
        + [(k, Layout) for k in ("q", "k", "v")]
        + [("softcap", ct.c_float)]
    )


class Task(ct.Structure):
    _fields_ = [
        (k, ct.c_uint32)
        for k in ("batch", "head", "row", "rows", "q_length", "kv_length", "begin", "end")
    ]


def load(dtype, directory):
    manifest = build_prefill(dtype)
    directory = Path(directory) / dtype
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    libs = {k: ct.CDLL(v["path"]) for k, v in manifest["libraries"].items()}
    for key in ("migrated", "oracle"):
        libs[key].prefill_params_size.restype = ct.c_uint32
        if libs[key].prefill_params_size() != ct.sizeof(Params):
            raise RuntimeError("Prefill launch ABI does not match the loaded probe")
        libs[key].launch_prefill.argtypes = [ct.c_void_p] * 17 + [ct.c_int]
        libs[key].launch_prefill.restype = ct.c_int
    for key in ("merge_migrated", "merge_oracle"):
        libs[key].launch_merge.argtypes = [ct.c_void_p] * 9
        libs[key].launch_merge.restype = ct.c_int
        libs[key].launch_merge_softcap.argtypes = [ct.c_void_p] * 9 + [ct.c_float]
        libs[key].launch_merge_softcap.restype = ct.c_int
    (directory / "loaded.json").write_text(json.dumps(manifest["libraries"], indent=2) + "\n")
    from tests.mla.support.components.dense import Schedule

    host = libs["schedule"]
    host.prepare_dense.argtypes = [ct.c_int64] * 4 + [ct.c_void_p] * 2
    host.prepare_dense.restype = ct.c_int
    schedule, details = Schedule(), (ct.c_uint64 * 16)()
    if host.prepare_dense(1, 1, 8, 31, ct.byref(schedule), details) != 0:
        raise RuntimeError("Failed to query the actual platform core resources")
    libs["cores"] = int(details[2])
    (directory / "platform.json").write_text(
        json.dumps(dict(aic=details[2], aiv=details[3])) + "\n"
    )
    return libs, directory


def make_case(spec, dtype, max_cores):
    from tests.mla.support.mla_prefill_reference import fragment_rows
    from tests.mla.support.mla_ref import normalize_fa3_window
    from tests.mla.support.components.split_kv_merge import guarded

    batch, sq, heads, sk, kv_heads = spec["shape"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    generator = torch.Generator().manual_seed(20261015)
    inputs = {
        key: torch.randn(batch, seq, h, width, generator=generator).to(dtype)
        for key, seq, h, width in (
            ("q", sq, heads, 192),
            ("k", sk, kv_heads, 192),
            ("v", sk, kv_heads, 128),
        )
    }
    if spec.get("max_change"):
        inputs["q"].zero_()
        inputs["q"][..., 0] = 16
        inputs["k"][..., 0] = torch.linspace(-4, 4, sk)[None, :, None]
    if spec.get("broadcast"):
        for key in ("k", "v"):
            inputs[key] = inputs[key][:, :1, :1].expand_as(inputs[key])
    lengths = spec.get("lengths", [sk] * batch)
    for b, length in enumerate(lengths):
        for key in ("k", "v"):
            if length < sk:
                inputs[key][b, length:] = torch.nan
    guards, input_guards = [], []
    device = {}
    for key, tensor in inputs.items():
        if spec.get("strides"):
            b, s, h, d = tensor.shape
            strides = ((s + 2) * (h + 1) * (d + 16), (h + 1) * (d + 16), d + 16, 1)
            size = sum((z - 1) * v for z, v in zip(tensor.shape, strides)) + d
            owner = torch.full((size + 256,), 117, dtype=dtype)
            owner.as_strided(tensor.shape, strides, 128).copy_(tensor)
            storage = owner.npu()
            device[key] = storage.as_strided(tensor.shape, strides, 128)
            input_guards.append((owner, storage))
        elif spec.get("broadcast") and key in ("k", "v"):
            base = tensor[:, :1, :1].contiguous()
            storage = base.npu()
            device[key] = storage.expand_as(tensor)
            input_guards.append((base, storage))
        else:
            storage = tensor.contiguous().npu()
            device[key] = storage
            input_guards.append((tensor.contiguous(), storage))
    native = {k: v.contiguous().npu() for k, v in inputs.items()}
    if spec.get("split_rope"):
        native["qr"] = inputs["q"][..., 128:].contiguous().npu()
        native["kr"] = inputs["k"][..., 128:].contiguous().npu()
        native["q"] = inputs["q"][..., :128].contiguous().npu()
        native["k"] = inputs["k"][..., :128].contiguous().npu()
    for key in ("k", "v"):
        native[key + "_desc"] = torch.tensor(
            [24, 1 << 32, 0xFFFFFFFF, native[key].data_ptr()], dtype=torch.int64
        ).npu()
    left, right = normalize_fa3_window(
        sq, sk, causal=spec.get("causal", False), window_size=spec.get("window", (-1, -1))
    )
    kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
    window = WindowParams(kind, sk - 1 if left < 0 else left, sq - 1 if right < 0 else right)
    p = Params(
        batch,
        sq,
        sk,
        heads,
        kv_heads,
        spec["m"],
        spec["n"],
        1,
        spec.get("scale", 192**-0.5),
        window,
    )
    for key, value in device.items():
        span = sum((size - 1) * stride for size, stride in zip(value.shape, value.stride())) + 1
        setattr(p, key, Layout(*value.stride()[:3], span))
    p.softcap = spec.get("softcap", 0.0)
    tasks = fragment_rows(spec)
    capacity = min(max_cores, len(tasks))
    outputs = {
        name: [
            guarded(torch.full((batch, sq, heads, 128), 12345, dtype=dtype), guards),
            guarded(torch.full((batch, heads, sq), 12345.0), guards),
        ]
        for name in ("oracle", "migrated")
    }
    raw = {
        name: [
            guarded(torch.full((capacity, p.m, width), value), guards)
            for width, value in (
                (128, 0.0 if name == "oracle" else 12345.0),
                (8, 0.0 if name == "oracle" else 12345.0),
                (8, -torch.inf if name == "oracle" else 12345.0),
            )
        ]
        for name in outputs
    }
    ws = {
        name: guarded(
            torch.full(
                (capacity * (12 * p.m * ((p.n + 31) // 32 * 32) + 16 * p.m * 128) + 4096,),
                205,
                dtype=torch.uint8,
            ),
            guards,
        )
        for name in outputs
    }
    return dict(
        capacity=capacity,
        spec=spec,
        dtype=dtype,
        inputs=inputs,
        device=device,
        native=native,
        params=p,
        tasks=tasks,
        outputs=outputs,
        raw=raw,
        ws=ws,
        guards=guards,
        input_guards=input_guards,
        qlens=torch.full((batch,), sq, dtype=torch.int64).npu(),
        klens=torch.tensor(lengths, dtype=torch.int64).npu(),
        mask=torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu(),
    )


def wave(case, first):
    p = case["params"]
    rows = case["tasks"][first : first + case["capacity"]]
    p.cores = len(rows)
    tasks = (Task * 26)()
    for index, ((virtual, block, split, begin, end), row, mask) in enumerate(rows):
        b, h = divmod(virtual, p.kv_heads)
        tasks[index] = Task(b, h, row, mask.shape[0], p.sq, mask.shape[1], begin, end)
    return tasks


def launch_case(libs, case, name, tasks, partial=True):
    p = case["params"]
    tensors = case["native"] if name == "oracle" else case["device"]
    keys = ("q", "k_desc", "v_desc") if name == "oracle" else ("q", "k", "v")
    ptrs = [tensors[k].data_ptr() for k in keys]
    ptrs += [tensors[k].data_ptr() if k in tensors else None for k in ("qr", "kr")]
    ptrs += [
        t.data_ptr()
        for t in (
            *case["outputs"][name],
            case["ws"][name],
            *case["raw"][name],
            case["qlens"],
            case["klens"],
            case["mask"],
        )
    ]
    return libs[name].launch_prefill(
        torch.npu.current_stream().npu_stream, *ptrs, ct.byref(p), ct.byref(tasks), int(partial)
    )


def prepare_merge(case, raw):
    from tests.mla.support.components.split_kv_merge import guarded, make_params

    spec = case["spec"]
    batch, sq, heads, sk, kv_heads = spec["shape"]
    g = heads // kv_heads
    virtual = dict(
        spec,
        shape=(batch * kv_heads, sq, g, sk),
        lengths=[v for v in spec.get("lengths", [sk] * batch) for _ in range(kv_heads)],
    )
    storage = []
    s = spec["splits"]
    outputs = {
        name: [
            guarded(
                torch.full((batch * kv_heads, sq, g, 128), 12345, dtype=case["dtype"]), storage
            ),
            guarded(torch.full((batch * kv_heads, g, sq), 12345.0), storage),
        ]
        for name in ("oracle", "migrated")
    }
    accum = [
        guarded(torch.full((s, batch * kv_heads, g, sq, 128), 12345.0), storage),
        guarded(torch.full((s, batch * kv_heads, g, sq), 12345.0), storage),
    ]
    groups = []
    for group in range(len(case["tasks"]) // s):
        p = make_params(virtual, case["tasks"], group)
        lo, hi = group * s, (group + 1) * s
        valid = torch.zeros((s, p.m), dtype=torch.bool)
        valid[:, : p.rows] = torch.stack([mask.any(-1) for _, _, mask in case["tasks"][lo:hi]])
        canonical = [x[lo:hi].clone() for x in raw["oracle"]]
        canonical[0][~valid] = 0
        canonical[1][~valid] = 0
        canonical[2][~valid] = -torch.finfo(torch.float32).max
        canonical[1][0, ~valid.any(0)] = 1
        groups.append(
            dict(
                params=p,
                valid=valid,
                device={
                    name: [
                        guarded(t, storage)
                        for t in (canonical if name == "oracle" else [x[lo:hi] for x in raw[name]])
                    ]
                    for name in outputs
                },
            )
        )
    return dict(groups=groups, outputs=outputs, accum=accum, storage=storage, splits=s)


def launch_merge(libs, case, name, *, canonical=False, softcap=0.0):
    for group in case["groups"]:
        p = type(group["params"]).from_buffer_copy(group["params"])
        p.adapt = int(name == "migrated" and not canonical)
        library = libs["merge_" + name]
        function = library.launch_merge if softcap == 0 else library.launch_merge_softcap
        code = function(
            torch.npu.current_stream().npu_stream,
            *(
                t.data_ptr()
                for t in (
                    *group["device"]["oracle" if canonical else name],
                    *case["outputs"][name],
                    *case["accum"],
                )
            ),
            ct.byref(p),
            *(() if softcap == 0 else (softcap,)),
        )
        if code:
            return code
    return 0


def merge_case(libs, case, raw):
    merge = prepare_merge(case, raw)
    for _ in range(3):
        for name in merge["outputs"]:
            if name == "oracle" and merge["splits"] > 192:
                continue
            code = launch_merge(libs, merge, name)
            if code:
                raise RuntimeError(f"{name} merge failed: {code}")
        torch.npu.synchronize()
    return (
        {k: [t.cpu() for t in v] for k, v in merge["outputs"].items()},
        [t.cpu() for t in merge["accum"]],
        merge["storage"],
    )
