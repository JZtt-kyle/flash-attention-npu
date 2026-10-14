"""Independent BSND causal component, pinned source and dual CPU math.

The source mask/length arrays are constructed directly from raw case inputs;
no oracle reads the migrated schedule to determine which keys are visible.
"""

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import resource
from pathlib import Path
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.mla_dense_build import ROOT, build_probes, sha256
from tests.mla.support.mla_profile_cases import CASES, S1G_CASES, KV_CASES, LONG_KV_CASES
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_dense import Schedule, make_case as dense_case


# Frozen before sampling. Preserve the four existing capacity matrices as
# causal variants; add effective-length, partial/whole masked and zero cases.
PERFORMANCE_CASES = [
    dict(shape=shape, scale=scale) for shape, scale in CASES + S1G_CASES + KV_CASES + LONG_KV_CASES
] + [
    dict(shape=(2, 7, 16, 33), lengths=[3, 0]),
    dict(shape=(2, 513, 3, 257), lengths=[257, 129]),
    dict(shape=(2, 3, 128, 65), lengths=[0, 0]),
]
SMOKE_CASES = [
    dict(shape=(2, 7, 16, 33), lengths=[3, 0]),
    dict(shape=(1, 257, 1, 257), cores=1),
    dict(shape=(1, 1, 128, 16384)),
    dict(shape=(21, 1, 8, 16384)),
    dict(shape=(2, 513, 3, 257), lengths=[257, 129]),
    dict(shape=(10, 3, 86, 8193)),
]


def load_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["causal"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    for name, digest in manifest["dependencies"].items():
        assert sha256(Path(name)) == digest, name
    libraries = {}
    for name, record in manifest["libraries"].items():
        assert sha256(Path(record["path"])) == record["sha256"]
        libraries[name] = ct.CDLL(record["path"])
    for name in ("oracle", "migrated"):
        libraries[name].launch_causal.argtypes = (
            [ct.c_void_p] * 10 + [ct.c_float] + [ct.c_void_p] * 2
        )
        libraries[name].launch_causal.restype = ct.c_int
    libraries["schedule"].prepare_causal.argtypes = (
        [ct.c_int64] * 4 + [ct.c_void_p] + [ct.c_int64] * 3 + [ct.c_void_p] * 2
    )
    libraries["schedule"].prepare_causal.restype = ct.c_int
    libraries["dtype"] = torch.float16 if manifest["dtype"] == "fp16" else torch.bfloat16
    # The source sparse-mode 3 ABI requires this real, upper triangular mask.
    libraries["mask"] = torch.ones((2048, 2048), dtype=torch.bool).triu(1).npu()
    (directory / "loaded.json").write_text(json.dumps(manifest["libraries"], indent=2) + "\n")
    return libraries, directory


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def causal_attention(tmp_path_factory, request):
    base = Path(os.environ.get("MLA_CAUSAL_ATTENTION_DIR", tmp_path_factory.mktemp("mla-causal")))
    directory = base / request.param
    build_probes(directory, request.param, causal=True)
    return load_attention(directory)


def make_case(probes, spec):
    batch, sq, heads, sk = spec["shape"]
    lengths = spec.get("lengths", [sk] * batch)
    assert len(lengths) == batch
    raw = (ct.c_int64 * max(1, batch))(*lengths)
    schedule, details = Schedule(), (ct.c_uint64 * 16)()
    assert (
        probes[0]["schedule"].prepare_causal(
            batch,
            sq,
            heads,
            sk,
            raw,
            spec.get("cores", 0),
            spec.get("m", 0),
            spec.get("n", 0),
            ct.byref(schedule),
            details,
        )
        == 0
    )
    case = dense_case(
        probes,
        batch,
        sq,
        heads,
        sk,
        scale=spec.get("scale"),
        offset=spec.get("offset", 1),
        perturb=spec.get("perturb"),
        amplitude=spec.get("amplitude", 1.0),
        schedule_details=(schedule, details),
    )
    case["config"] = spec
    case["lengths"] = list(lengths)
    case["device_lengths"] = {
        "oracle": torch.tensor(lengths or [0], dtype=torch.int64).npu(),
        "migrated": torch.tensor(lengths or [0], dtype=torch.int32).npu(),
    }
    case["mask"] = probes[0]["mask"]
    if spec.get("max_change"):
        for name in ("q", "qv"):
            case["inputs"][name].zero_()
        case["inputs"]["qv"][..., 0] = 32 * spec.get("amplitude", 1.0)
        for b, length in enumerate(lengths):
            case["inputs"]["v"][b, :length, 0, 0] = torch.linspace(-24, 24, length)
    # Poison the unused physical capacity, independently of the schedule.
    for b, length in enumerate(lengths):
        case["inputs"]["k"][b, length:] = float("nan")
        case["inputs"]["v"][b, length:] = float("nan")
    for tensor, (before, device, prefix) in zip(case["inputs"].values(), case["storage"][:4]):
        before[prefix : prefix + tensor.numel()] = tensor.flatten()
        device.copy_(before)
    return case


def launch(probes, case, name):
    d = case["device"]
    pointers = [
        d["qv"],
        d["q"],
        d["v"],
        d["k"],
        case["desc"],
        *case["outputs"][name],
        case["workspace"][name],
    ]
    # Empty tensors have a null data_ptr; no reads are permitted. The valid
    # backing allocation lets the source perform its zero-length init path.
    fallback = case["storage"][0][1].data_ptr()
    return probes[0][name].launch_causal(
        torch.npu.current_stream().npu_stream,
        *(x.data_ptr() if x.numel() else fallback for x in pointers),
        ct.byref(case["schedule"]),
        case["scale"],
        case["device_lengths"][name].data_ptr(),
        case["mask"].data_ptr(),
    )


def references(case, workers=4):
    # Samples are independent. Keep the original math and intra-op thread
    # count; bound parallelism and memory without changing dot products.
    def sample(b):
        length = case["lengths"][b]
        inputs = {
            key: value[b : b + 1, :length] if key in ("k", "v") else value[b : b + 1]
            for key, value in case["inputs"].items()
        }
        refs = mla_reference_pair(**inputs, softmax_scale=case["scale"], causal=True)
        if length == 0 and case["shape"][3]:
            refs[1].fill_(-torch.inf)
            refs[3].fill_(-torch.inf)
        return refs

    batch, sq, heads, _ = case["shape"]
    result = [
        torch.empty((batch, sq, heads, 512), dtype=case["inputs"]["q"].dtype),
        torch.empty((batch, heads, sq), dtype=torch.float32),
    ]
    result += [torch.empty_like(t) for t in result]
    with ThreadPoolExecutor(max_workers=min(workers, max(1, batch))) as pool:
        for b, refs in enumerate(pool.map(sample, range(batch))):
            for i, ref in enumerate(refs):
                result[i][b : b + 1] = ref
    return result


def check_case(probes, case, repeats=3):
    started = time.monotonic()
    first = None
    for _ in range(repeats):
        for name in ("oracle", "migrated"):
            assert launch(probes, case, name) == 0
            try:
                torch.npu.synchronize()
            except RuntimeError as exc:
                raise RuntimeError(f"{name} causal launch failed for {case['config']}") from exc
        torch.npu.synchronize()
        results = {
            name: [t.cpu() for t in case["outputs"][name]] for name in ("oracle", "migrated")
        }
        if first is not None:
            for actual, previous in zip(results["migrated"], first):
                torch.testing.assert_close(actual, previous, rtol=0, atol=0)
        first = results["migrated"]
        batch, sq, heads, capacity = case["shape"]
        for b, length in enumerate(case["lengths"]):
            empty = max(0, sq - length)
            for name in results:
                out, lse = results[name]
                assert not out[b].isnan().any() and not lse[b].isnan().any()
                assert (out[b, :empty] == 0).all()
                expected = torch.inf if name == "oracle" or capacity == 0 else -torch.inf
                assert (lse[b, :, :empty] == expected).all()
            torch.testing.assert_close(first[0][b], results["oracle"][0][b], rtol=0, atol=0)
            torch.testing.assert_close(
                first[1][b, :, empty:], results["oracle"][1][b, :, empty:], rtol=0, atol=0
            )
    # Each independent math call sees just the raw effective sequence. Global
    # empty-capacity and a zero effective KV sample have different LSE signs.
    refs = references(case)
    source = [t.clone() for t in results["oracle"]]
    # Empty-row signs were checked on both raw results above. Normalize only
    # that documented FA3 adaptation for the source/math comparison.
    for b, length in enumerate(case["lengths"]):
        source[1][b, :, : max(0, sq - length)] = torch.inf if capacity == 0 else -torch.inf
    comparisons = {}
    for i, name in enumerate(("out", "lse")):
        comparisons[name] = assert_mla_source_close(
            first[i], source[i], refs[i], refs[i + 2], name=name
        )
    for i, (before, device, prefix) in enumerate(case["storage"]):
        after = device.cpu()
        if i < 4:
            torch.testing.assert_close(after, before, rtol=0, atol=0, equal_nan=True)
        else:
            assert torch.equal(after[:prefix], before[:prefix])
            assert torch.equal(after[-16:], before[-16:])
    for workspace in case["ws_storage"].values():
        assert (workspace[:512].cpu() == 205).all() and (workspace[-512:].cpu() == 205).all()
    identity = hashlib.sha256(json.dumps(case["config"], sort_keys=True).encode()).hexdigest()[:16]
    torch.save(
        dict(
            inputs=case["inputs"],
            lengths=case["lengths"],
            scale=case["scale"],
            outputs=results,
            references=refs,
        ),
        probes[1] / f"case-{identity}.pt",
    )
    s = case["schedule"]
    record = dict(
        config=case["config"],
        comparisons=comparisons,
        dtype=case["dtype"],
        seconds=time.monotonic() - started,
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        npu_peak_allocated_bytes=torch.npu.max_memory_allocated(),
        reference_workers=min(4, max(1, batch)),
        reference_intraop_threads=torch.get_num_threads(),
        repeats=repeats,
        tile=[s.m_base, s.s2_base],
        cores=s.cores,
        endpoints=list(zip(s.b_end[: s.cores], s.m_end[: s.cores], s.s_end[: s.cores])),
        evidence=f"case-{identity}.pt",
    )
    with (probes[1] / "accuracy.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


@pytest.mark.parametrize("spec", SMOKE_CASES)
def test_cost_and_mask_smoke(causal_attention, spec):
    check_case(causal_attention, make_case(causal_attention, spec))


def test_cost_and_mask_smoke_parallel_reference(causal_attention):
    case = make_case(causal_attention, dict(shape=(3, 7, 3, 33), lengths=[0, 3, 33]))
    for serial, parallel in zip(references(case, workers=1), references(case)):
        torch.testing.assert_close(serial, parallel, rtol=0, atol=0)


@pytest.mark.parametrize("spec", PERFORMANCE_CASES)
def test_performance_geometries(causal_attention, spec):
    check_case(causal_attention, make_case(causal_attention, spec))


@pytest.mark.parametrize("heads", (1, 3, 15, 17, 73, 128, 129, 171))
@pytest.mark.parametrize("sq,sk", ((7, 3), (7, 7), (7, 33), (129, 65)))
def test_head_and_empty_boundaries(causal_attention, heads, sq, sk):
    check_case(
        causal_attention, make_case(causal_attention, dict(shape=(2, sq, heads, sk), cores=1))
    )


@pytest.mark.parametrize(
    "sk", (1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1025)
)
@pytest.mark.parametrize("m", (128, 129, 255, 256))
def test_s2_and_m_tails(causal_attention, sk, m):
    check_case(
        causal_attention, make_case(causal_attention, dict(shape=(1, 171, 3, sk), m=m, cores=1))
    )


@pytest.mark.parametrize(
    "spec",
    [
        dict(shape=(0, 7, 8, 33)),
        dict(shape=(2, 0, 8, 33)),
        dict(shape=(2, 7, 8, 0)),
        dict(shape=(2, 7, 8, 33), lengths=[0, 0]),
        dict(shape=(41, 7, 73, 513), lengths=([0, 1, 3, 513] * 10 + [0])),
        dict(shape=(5, 171, 3, 513), lengths=[0, 17, 0, 513, 0], cores=1),
    ],
)
def test_empty_and_pipeline(causal_attention, spec):
    check_case(causal_attention, make_case(causal_attention, spec), repeats=5)


@pytest.mark.parametrize("scale", (None, 0.0, -0.23, 0.23))
@pytest.mark.parametrize("perturb", range(4))
def test_scale_and_halves(causal_attention, scale, perturb):
    spec = dict(shape=(2, 7, 17, 1025), lengths=[3, 1025], scale=scale, perturb=perturb)
    check_case(causal_attention, make_case(causal_attention, spec))


@pytest.mark.parametrize("amplitude", (0.125, 4.0, 8.0))
def test_amla_and_masked_max_changes(causal_attention, amplitude):
    spec = dict(shape=(2, 129, 3, 1025), lengths=[17, 1025], amplitude=amplitude, max_change=True)
    check_case(causal_attention, make_case(causal_attention, spec))


def test_interleaved_streams(causal_attention):
    cases = [make_case(causal_attention, s) for s in SMOKE_CASES[:2]]
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    torch.npu.synchronize()
    for stream, case in zip(streams, cases):
        with torch.npu.stream(stream):
            for name in ("oracle", "migrated"):
                assert launch(causal_attention, case, name) == 0
    torch.npu.synchronize()
    before = [[t.clone() for t in c["outputs"]["migrated"]] for c in cases]
    for case, expected in zip(cases, before):
        check_case(causal_attention, case)
        for result, old in zip(case["outputs"]["migrated"], expected):
            torch.testing.assert_close(result, old, rtol=0, atol=0)
