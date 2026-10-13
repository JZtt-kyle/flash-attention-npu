"""BSND PA causal: v10, independent page/mask math and public/internal dispatch."""

from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla_dense_build import build_probes
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.mla_source_compare import assert_mla_source_close
from tests.mla_validation_policy import public_cases
from tests import test_flash_attn_npu_v3_mla_paged_bsnd_production as production
from tests import test_flash_attn_npu_v3_mla_paged_public as public_base
from tests.test_flash_attn_npu_v3_mla_paged_attention import paged_probe as paged_probe
from tests.test_flash_attn_npu_v3_mla_bsnd_stride_production import storage_fault as storage_fault

component = production.component
REPRESENTATIVES = [
    component.config(112, q=(7, 7), lengths=(3, 225), uq=None, heads=8),
    component.config(528, q=(257, 257), lengths=(257, 513), uq=None, heads=1),
    component.config(16, q=(7, 7), lengths=(3, 33), uq=(5, 1), repeat=True),
]
PERFORMANCE_CASES = public_cases(
    [
        *[dict(c, uq=None) for c in production.PERFORMANCE_CASES],
        *REPRESENTATIVES[:2],
        component.config(112, "independent", q=(7, 7), lengths=(3, 7), uq=None),
    ],
    group="paged-bsnd",
)
ACCURACY_CASES = [s for s in PERFORMANCE_CASES if s not in REPRESENTATIVES] + [
    component.config(112, q=(513, 513), lengths=(257, 129), uq=(513, 257), heads=3),
    component.config(48, q=(7, 7), lengths=(3, 129), uq=(0, 0)),
]


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def causal_paged_attention(tmp_path_factory, request):
    root = Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path_factory.mktemp("causal-paged")))
    directory = root / "causal-paged-probes" / request.param
    build_probes(directory, request.param, paged=True, paged_causal=True)
    assert json.loads((directory / "build-manifest.json").read_text())["paged_causal"]
    return component.load_attention(directory)


@pytest.fixture(params=(torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def dtype(request):
    return request.param


def make_case(probes, producer, spec, **kwargs):
    return production.make_case(probes, producer, dict(spec, causal=True), **kwargs)


def forward_args(case):
    return dict(production.forward_args(case), causal=True)


def call(args, entry="public", *, lse=True):
    args = dict(args, causal=True)
    if entry == "public":
        assert args.pop("seqused_q", None) is None
        return public_base.public(args, lse)
    return public_base.invoke(entry, args)


def metadata(args):
    q, k, table = (args[n] for n in ("q", "k", "page_table"))
    lengths = args.get("cache_seqlens")
    if lengths is None or isinstance(lengths, int):
        lengths = torch.full(
            (q.shape[0],),
            table.shape[1] * k.shape[1] if lengths is None else lengths,
            dtype=torch.int32,
            device=q.device,
        )
    return api.get_scheduler_metadata(
        q.shape[0],
        q.shape[1],
        table.shape[1] * k.shape[1],
        q.shape[2],
        1,
        64,
        lengths,
        qkv_dtype=q.dtype,
        headdim_v=512,
        page_size=k.shape[1],
        causal=True,
        num_splits=1,
    )


def masked_rows(case):
    used = case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"]
    return torch.tensor(
        [i + k < q for q, k in zip(used, case["config"]["lengths"]) for i in range(q)],
        dtype=torch.bool,
    )


def compare(probes, case, results):
    actual = [component.valid_outputs(case, [t.cpu() for t in result[:2]]) for result in results]
    refs = component.references(case)
    source = None
    reports = []
    if production.source_available(case):
        assert component.launch(probes, case, "oracle") == 0
        source = component.valid_outputs(
            case, [t.cpu() for t in case["outputs"]["oracle"]], source=True
        )
        masked = masked_rows(case)
        assert (source[0][masked] == 0).all() and source[1][:, masked].isposinf().all()
        source[1][:, masked] = -torch.inf
    for result in actual:
        if source is not None:
            reports.append(
                {
                    name: assert_mla_source_close(
                        result[i], source[i], refs[i], refs[i + 2], name=name
                    )
                    for i, name in enumerate(("out", "lse"))
                }
            )
        else:
            reports.append(dict(empty_math=component.check_math(result, refs)))
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    target = probes[1] / "causal-paged-accuracy"
    target.mkdir(exist_ok=True)
    identity = hashlib.sha256(
        json.dumps([case["config"], case["scale"]], sort_keys=True).encode()
    ).hexdigest()[:20]
    torch.save(
        dict(
            inputs=case["host"],
            pages=case["table_host"],
            actual=actual,
            source=source,
            references=refs,
            spec=case["config"],
            scale=case["scale"],
            reports=reports,
        ),
        target / f"{identity}.pt",
    )
    with (target / "results.jsonl").open("a") as f:
        f.write(json.dumps(dict(identity=identity, spec=case["config"], reports=reports)) + "\n")
    return reports


def check_spec(probes, producer, spec, **kwargs):
    case = make_case(probes, producer, spec, **kwargs)
    args = forward_args(case)
    entries = ("op", "eager", "cpp") if spec["uq"] is not None else ("public", "op", "eager", "cpp")
    with forbid_host_readback():
        results = [call(args, e) for e in entries]
        if spec["uq"] is None:
            record = metadata(args)
            results += [call(dict(args, scheduler_metadata=record), e) for e in entries]
    if spec["uq"] is None:
        assert [x & 0xFFFFFFFF for x in record.cpu().tolist()] == case["expected"]
    compare(probes, case, results)
    return case


@pytest.mark.parametrize("spec", REPRESENTATIVES)
def test_representative(causal_paged_attention, paged_probe, spec):
    check_spec(causal_paged_attention, paged_probe, spec)


def test_all_legal_pages(causal_paged_attention, paged_probe):
    for page in range(16, 1025, 16):
        check_spec(
            causal_paged_attention,
            paged_probe,
            component.config(page, q=(7, 7), lengths=(3, page + 1), uq=None, heads=3, repeat=True),
        )


@pytest.mark.parametrize("spec", ACCURACY_CASES)
def test_source_math(causal_paged_attention, paged_probe, spec):
    check_spec(causal_paged_attention, paged_probe, spec)


@pytest.mark.parametrize(
    "profile", ["bnsd", "broadcast", "overlap", "row_65536", "combined_65537", "table_overlap"]
)
def test_strides_and_scale(causal_paged_attention, paged_probe, profile):
    check_spec(
        causal_paged_attention,
        paged_probe,
        component.config(16, profile, q=(7, 7), lengths=(33, 7), uq=None, heads=5),
        scale=-0.125,
    )


@pytest.mark.parametrize(
    "q,k,uq,capacity",
    [
        ([], [], None, 0),
        ([0, 0], [17, 0], None, 224),
        ([7, 7], [0, 0], None, 0),
        ([7, 7], [0, 0], [5, 0], 224),
    ],
)
def test_empty(causal_paged_attention, paged_probe, q, k, uq, capacity):
    check_spec(
        causal_paged_attention,
        paged_probe,
        component.config(
            112, q=q, lengths=k, uq=uq, capacity=capacity, pool=0 if capacity == 0 else 7
        ),
    )


def check_constant(result, lengths, value=3, used=None):
    out, lse = result[:2]
    used = [out.shape[1]] * out.shape[0] if used is None else used
    for b, (q, k) in enumerate(zip(used, lengths)):
        count = (torch.arange(q) + k - q + 1).clamp(0, k)
        expected = torch.where(count > 0, value, 0).to(out.dtype)
        torch.testing.assert_close(
            out[b, :q].cpu(), expected[:, None, None].expand(q, out.shape[2], 512)
        )
        torch.testing.assert_close(
            lse[b, :, :q].cpu(), count.float().log()[None].expand(out.shape[2], q)
        )


@pytest.mark.parametrize("fake", [False, True])
def test_lengths_returns_offset_reuse(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs(dtype, sq=7)
        args["v"].fill_(3)
        for length in (None, 0, 3, 448, "tensor"):
            expected = [3, 113] if length == "tensor" else [448 if length is None else length] * 2
            args["cache_seqlens"] = torch.tensor(expected, dtype=torch.int32, device="npu")
            record = metadata(args)
            storage = torch.empty(record.numel() + 2, dtype=torch.int32, device="npu")
            storage[1:-1].copy_(record)
            args["cache_seqlens"] = args["cache_seqlens"] if length == "tensor" else length
            for reused in (None, storage[1:-1]):
                result = call(dict(args, scheduler_metadata=reused))
                assert len(result) == 4 and result[2] is not result[3]
                assert all(
                    t.numel() == 0 and t.dtype == torch.get_default_dtype() for t in result[2:]
                )
                out = call(dict(args, scheduler_metadata=reused), lse=False)
                assert out.shape == result[0].shape
                if not fake:
                    check_constant(result, expected)
                    torch.testing.assert_close(out, result[0], rtol=0, atol=0)


def test_compile_updates_and_used(dtype):
    torch._dynamo.reset()
    args = production.small_inputs(dtype, sq=7)
    args["v"].fill_(3)
    for mode in ("implicit", "generated", "reuse"):

        def invoke(kw):
            if mode == "generated":
                kw = dict(kw, scheduler_metadata=metadata(kw))
            return call(kw)

        compiled = torch.compile(invoke, backend="eager", fullgraph=True)
        for lengths in ([3, 113], [0, 0], [224, 7]):
            args["cache_seqlens"].copy_(torch.tensor(lengths, dtype=torch.int32))
            if mode == "reuse":
                args["scheduler_metadata"] = metadata(args)
            check_constant(compiled(args), lengths)
    args.pop("scheduler_metadata", None)
    args["seqused_q"] = torch.tensor([5, 1], dtype=torch.int32, device="npu")
    compiled = torch.compile(lambda kw: call(kw, "op"), backend="eager", fullgraph=True)
    check_constant(compiled(args), [224, 7], used=[5, 1])


@pytest.mark.parametrize("reuse", [False, True])
def test_stream_lifetime(dtype, reuse):
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    for step in range(3):
        with torch.npu.stream(creator):
            args = production.small_inputs(dtype, sq=7)
            args["v"].fill_(step + 1)
            args["cache_seqlens"].copy_(torch.tensor([3, 0], dtype=torch.int32))
            if reuse:
                args["scheduler_metadata"] = metadata(args)
                args["cache_seqlens"] = args["cache_seqlens"].clone()
        consumer.wait_stream(creator)
        with torch.npu.stream(consumer), forbid_host_readback():
            results.append(call(args))
        del args
        gc.collect()
        with torch.npu.stream(creator):
            pressure = [torch.empty(8192, dtype=dtype, device="npu") for _ in range(16)]
        del pressure
    consumer.synchronize()
    for step, result in enumerate(results):
        check_constant(result, [3, 0], step + 1)
    creator.synchronize()


def test_record_allows_new_pages_stride_and_scale(causal_paged_attention, paged_probe):
    spec = component.config(112, q=(7, 7), lengths=(3, 225), uq=None)
    original = make_case(causal_paged_attention, paged_probe, spec)
    record = metadata(forward_args(original))
    before = record.cpu().clone()
    for profile, repeat, scale in (("pa02_both", True, 0.0), ("bnsd", False, -0.125)):
        case = make_case(
            causal_paged_attention,
            paged_probe,
            dict(spec, profile=profile, repeat=repeat, pool=11),
            scale=scale,
        )
        result = call(dict(forward_args(case), scheduler_metadata=record))
        compare(causal_paged_attention, case, [result])
    torch.testing.assert_close(before, record.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("fake", [False, True])
def test_static_and_unopened_boundaries(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs(dtype)
        for change in (dict(num_splits=0), dict(window_size=(1, 0)), dict(softcap=1.0)):
            with pytest.raises(RuntimeError, match="not implemented"):
                call(dict(args, **change))
        record = metadata(args)
        for entry in ("op", "eager") if fake else ("op", "eager", "cpp"):
            with pytest.raises(RuntimeError, match="Q-used"):
                call(
                    dict(
                        args,
                        seqused_q=torch.ones(2, dtype=torch.int32, device="npu"),
                        scheduler_metadata=record,
                    ),
                    entry,
                )


@pytest.mark.parametrize("fake", [False, True])
def test_storage_bounds(fake, storage_fault):
    with FakeTensorMode() if fake else nullcontext():
        args = production.small_inputs()
        for name in ("q", "qv", "k", "v"):
            for entry in ("public", "op", "eager") if fake else ("public", "op", "eager", "cpp"):
                with (
                    storage_fault(args[name]),
                    pytest.raises(RuntimeError, match="(input view exceeds|storage.*bounds)"),
                ):
                    call(args, entry)


TRAPS = (
    "v6",
    "v10-unmasked",
    "header",
    "endpoint",
    "stale-kv",
    "negative-kv",
    "overflow-kv",
    "negative-page",
    "overflow-page",
    "zero-query-page",
    "zero-used-page",
    "negative-used",
    "overflow-used",
)


@pytest.mark.parametrize("mode", TRAPS)
def test_release_trap(dtype, mode):
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "traps"
    directory.mkdir(exist_ok=True)
    path = directory / f"{str(dtype).split('.')[-1]}-{mode}.log"
    command = [
        sys.executable,
        "-m",
        "tests.mla_causal_paged_error_probe",
        str(dtype).split(".")[-1],
        mode,
    ]
    with path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    text = path.read_text()
    assert (
        result.returncode != 0
        and "MLA_PAGED_CAUSAL_LAUNCHED" in text
        and "MLA_PAGED_CAUSAL_COMPLETED" not in text
    ), text[-3000:]
    assert any(
        token in text.lower()
        for token in (
            "507035",
            "aicore",
            "device task abort",
            "vector core",
            "acl stream synchronize",
        )
    ), text[-3000:]
