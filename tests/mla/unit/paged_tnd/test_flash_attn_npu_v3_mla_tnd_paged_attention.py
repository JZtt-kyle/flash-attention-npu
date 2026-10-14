"""Independent TND Q / ND PA attention; allocation prefixes never follow used.

Source Q is independently packed before its pinned PA/non-FD invocation. KV
lengths remain non-cumulative. The shared math adapter indexes original pages
and allocation segments, without reading v6 spans or device Copy addresses.
"""

import itertools
import json
import os
import signal
import subprocess
import sys

import pytest
import torch

from tests.mla.unit.paged_bsnd import test_flash_attn_npu_v3_mla_paged_attention as bsnd
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_attention import (
    paged_attention as paged_attention,
    paged_probe as paged_probe,
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    make_case,
    launch,
    check_case,
    check_rejected,
    references,
    valid_outputs,
    load_attention as load_attention,
    configure_producer as configure_producer,
    forbid_host_readback,
    enqueue,
)


def config(page=112, profile="independent", q=(5, 2), lengths=None, uq=(3, 1), heads=8, **kwargs):
    return bsnd.config(page, profile, q, lengths, uq, heads, layout=2, **kwargs)


# Frozen before measurement: native PA01/PA02 page strides, independent Q
# allocations, zero-Q/zero-KV samples and enough physical pages for long KV.
PERFORMANCE_CASES = [
    config(p, profile, q, kv, uq, h, pool=2 + sum((n + p - 1) // p for n in kv))
    for p in (16, 112, 128, 256, 528, 1024)
    for profile, q, kv, uq, h in (
        ("contiguous", (1, 0, 1), (257, 17, 0), None, 128),
        ("pa02_both", (7, 0, 3), (8193, 17, 1025), (3, 0, 1), 128),
    )
]


@pytest.mark.parametrize("page", range(16, 1025, 16))
@pytest.mark.parametrize("profile", ("contiguous", "pa02_k", "pa02_v", "pa02_both"))
def test_all_page_sizes(paged_attention, paged_probe, page, profile):
    case = make_case(
        paged_attention,
        paged_probe,
        config(
            page,
            profile,
            (5, 0, 3),
            (page + 1, 0, 2 * page - 3),
            None if profile == "contiguous" else (3, 0, 1),
            heads=16,
        ),
    )
    assert case["outputs"]["migrated"][0].shape == (8, 16, 512)
    assert case["outputs"]["migrated"][1].shape == (16, 8)
    check_case(paged_attention, case)


@pytest.mark.parametrize(
    "spec",
    PERFORMANCE_CASES
    + [
        config(
            112,
            "ntd",
            (7, 0, 3, 9, 1),
            (17, 65, 513, 0, 1),
            (3, 0, 1, 7, 0),
            171,
            aic=1,
            repeat=True,
        ),
        config(528, "independent", (5, 0, 3), (512, 0, 513), (2, 0, 1), 256),
        config(16, "independent", (3, 2), (16384, 8193), (1, 2), 8, capacity=16384),
        config(128, "independent", (3, 2) * 5, (511,) + (0,) * 9, (2, 1) * 5, 128, aic=15),
        *[
            config(112, "independent", (5, 2), (127, 129), (1, 2), h)
            for h in (1, 15, 16, 127, 128, 129, 255, 256)
        ],
    ],
)
def test_task_ring(paged_attention, paged_probe, spec):
    case = make_case(paged_attention, paged_probe, spec, full_launch=spec["heads"] == 255)
    check_case(paged_attention, case)


@pytest.mark.parametrize(
    "profile",
    ["ntd", "independent", "broadcast", "overlap", "table_broadcast", "table_overlap"]
    + [
        f"{axis}_{s}"
        for axis, s in itertools.product(
            ("row", "matrix", "head", "combined"), (65535, 65536, 65537)
        )
    ],
)
def test_strides(paged_attention, paged_probe, profile):
    lengths = (257, 257) if profile == "table_broadcast" else (257, 17)
    check_case(
        paged_attention,
        make_case(paged_attention, paged_probe, config(profile=profile, lengths=lengths)),
    )


@pytest.mark.parametrize("pool,capacity", ((7, 0), (0, 0), (0, 224), (7, 224)))
@pytest.mark.parametrize("used", (None, (3, 1), (0, 0)))
def test_empty_cache(paged_attention, paged_probe, pool, capacity, used):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            config(lengths=(0, 0), uq=used, pool=pool, capacity=capacity),
        ),
    )


@pytest.mark.parametrize(
    "q,kv,used,capacity",
    (([], [], None, 0), ([0, 0], [257, 1], None, 448), ([5, 2], [257, 1], [0, 0], 448)),
)
def test_zero_query(paged_attention, paged_probe, q, kv, used, capacity):
    check_case(
        paged_attention,
        make_case(
            paged_attention, paged_probe, config(q=q, lengths=kv, uq=used, capacity=capacity)
        ),
    )


@pytest.mark.parametrize("heads", (1, 15, 129))
def test_empty_small_rows(paged_attention, paged_probe, heads):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            config(16, "contiguous", (0, 1), (0, 0), None, heads, pool=0, capacity=0),
        ),
    )


@pytest.mark.parametrize("scale", (0.0, -0.125, 0.25, None))
@pytest.mark.parametrize("perturb", range(4))
def test_scale_halves_and_max_change(paged_attention, paged_probe, scale, perturb):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            config(lengths=(1025, 513)),
            scale=scale,
            perturb=perturb,
            amplitude=4,
        ),
    )


def test_independent_schedules(paged_attention, paged_probe, paged_oracles):
    items, rows = [], []
    for spec in PERFORMANCE_CASES + [config(p) for p in range(16, 1025, 16)]:
        case = make_case(paged_attention, paged_probe, spec)
        items.append(case["item"])
        rows.append(case["expected"])
    source, native = (
        bsnd.execute(paged_oracles[0], items),
        bsnd.execute(paged_oracles[0], items, native=True),
    )
    kind = 1 if paged_attention[0]["dtype"] == torch.float16 else 2
    for item, row, wanted in zip(items, rows, source):
        bsnd.decode_record(row, item, kind, wanted)
    (paged_attention[1] / "tnd-schedules.json").write_text(
        json.dumps(dict(items=items, source=source, native=native, metadata=rows), indent=2) + "\n"
    )


def rejection_case(probes, producer):
    return make_case(
        probes, producer, config(q=(5, 2), uq=(3, 1), lengths=(257, 257), capacity=448, aic=1)
    )


def test_cache_and_length_descriptors_precede_attention(paged_attention, paged_probe):
    case = rejection_case(paged_attention, paged_probe)
    for path, value, status in bsnd.CACHE_FAULTS:
        assert (
            launch(
                paged_attention, case, checked=False, diagnostic=True, cache_changes={path: value}
            )
            == 0
        )
        check_rejected(case, status)
    for words in (0, 8, 407, case["metadata"].numel() - 1, case["metadata"].numel() + 1, 2**32 - 1):
        assert launch(paged_attention, case, checked=False, diagnostic=True, words=words) == 0
        check_rejected(case, 1)
    for pointer in (0, 1, 2**64 - 4):
        assert (
            launch(paged_attention, case, checked=False, diagnostic=True, metadata_address=pointer)
            == 0
        )
        check_rejected(case, 1)
    for axis, name in enumerate(bsnd.ARRAY_NAMES):
        for field, value in (("address", 1), ("address", 2**64 - 4), ("count", 2**64 - 1)):
            assert (
                launch(
                    paged_attention,
                    case,
                    checked=False,
                    diagnostic=True,
                    descriptors={f"{name}.{field}": value},
                )
                == 0
            )
            check_rejected(case, (12, 13, 11)[axis])
    check_case(paged_attention, case)


def test_corruption_precedes_attention(paged_attention, paged_probe):
    case = rejection_case(paged_attention, paged_probe)
    # All shared ABI error categories apply to TND too.
    for word, value, status in bsnd.CORRUPTIONS:
        old = case["expected"][word]
        if old == value:
            continue
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
        check_rejected(case, status)
        case["metadata"][word] = old
    # Second allocation starts at 5; Q-used of the first sample is only 3.
    for word, value, status in ((413, 3, 12), (414, 1, 12), (415, 1, 12), (416, 3, 13)):
        old = case["expected"][word]
        case["metadata"][word] = value
        assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
        check_rejected(case, status)
        case["metadata"][word] = old
    check_case(paged_attention, case)


@pytest.mark.parametrize("axis,status", ((0, 12), (1, 13), (2, 11)))
def test_stale_lengths_and_regeneration(paged_attention, paged_probe, axis, status):
    case = rejection_case(paged_attention, paged_probe)
    case["inputs"][axis][1] = 4 if axis == 0 else 0
    with forbid_host_readback():
        assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
    check_rejected(case, status)
    spec = dict(case["config"])
    if axis == 0:
        spec["q"] = [4, 3]
    else:
        field = "uq" if axis == 1 else "lengths"
        spec[field] = [spec[field][0], 0]
    updated = make_case(paged_attention, paged_probe, spec)
    updated["metadata_storage"], updated["metadata"] = case["metadata_storage"], case["metadata"]
    updated["inputs"] = case["inputs"]
    with forbid_host_readback():
        enqueue(
            paged_probe, updated["args"], updated["metadata_storage"], updated["inputs"], worker=1
        )
    check_case(paged_attention, updated)


@pytest.mark.parametrize("empty_query", (False, True))
def test_pages_rechecked_without_queries(paged_attention, paged_probe, empty_query):
    spec = config(q=(0, 0) if empty_query else (5, 2), uq=None if empty_query else (3, 0))
    case = make_case(paged_attention, paged_probe, spec)
    for value in (-1, -(2**31), spec["pool"], 2**31 - 1):
        case["tables"]["migrated"][1, 1] = value
        assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
        check_rejected(case, 15)


@pytest.mark.parametrize("profile", ("contiguous", "pa02_both", "ntd", "broadcast"))
def test_reuse_with_new_cache_and_scale(paged_attention, paged_probe, profile):
    old = make_case(paged_attention, paged_probe, config())
    new = make_case(paged_attention, paged_probe, config(profile=profile, pool=11), scale=-0.125)
    assert new["expected"] == old["expected"]
    new["metadata_storage"], new["metadata"] = old["metadata_storage"], old["metadata"]
    check_case(paged_attention, new)


@pytest.mark.parametrize(
    "name,axis",
    [
        (name, axis)
        for name in ("q", "qv", "k", "v")
        for axis in ((1, 2) if name in ("q", "qv") else (0, 1))
    ],
)
def test_wide_active_stride(paged_attention, paged_probe, name, axis):
    torch.npu.empty_cache()
    case = make_case(
        paged_attention,
        paged_probe,
        config(16, f"wide_{name}_{axis}", (1, 1), (17, 17), None, 2, pool=2),
    )
    check_case(paged_attention, case)


def test_wide_page_table_and_singletons(paged_attention, paged_probe):
    torch.npu.empty_cache()
    case = make_case(paged_attention, paged_probe, config(profile="table_wide"))
    assert case["tables"]["migrated"].stride(0) > 2**32
    check_case(paged_attention, case)
    del case
    torch.npu.empty_cache()
    check_case(
        paged_attention,
        make_case(
            paged_attention, paged_probe, config(16, "singleton", (0, 1), (0, 1), None, 1, pool=1)
        ),
    )


@pytest.mark.parametrize("field", ("negative", "inner", "storage", "offset", "shape", "overflow"))
@pytest.mark.parametrize("zero_used", (False, True))
def test_invalid_query_view(paged_attention, paged_probe, field, zero_used):
    case = make_case(paged_attention, paged_probe, config(uq=(0, 0) if zero_used else (3, 1)))
    for index in range(2):
        views = (bsnd.TensorView * 2).from_buffer_copy(case["views"]["migrated"])
        view = views[index]
        if field == "negative":
            view.strides[1] = -1
        elif field == "inner":
            view.strides[3] = 2
        elif field == "storage":
            view.storage = 2
        elif field == "offset":
            view.offset = 2**63 - 1
        elif field == "shape":
            view.sizes[1] += 1
        else:
            view.strides[1] = 2**63 - 1
        assert launch(paged_attention, case, views=views) == -3
        for tensor in case["output_bases"]["migrated"]:
            assert (tensor.cpu() == 123).all()
        assert (case["ws_bases"]["migrated"].cpu() == 205).all()


def test_stream_and_allocator_lifetime(paged_attention, paged_probe):
    streams, saved = [torch.npu.Stream(), torch.npu.Stream()], []
    for index in range(6):
        stream = streams[index % 2]
        with torch.npu.stream(stream):
            case = make_case(paged_attention, paged_probe, config(112 if index % 2 else 528))
            refs = references(case)
            with forbid_host_readback():
                assert launch(paged_attention, case) == 0
            event = torch.npu.Event()
            event.record(stream)
            saved.append((case["outputs"]["migrated"], case["item"], refs, event))
        del case
        churn = torch.empty(8 << 20, dtype=torch.uint8, device="npu").fill_(17)
        del churn
    for outputs, item, refs, event in saved:
        event.synchronize()
        bsnd.check_math(valid_outputs(dict(item=item), [t.cpu() for t in outputs]), refs)


@pytest.mark.parametrize("fault", (*bsnd.FAULTS, "stale_prefix"))
def test_release_trap(paged_attention, paged_probe, fault):
    directory = paged_attention[1] / f"tnd-error-{fault}"
    directory.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_paged_attention_error_probe",
        str(paged_attention[1]),
        str(paged_probe["directory"] / "build-manifest.json"),
        fault,
        str(directory),
        "2",
    ]
    (directory / "command.json").write_text(json.dumps(command) + "\n")
    with (directory / "subprocess.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            result = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    assert result == 0, (directory / "subprocess.log").read_text()[-3000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["fault"] == fault and report["layout"] == 2 and report["error"]
