"""Standalone PA attention helpers and BSND tests with source/independent O/LSE math.

Only the source adapter materializes layouts outside the CANN ABI. Neither
oracle reads the v6 spans to interpret query-used, logical KV or page IDs.
"""

import itertools
import json
import os

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_paged_reference import references as references
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.used_attention import check_math
from tests.mla.support.components.paged_metadata_aicpu import paged_probe as paged_probe, enqueue
from tests.mla.support.components.paged_metadata_consumer import (
    TensorView,
    ARRAY_NAMES,
    CORRUPTIONS,
    CACHE_FAULTS,
)
from tests.mla.support.components.paged_schedule import (
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    execute,
)
from tests.mla.support.components.paged_metadata import decode_record

from tests.mla.support.components.paged_attention import (
    FAULTS as FAULTS,
    PERFORMANCE_CASES as PERFORMANCE_CASES,
    PagedSchedule as PagedSchedule,
    address as address,
    cache_for as cache_for,
    cache_strides as cache_strides,
    check_case as check_case,
    check_rejected as check_rejected,
    config as config,
    configure_producer as configure_producer,
    descriptor as descriptor,
    fault_case as fault_case,
    launch as launch,
    load_attention as load_attention,
    make_case as make_case,
    paged_attention as paged_attention,
    rejection_case as rejection_case,
    valid_outputs as valid_outputs,
)


@pytest.mark.parametrize("page", range(16, 1025, 16))
@pytest.mark.parametrize("profile", ("contiguous", "pa02_k", "pa02_v", "pa02_both"))
def test_all_page_sizes(paged_attention, paged_probe, page, profile):
    # PA01 Q=[3,1], and full BSND allocation in the compact control.
    case = make_case(
        paged_attention,
        paged_probe,
        config(page, profile, uq=None if profile == "contiguous" else (3, 1)),
    )
    check_case(paged_attention, case)


@pytest.mark.parametrize(
    "spec",
    PERFORMANCE_CASES
    + [
        config(
            112,
            "independent",
            (9,) * 5,
            (17, 65, 513, 0, 1),
            (3, 0, 1, 7, 0),
            171,
            aic=1,
            repeat=True,
        ),
        config(528, "independent", (3,) * 3, (512, 0, 513), (2, 0, 1), 256),
        config(16, "independent", (3, 3), (16384, 8193), (1, 2), 8, capacity=16384),
        config(128, "independent", (3,) * 10, (511,) + (0,) * 9, (2,) * 10, 128, aic=15),
        *[
            config(112, "independent", (3, 3), (127, 129), (1, 2), h)
            for h in (1, 15, 16, 127, 128, 129, 255, 256)
        ],
    ],
)
def test_task_ring(paged_attention, paged_probe, spec):
    check_case(
        paged_attention,
        make_case(paged_attention, paged_probe, spec, full_launch=spec["heads"] == 255),
    )


@pytest.mark.parametrize(
    "profile",
    ["bnsd", "independent", "broadcast", "overlap", "table_broadcast", "table_overlap"]
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
        make_case(paged_attention, paged_probe, config(112, profile, lengths=lengths)),
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


@pytest.mark.parametrize("layout,pool,used", itertools.product((0, 2), (0, 7), (False, True)))
def test_zero_lengths_with_static_workspace(paged_attention, paged_probe, layout, pool, used):
    spec = config(
        layout=layout,
        q=(3, 3) if layout == 0 else (3, 1),
        lengths=(0, 0),
        uq=(2, 0) if used else None,
        pool=pool,
        capacity=448,
    )
    case = make_case(paged_attention, paged_probe, spec, full_launch=True)
    assert case["context"].block_dim == paged_probe["platform"].aic
    assert case["expected"][20] == 1
    case["context"].workspace = 1 << 20
    case["context"].libapi = paged_probe["platform"].libapi
    base = torch.full((case["context"].workspace + 1024,), 205, dtype=torch.uint8, device="npu")
    case["ws_bases"]["migrated"] = base
    case["workspace"]["migrated"] = base[512:-512]
    check_case(paged_attention, case)
    assert (base.cpu() == 205).all()  # An empty schedule never consumes the reservation.


@pytest.mark.parametrize(
    "q,kv,used,capacity",
    (([], [], None, 0), ([0, 0], [257, 1], None, 448), ([3, 3], [257, 1], [0, 0], 448)),
)
def test_zero_query(paged_attention, paged_probe, q, kv, used, capacity):
    check_case(
        paged_attention,
        make_case(
            paged_attention, paged_probe, config(q=q, lengths=kv, uq=used, capacity=capacity)
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
            config(112, lengths=(1025, 513)),
            scale=scale,
            perturb=perturb,
            amplitude=4,
        ),
    )


def test_independent_schedules(paged_attention, paged_probe, paged_oracles):
    specs = PERFORMANCE_CASES + [config(p) for p in range(16, 1025, 16)]
    items = []
    rows = []
    for spec in specs:
        case = make_case(paged_attention, paged_probe, spec)
        items.append(case["item"])
        rows.append(case["expected"])
    source, native = execute(paged_oracles[0], items), execute(paged_oracles[0], items, native=True)
    kind = 1 if paged_attention[0]["dtype"] == torch.float16 else 2
    for item, row, wanted in zip(items, rows, source):
        decode_record(row, item, kind, wanted)
    (paged_attention[1] / "schedules.json").write_text(
        json.dumps(dict(items=items, source=source, native=native, metadata=rows), indent=2) + "\n"
    )


def test_corruption_precedes_attention(paged_attention, paged_probe):
    case = rejection_case(paged_attention, paged_probe)
    for word, value, status in CORRUPTIONS:
        old = case["expected"][word]
        assert old != value
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        with forbid_host_readback():
            assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
        check_rejected(case, status)
        case["metadata"][word] = old
    check_case(paged_attention, case)


def test_cache_descriptors_precede_attention(paged_attention, paged_probe):
    case = rejection_case(paged_attention, paged_probe)
    for path, value, status in CACHE_FAULTS:
        with forbid_host_readback():
            assert (
                launch(
                    paged_attention,
                    case,
                    checked=False,
                    diagnostic=True,
                    cache_changes={path: value},
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
    for axis, name in enumerate(ARRAY_NAMES):
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


@pytest.mark.parametrize("axis", (1, 2))
def test_stale_lengths_and_regeneration(paged_attention, paged_probe, axis):
    case = rejection_case(paged_attention, paged_probe)
    case["inputs"][axis][1] = 0
    with forbid_host_readback():
        assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
    check_rejected(case, 13 if axis == 1 else 11)
    # The next call regenerates from the changed device length arrays.
    spec = dict(case["config"])
    field = "uq" if axis == 1 else "lengths"
    spec[field] = [spec[field][0], 0]
    updated = make_case(paged_attention, paged_probe, spec)
    updated["metadata_storage"] = case["metadata_storage"]
    updated["metadata"] = case["metadata"]
    updated["inputs"] = case["inputs"]
    with forbid_host_readback():
        enqueue(
            paged_probe, updated["args"], updated["metadata_storage"], updated["inputs"], worker=1
        )
    check_case(paged_attention, updated)


@pytest.mark.parametrize("empty_query", (False, True))
def test_pages_rechecked_even_without_queries(paged_attention, paged_probe, empty_query):
    spec = config(q=(0, 0) if empty_query else (3, 3), uq=None if empty_query else (3, 0))
    case = make_case(paged_attention, paged_probe, spec)
    # Batch 1 has no active Q, but its entire effective page prefix is required.
    for value in (-1, -(2**31), spec["pool"], 2**31 - 1):
        case["tables"]["migrated"][1, 1] = value
        assert launch(paged_attention, case, checked=False, diagnostic=True) == 0
        check_rejected(case, 15)


@pytest.mark.parametrize("profile", ("contiguous", "pa02_both", "independent", "broadcast"))
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
        for axis in ((0, 1, 2) if name in ("q", "qv") else (0, 1))
    ],
)
def test_wide_active_stride(paged_attention, paged_probe, name, axis):
    case = make_case(
        paged_attention,
        paged_probe,
        config(16, f"wide_{name}_{axis}", q=(2, 2), lengths=(17, 17), uq=(2, 1), heads=2, pool=2),
    )
    check_case(paged_attention, case)


@pytest.mark.parametrize("field", ("negative", "inner", "storage", "offset", "shape", "overflow"))
def test_invalid_query_view(paged_attention, paged_probe, field):
    case = rejection_case(paged_attention, paged_probe)
    views = (TensorView * 2).from_buffer_copy(case["views"]["migrated"])
    if field == "negative":
        views[0].strides[1] = -1
    elif field == "inner":
        views[1].strides[3] = 2
    elif field == "storage":
        views[1].storage = 2
    elif field == "offset":
        views[1].offset = 2**63 - 1
    elif field == "shape":
        views[0].sizes[1] += 1
    else:
        views[0].strides[0] = 2**63 - 1
    assert launch(paged_attention, case, views=views) == -3
    for t in case["output_bases"]["migrated"]:
        assert (t.cpu() == 123).all()
    assert (case["ws_bases"]["migrated"].cpu() == 205).all()


def test_stream_and_allocator_lifetime(paged_attention, paged_probe):
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    saved = []
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
        # Recycle inputs/workspace on the default stream while worker kernels
        # still own them via record_stream; each invocation owns its workspace.
        churn = torch.empty(8 << 20, dtype=torch.uint8, device="npu").fill_(17)
        del churn
    for outputs, item, refs, event in saved:
        event.synchronize()
        actual = valid_outputs(dict(item=item), [t.cpu() for t in outputs])
        check_math(actual, refs)


@pytest.mark.parametrize("fault", FAULTS)
def test_release_trap(paged_attention, paged_probe, fault):
    import signal
    import subprocess
    import sys

    directory = paged_attention[1] / f"error-{fault}"
    directory.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_paged_attention_error_probe",
        str(paged_attention[1]),
        str(paged_probe["directory"] / "build-manifest.json"),
        fault,
        str(directory),
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
    assert report["fault"] == fault and report["error"]


def test_wide_page_table_and_singletons(paged_attention, paged_probe):
    torch.npu.empty_cache()
    case = make_case(paged_attention, paged_probe, config(profile="table_wide"))
    assert case["tables"]["migrated"].stride(0) > 2**32
    check_case(paged_attention, case)
    del case
    torch.npu.empty_cache()
    case = make_case(
        paged_attention,
        paged_probe,
        config(16, "singleton", q=(1,), lengths=(1,), uq=None, heads=1, pool=1),
    )
    check_case(paged_attention, case)


@pytest.mark.parametrize("heads", (1, 15, 129))
def test_empty_small_row_counts(paged_attention, paged_probe, heads):
    check_case(
        paged_attention,
        make_case(
            paged_attention,
            paged_probe,
            config(
                16, "contiguous", q=(1,), lengths=(0,), uq=None, heads=heads, pool=0, capacity=0
            ),
        ),
    )
