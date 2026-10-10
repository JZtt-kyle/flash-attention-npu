"""CPU TND address validation against PyTorch and an unbounded-integer oracle.

The mixed-layout checks compose the existing allocation/used parser and BSND
address component; no attention launch or production stride support is enabled.
MLA_TND_LAYOUT_DIR preserves commands, cases and source/binary hashes;
MLA_TND_LAYOUT_SANITIZE=1 enables ASan/UBSan for the TND address driver.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import subprocess

import pytest
import torch

from tests.test_flash_attn_npu_v3_mla_bsnd_layout import (
    I64_MAX,
    MAX_ELEMENTS,
    U64_MAX,
    evaluate,
    layout_driver as layout_driver,
    oracle as bsnd_oracle,
    tensor_case,
)
from tests.test_flash_attn_npu_v3_mla_tnd_schedule import compile_driver
from tests.test_flash_attn_npu_v3_mla_used_schedule import execute, used_case

ROOT = Path(__file__).resolve().parents[1]
MLA = ROOT / "csrc/ascend910/flash_attn_npu_3/mla"
SENTINEL = [11, 22, 33]


@pytest.fixture(scope="session")
def tnd_layout_driver(tmp_path_factory):
    target = os.environ.get("MLA_TND_LAYOUT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-tnd-layout")
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "layout"
    sources = [
        MLA / "mla_tnd_layout.cpp",
        MLA / "mla_bsnd_layout.cpp",
        ROOT / "tests/mla_tnd_layout_driver.cpp",
    ]
    command = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O2",
        "-g",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-D_GLIBCXX_ASSERTIONS",
    ]
    if os.environ.get("MLA_TND_LAYOUT_SANITIZE") == "1":
        command += [
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
        ]
    command += ["-I" + str(MLA), *map(str, sources), "-o", str(binary)]
    (directory / "compile-command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / "compile.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    files = [*sources, MLA / "mla_tnd_layout.h", MLA / "mla_bsnd_layout.h", Path(__file__), binary]
    (directory / "manifest.json").write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2)
        + "\n"
    )
    return binary


@pytest.fixture(scope="session")
def sequence_driver(tnd_layout_driver):
    directory = tnd_layout_driver.parent / "lengths"
    directory.mkdir(exist_ok=True)
    return compile_driver(directory, False, used=True)


def case(sizes=(7, 4, 64), strides=(256, 64, 1), offset=0, storage=3584, coords=()):
    return dict(sizes=sizes, strides=strides, offset=offset, storage=storage, coords=coords)


def oracle(item):
    tokens, heads, dim = item["sizes"]
    strides, offset = item["strides"], item["offset"]
    if tokens < 0 or heads <= 0 or dim not in (64, 512):
        return [1, *SENTINEL]
    if min(strides[:2]) < 0 or strides[2] != 1:
        return [2, *SENTINEL]
    if offset < 0:
        return [3, *SENTINEL]
    # No fixed-width arithmetic, BSND mapping or migrated helper in the oracle.
    span = (
        0
        if tokens == 0
        else 1 + sum((size - 1) * stride for size, stride in zip(item["sizes"], strides))
    )
    if span and 2 * (offset + span) > I64_MAX:
        return [4, *SENTINEL]
    if span and 2 * (offset + span) > item["storage"]:
        return [3, *SENTINEL]
    return [0, *strides[:2], span]


def check_cases(binary, cases):
    for item, actual in zip(cases, evaluate(binary, cases)):
        expected = oracle(item)
        if expected[0] == 0:
            expected += [
                sum(c * s for c, s in zip(coord, item["strides"])) for coord in item["coords"]
            ]
        assert actual == expected, item


def make_view(kind, dim, dtype):
    def backing(shape):
        return torch.randn(shape, dtype=dtype)

    if kind == "contiguous":
        return backing((7, 4, dim))
    if kind == "ntd":
        return backing((4, 7, dim)).transpose(0, 1)
    if kind == "token_slice":
        return backing((15, 4, dim))[1::2]
    if kind == "head_slice":
        return backing((7, 9, dim))[:, 1::2]
    if kind == "combined":
        return backing((9, 15, dim + 17))[1::2, 1::2, 7 : dim + 7].transpose(0, 1)
    if kind == "broadcast":
        return backing((1, 1, dim)).expand(7, 4, dim)
    if kind == "overlap":
        return backing((dim + 100,)).as_strided((7, 4, dim), (7, 11, 1), 3)
    if kind == "singleton":
        return backing((dim + 3,)).as_strided((1, 1, dim), (I64_MAX, I64_MAX, 1), 3)
    if kind == "empty_token":
        return backing((0,)).as_strided((0, 4, dim), (I64_MAX, I64_MAX, 1), 99)
    raise AssertionError(kind)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 512])
@pytest.mark.parametrize(
    "kind",
    [
        "contiguous",
        "ntd",
        "token_slice",
        "head_slice",
        "combined",
        "broadcast",
        "overlap",
        "singleton",
        "empty_token",
    ],
)
def test_real_views(tnd_layout_driver, dtype, dim, kind):
    view = make_view(kind, dim, dtype)
    item = tensor_case(view)
    actual = evaluate(tnd_layout_driver, [item])[0]
    assert actual[:4] == oracle(item)
    # Scalar data_ptr is PyTorch's independent address oracle, including offset.
    for coord, offset in zip(item["coords"], actual[4:]):
        assert view[coord].data_ptr() == view.data_ptr() + 2 * offset


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mixed", [False, True], ids=["tnd", "mixed"])
@pytest.mark.parametrize("gaps", [False, True], ids=["t01", "padding"])
@pytest.mark.parametrize("used", ["absent", "query", "kv", "both", "zero"])
def test_independent_views_allocation_and_used(
    tnd_layout_driver, layout_driver, sequence_driver, dtype, mixed, gaps, used
):
    q = (5, 0, 3) if gaps else (3, 0, 1)
    kv = ((37 if gaps else 33),) * 3 if mixed else ((21, 0, 37) if gaps else (17, 0, 33))
    uq = (2, 0, 1) if used in ("query", "both") else ((0, 0, 0) if used == "zero" else None)
    uk = (11, 0, 19) if used in ("kv", "both") else ((0, 0, 0) if used == "zero" else None)
    item = used_case(layout=2 if mixed else 1, q=q, kv=kv, uq=uq, uk=uk, heads=8)
    parsed = execute(sequence_driver, [item], parse=True)[0]
    assert parsed[0] == 0 and len(parsed) == 5 + 6 * len(q)
    spans = [parsed[start : start + 6] for start in range(5, len(parsed), 6)]
    (sequence_driver.parent / f"{mixed}-{gaps}-{used}.json").write_text(
        json.dumps(dict(input=item, output=parsed), indent=2) + "\n"
    )
    views = []
    for index, dim in enumerate((64, 512, 64, 512)):
        query = index < 2
        allocation = q if query else kv
        effective = uq if query else uk
        effective = allocation if effective is None else effective
        heads = 8 if query else 1
        offset = index + 1
        # Q is an NTD view; qv and TND K/V have independently sliced axes.
        if mixed and not query:
            base = torch.full((7, 2 * kv[0] + 1, 3, dim + 19 + index), float("nan"), dtype=dtype)
            view = base[1::2, 1::2, 1::2, offset : offset + dim]
        elif index == 0:
            base = torch.full((heads * 2 + 1, sum(q) * 2 + 1, dim + 19), float("nan"), dtype=dtype)
            view = base[1::2, 1::2, offset : offset + dim].transpose(0, 1)
        else:
            base = torch.full(
                (sum(allocation) * 2 + 1, heads * 2 + 1, dim + 19 + index),
                float("nan"),
                dtype=dtype,
            )
            view = base[1::2, 1::2, offset : offset + dim]
        views.append(view)
        logical = torch.full(view.shape, float("nan"), dtype=dtype)
        start = 0
        for batch, (capacity, length) in enumerate(zip(allocation, effective)):
            selection = (
                (batch, slice(0, length)) if view.ndim == 4 else slice(start, start + length)
            )
            view[selection].fill_(index * 10 + batch + 1)
            logical[selection].fill_(index * 10 + batch + 1)
            start += capacity

        coordinates, expected = [], []
        start_axis = 0 if query else 1
        for batch, span in enumerate(spans):
            start, capacity, length = span[start_axis::2]
            assert start == sum(allocation[:batch])
            assert (capacity, length) == (allocation[batch], effective[batch])
            for row, head, d in itertools.product(range(capacity), range(heads), (0, 1, dim - 1)):
                if view.ndim == 4:
                    coordinates.append((start // kv[0], row, head, d))
                    expected.append(logical[batch, row, head, d])
                else:
                    coordinates.append((start + row, head, d))
                    expected.append(logical[sum(allocation[:batch]) + row, head, d])
        descriptor = dict(tensor_case(view), coords=coordinates)
        driver, reference, width = (
            (layout_driver, bsnd_oracle, 5) if view.ndim == 4 else (tnd_layout_driver, oracle, 4)
        )
        actual = evaluate(driver, [descriptor])[0]
        assert actual[:width] == reference(descriptor)
        flat = torch.empty(0, dtype=dtype).set_(
            view.untyped_storage(), 0, (view.untyped_storage().nbytes() // 2,), (1,)
        )
        for coordinate, relative, wanted in zip(coordinates, actual[width:], expected):
            selected = flat[view.storage_offset() + relative]
            assert selected.data_ptr() == view[coordinate].data_ptr()
            torch.testing.assert_close(selected, wanted, rtol=0, atol=0, equal_nan=True)
        # Truncate only the final allocated byte: even all-zero used must reject.
        short = dict(
            descriptor, storage=2 * (view.storage_offset() + actual[width - 1]) - 1, coords=()
        )
        rejected = evaluate(driver, [short])[0]
        assert rejected == reference(short) and rejected[0] == 3
    assert len({view.storage_offset() for view in views}) == 4


@pytest.mark.parametrize("axis", [0, 1], ids=["token", "head"])
@pytest.mark.parametrize(
    "stride",
    [
        0,
        1,
        65535,
        65536,
        65537,
        (1 << 32) - 1,
        1 << 32,
        (1 << 32) + 1,
        MAX_ELEMENTS - 64,
        MAX_ELEMENTS - 63,
        I64_MAX,
    ],
)
def test_stride_field_and_byte_boundaries(tnd_layout_driver, axis, stride):
    sizes, strides = [1, 1, 64], [I64_MAX, I64_MAX, 1]
    sizes[axis], strides[axis] = 2, stride
    item = case(sizes, strides, storage=U64_MAX)
    if oracle(item)[0] == 0:
        item["coords"] = [(0, 0, 0), (sizes[0] - 1, sizes[1] - 1, 63)]
    check_cases(tnd_layout_driver, [item])


def test_invalid_shapes_strides_and_empty_views(tnd_layout_driver):
    items = []
    for axis in range(3):
        for size in (-1, 0, 1, 63, 64, 128, 512, I64_MAX):
            sizes = [7, 4, 64]
            sizes[axis] = size
            items.append(case(sizes=sizes, storage=U64_MAX))
        for stride in (-1, 0, 2, I64_MAX):
            strides = [256, 64, 1]
            strides[axis] = stride
            items.append(case(strides=strides, storage=U64_MAX))
    for offset in (-1, 0, 99, MAX_ELEMENTS, I64_MAX):
        items.append(case((0, I64_MAX, 512), (I64_MAX, I64_MAX, 1), offset, 0))
    for strides in ((-1, 0, 1), (0, -1, 1), (0, 0, 2)):
        items.append(case((0, 4, 64), strides, 99, 0))
    check_cases(tnd_layout_driver, items)


def test_storage_and_offset_boundaries(tnd_layout_driver):
    items = []
    for offset in (
        -1,
        0,
        1,
        (1 << 32) - 1,
        1 << 32,
        (1 << 32) + 1,
        MAX_ELEMENTS - 64,
        MAX_ELEMENTS - 63,
        I64_MAX,
    ):
        for storage in (0, 127, 128, 129, 130, I64_MAX - 1, I64_MAX, U64_MAX):
            items.append(case((1, 1, 64), (I64_MAX, I64_MAX, 1), offset, storage))
    items += [
        case((I64_MAX, I64_MAX, 512), (I64_MAX, I64_MAX, 1), storage=U64_MAX),
        case((2, 2, 512), (MAX_ELEMENTS // 2, MAX_ELEMENTS // 2, 1), storage=U64_MAX),
        case((I64_MAX, I64_MAX, 64), (0, 0, 1), storage=128),
    ]
    check_cases(tnd_layout_driver, items)


def test_randomized_big_integer_oracle(tnd_layout_driver):
    rng = random.Random(20261010)
    items = []
    for _ in range(1200):
        sizes = [rng.choice([0, 1, 2, 33, 257, I64_MAX]) for _ in range(2)] + [
            rng.choice([64, 512])
        ]
        strides = [rng.choice([0, 1, 13, 64, 512, 65536, 1 << 32, I64_MAX]) for _ in range(2)] + [1]
        offset = rng.choice([0, 1, 17, MAX_ELEMENTS - 64, I64_MAX])
        needed = 2 * (offset + 1 + sum((n - 1) * s for n, s in zip(sizes, strides)))
        storage = max(0, min(U64_MAX, needed + rng.choice([-2, -1, 0, 1, 2])))
        items.append(case(sizes, strides, offset, storage))
    check_cases(tnd_layout_driver, items)
