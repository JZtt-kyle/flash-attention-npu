"""CPU address checks against real PyTorch views and an unbounded-integer oracle.

No attention launch or public stride support is provided by this component.
MLA_BSND_LAYOUT_DIR preserves build commands and complete driver cases;
MLA_BSND_LAYOUT_SANITIZE=1 enables ASan/UBSan on the same acceptance matrix.
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

ROOT = Path(__file__).resolve().parents[1]
MLA = ROOT / "csrc/ascend910/flash_attn_npu_3/mla"
I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1
MAX_ELEMENTS = I64_MAX // 2
SENTINEL = [11, 22, 33, 44]


@pytest.fixture(scope="session")
def layout_driver(tmp_path_factory):
    target = os.environ.get("MLA_BSND_LAYOUT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-bsnd-layout")
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "layout"
    sources = [MLA / "mla_bsnd_layout.cpp", ROOT / "tests/mla_bsnd_layout_driver.cpp"]
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
    if os.environ.get("MLA_BSND_LAYOUT_SANITIZE") == "1":
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
    files = [*sources, MLA / "mla_bsnd_layout.h", Path(__file__), binary]
    (directory / "manifest.json").write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2)
        + "\n"
    )
    return binary


def case(sizes=(2, 3, 4, 64), strides=(768, 256, 64, 1), offset=0, storage=3072, coords=()):
    return dict(sizes=sizes, strides=strides, offset=offset, storage=storage, coords=coords)


def evaluate(binary, cases):
    rows = []
    for item in cases:
        rows.append(
            [
                *item["sizes"],
                *item["strides"],
                item["offset"],
                item["storage"],
                len(item["coords"]),
                *itertools.chain.from_iterable(item["coords"]),
            ]
        )
    text = "".join(" ".join(map(str, row)) + "\n" for row in rows)
    label = hashlib.sha256(text.encode()).hexdigest()[:20]
    (binary.parent / (label + "-input.json")).write_text(json.dumps(cases, indent=2) + "\n")
    result = subprocess.run([str(binary)], input=text, capture_output=True, text=True, timeout=30)
    (binary.parent / (label + "-output.txt")).write_text(result.stdout)
    (binary.parent / (label + "-stderr.txt")).write_text(result.stderr)
    result.check_returncode()
    output = [list(map(int, row.split(","))) for row in result.stdout.splitlines()]
    assert len(output) == len(cases)
    return output


def oracle(item):
    b, s, h, d = item["sizes"]
    strides, offset = item["strides"], item["offset"]
    if b < 0 or s < 0 or h <= 0 or d not in (64, 512):
        return [1, *SENTINEL]
    if min(strides[:3]) < 0 or strides[3] != 1:
        return [2, *SENTINEL]
    if offset < 0:
        return [3, *SENTINEL]
    # Python integers never wrap. This oracle does not reproduce the C++
    # checked-arithmetic algorithm or consume any migrated address helper.
    span = (
        0
        if b == 0 or s == 0
        else 1 + sum((size - 1) * stride for size, stride in zip(item["sizes"], strides))
    )
    if span and 2 * (offset + span) > I64_MAX:
        return [4, *SENTINEL]
    if span and 2 * (offset + span) > item["storage"]:
        return [3, *SENTINEL]
    return [0, *strides[:3], span]


def check_cases(binary, cases):
    for item, actual in zip(cases, evaluate(binary, cases)):
        assert actual == oracle(item), item


def tensor_case(view):
    coords = list(
        itertools.product(*(range(n) for n in view.shape[:-1]), (0, 1, view.shape[-1] - 1))
    )
    return case(
        view.shape, view.stride(), view.storage_offset(), view.untyped_storage().nbytes(), coords
    )


def make_view(kind, dim, dtype):
    def backing(shape):
        return torch.randn(shape, dtype=dtype)

    if kind == "contiguous":
        return backing((2, 3, 4, dim))
    if kind == "bnsd":
        return backing((2, 4, 3, dim)).transpose(1, 2)
    if kind == "batch_slice":
        return backing((5, 3, 4, dim))[1::2]
    if kind == "row_slice":
        return backing((2, 7, 4, dim))[:, 1::2]
    if kind == "head_slice":
        return backing((2, 3, 9, dim))[:, :, 1::2]
    if kind == "combined":
        return backing((5, 9, 7, dim + 17))[1::2, 1::2, 1::2, 7 : dim + 7].transpose(1, 2)
    if kind == "broadcast":
        return backing((1, 3, 1, dim)).expand(2, 3, 4, dim)
    if kind == "overlap":
        return backing((dim + 100,)).as_strided((2, 3, 4, dim), (7, 11, 5, 1), 3)
    if kind == "singleton":
        return backing((dim + 3,)).as_strided((1, 1, 1, dim), (I64_MAX, I64_MAX, I64_MAX, 1), 3)
    if kind == "empty_batch":
        return backing((0,)).as_strided((0, 3, 4, dim), (I64_MAX, I64_MAX, I64_MAX, 1), 99)
    if kind == "empty_row":
        return backing((0,)).as_strided((2, 0, 4, dim), (I64_MAX, I64_MAX, I64_MAX, 1), 99)
    raise AssertionError(kind)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 512])
@pytest.mark.parametrize(
    "kind",
    [
        "contiguous",
        "bnsd",
        "batch_slice",
        "row_slice",
        "head_slice",
        "combined",
        "broadcast",
        "overlap",
        "singleton",
        "empty_batch",
        "empty_row",
    ],
)
def test_real_views(layout_driver, dtype, dim, kind):
    view = make_view(kind, dim, dtype)
    item = tensor_case(view)
    actual = evaluate(layout_driver, [item])[0]
    assert actual[:5] == oracle(item)
    # Scalar indexing is PyTorch's independent address oracle. In particular,
    # scalar pointers include storage_offset, whereas driver offsets do not.
    for coord, offset in zip(item["coords"], actual[5:]):
        assert view[coord].data_ptr() == view.data_ptr() + 2 * offset


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_independent_q_qv_k_v_offsets_and_used(layout_driver, dtype):
    shapes = [(2, 7, 8, 64), (2, 7, 8, 512), (2, 33, 1, 64), (2, 33, 1, 512)]
    views = []
    for index, (b, s, h, d) in enumerate(shapes):
        base = torch.full(
            (b * 2 + 1, s * 2 + 1, h * 2 + 1, d + 13 + index), float("nan"), dtype=dtype
        )
        view = base[1::2, 1::2, 1::2, index + 1 : d + index + 1]
        for batch, used in enumerate((3, 1) if index < 2 else (17, 0)):
            view[batch, :used].fill_(index + batch + 1)
        views.append(view)
    cases = [tensor_case(view) for view in views]
    outputs = evaluate(layout_driver, cases)
    assert len({view.storage_offset() for view in views}) == 4
    for index, (view, item, actual) in enumerate(zip(views, cases, outputs)):
        assert actual[:5] == oracle(item)
        flat = torch.empty(0, dtype=dtype).set_(
            view.untyped_storage(), 0, (item["storage"] // 2,), (1,)
        )
        for coord, offset in zip(item["coords"], actual[5:]):
            selected = flat[view.storage_offset() + offset]
            assert selected.data_ptr() == view[coord].data_ptr()
            batch, row = coord[:2]
            used = ((3, 1) if index < 2 else (17, 0))[batch]
            assert bool(torch.isnan(selected)) == (row >= used)
        # A short used length must never make a truncated allocation acceptable.
        short = dict(item, storage=2 * (view.storage_offset() + actual[4]) - 1, coords=())
        assert evaluate(layout_driver, [short])[0] == [3, *SENTINEL]


@pytest.mark.parametrize("axis", [0, 1, 2])
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
def test_stride_field_and_byte_boundaries(layout_driver, axis, stride):
    sizes, strides = [1, 1, 1, 64], [I64_MAX, I64_MAX, I64_MAX, 1]
    sizes[axis], strides[axis] = 2, stride
    check_cases(layout_driver, [case(sizes, strides, storage=U64_MAX)])


def test_invalid_shapes_strides_and_empty_views(layout_driver):
    items = []
    for axis in range(4):
        for size in (-1, 0, 1, 63, 64, 128, 512, I64_MAX):
            sizes = [2, 3, 4, 64]
            sizes[axis] = size
            items.append(case(sizes=sizes, storage=U64_MAX))
        for stride in (-1, 0, 2, I64_MAX):
            strides = [768, 256, 64, 1]
            strides[axis] = stride
            items.append(case(strides=strides, storage=U64_MAX))
    for sizes in ((0, I64_MAX, I64_MAX, 512), (I64_MAX, 0, I64_MAX, 64)):
        for offset in (-1, 0, 99, MAX_ELEMENTS, I64_MAX):
            items.append(case(sizes, (I64_MAX, I64_MAX, I64_MAX, 1), offset, 0))
        items.append(case(sizes, (-1, 0, 0, 1), 0, 0))
        items.append(case(sizes, (0, 0, 0, 2), 0, 0))
    check_cases(layout_driver, items)


def test_storage_and_offset_boundaries(layout_driver):
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
            items.append(case((1, 1, 1, 64), (I64_MAX, I64_MAX, I64_MAX, 1), offset, storage))
    # Products and sums exceed uint64 as well as int64 in these cases.
    items += [
        case((I64_MAX, I64_MAX, I64_MAX, 512), (I64_MAX, I64_MAX, I64_MAX, 1), storage=U64_MAX),
        case((2, 2, 2, 512), (MAX_ELEMENTS // 2, MAX_ELEMENTS // 2, 1, 1), storage=U64_MAX),
        case((I64_MAX, I64_MAX, I64_MAX, 64), (0, 0, 0, 1), storage=128),
    ]
    check_cases(layout_driver, items)


def test_randomized_big_integer_oracle(layout_driver):
    rng = random.Random(20261010)
    items = []
    for _ in range(1200):
        sizes = [rng.choice([0, 1, 2, 33, 257, I64_MAX]) for _ in range(3)] + [
            rng.choice([64, 512])
        ]
        strides = [rng.choice([0, 1, 13, 64, 512, 65536, 1 << 32, I64_MAX]) for _ in range(3)] + [1]
        offset = rng.choice([0, 1, 17, MAX_ELEMENTS - 64, I64_MAX])
        needed = 2 * (offset + 1 + sum((n - 1) * s for n, s in zip(sizes, strides)))
        storage = max(0, min(U64_MAX, needed + rng.choice([-2, -1, 0, 1, 2])))
        items.append(case(sizes, strides, offset, storage))
    check_cases(layout_driver, items)
