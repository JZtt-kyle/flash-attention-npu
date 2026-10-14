"""CPU checks for reference reuse; no compiler, extension or NPU required."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tests.mla.support import mla_paged_reference as reference


def make_case(layout=0, dtype=torch.float16):
    generator = torch.Generator().manual_seed(24)
    query = (1, 5, 2) if layout == 2 else (2, 3, 2)
    shapes = dict(q=(*query, 64), qv=(*query, 512), k=(3, 16, 1, 64), v=(3, 16, 1, 512))
    host = {
        name: torch.randn(shape, generator=generator).to(dtype) for name, shape in shapes.items()
    }
    return dict(
        host=host,
        item=dict(layout=layout, q=[3, 2], uq=[2, 1]),
        config=dict(page=16, lengths=[17, 0]),
        context=SimpleNamespace(heads=2),
        table_host=torch.tensor([[2, 0], [1, 0]], dtype=torch.int32),
        scale=1 / 24,
    )


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    monkeypatch.setenv("GOLDEN_CACHE_REFRESH", "0")
    monkeypatch.setenv("GOLDEN_CACHE_STATS_FILE", str(tmp_path / "events.tsv"))
    return tmp_path


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", [0, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("used", [None, [2, 1], [0, 0]])
def test_hit_matches_uncached_for_layout_used_and_empty(
    cache, monkeypatch, layout, dtype, used, causal
):
    case = make_case(layout, dtype)
    case["item"]["uq"] = used
    case["causal"] = causal
    expected = reference.uncached_references(case)
    fresh = reference.references(case)

    def forbidden(_):
        raise AssertionError("cache hit unexpectedly recomputed the golden")

    monkeypatch.setattr(reference, "uncached_references", forbidden)
    hit = reference.references(case)
    for a, b, c in zip(expected, fresh, hit):
        assert torch.equal(a, b) and torch.equal(a, c)
    assert "hit\t" in (cache / "events.tsv").read_text()


@pytest.mark.parametrize(
    "field",
    [
        "q",
        "qv",
        "k",
        "v",
        "page_table",
        "scale",
        "causal",
        "used",
        "prefix",
        "length",
        "stride",
        "offset",
        "layout",
        "source",
    ],
)
def test_changed_input_or_adapter_cannot_hit(cache, monkeypatch, tmp_path, field):
    case = make_case()
    count = []

    def compute(_):
        count.append(1)
        return [torch.zeros(1) for _ in range(4)]

    monkeypatch.setattr(reference, "uncached_references", compute)
    reference.references(case)
    other = copy.deepcopy(case)
    if field in case["host"]:
        other["host"][field].flatten()[0] += 1
    elif field == "page_table":
        other["table_host"][0, 0] = 1
    elif field == "causal":
        other["causal"] = True
    elif field == "scale":
        other["scale"] *= 2
    elif field == "used":
        other["item"]["uq"] = [1, 1]
    elif field == "prefix":
        other["item"]["q"] = [2, 3]
    elif field == "length":
        other["config"]["lengths"] = [16, 0]
    elif field == "layout":
        other["item"]["layout"] = 2
    elif field in ("stride", "offset"):
        q = other["host"]["q"]
        if field == "stride":
            storage = torch.empty((*q.shape[:-1], q.shape[-1] + 8), dtype=q.dtype)
            view = storage[..., : q.shape[-1]]
        else:
            storage = torch.empty(q.numel() + 1, dtype=q.dtype)
            view = storage[1:].view_as(q)
        view.copy_(q)
        other["host"]["q"] = view
    else:
        source = tmp_path / "changed_adapter.py"
        source.write_text("# changed adapter\n")
        monkeypatch.setattr(reference, "__file__", str(source))
    reference.references(other)
    assert len(count) == 2


def test_off_refresh_and_corruption_recompute(cache, monkeypatch):
    case = make_case()
    expected = reference.references(case)
    artifact = next(cache.rglob("case_*.tar.gz"))
    artifact.write_bytes(b"interrupted cache write")
    with pytest.warns(RuntimeWarning, match="recomputing"):
        recovered = reference.references(case)
    monkeypatch.setenv("GOLDEN_CACHE_REFRESH", "1")
    refreshed = reference.references(case)
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "off")
    disabled = reference.references(case)
    for group in (recovered, refreshed, disabled):
        assert all(torch.equal(a, b) for a, b in zip(expected, group))
