"""CPU proof that reference reuse preserves both independent equations."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from tests.common import mla_ref as reference
from tests.common import mla_reference_cache as cache_module


def inputs(dtype=torch.float16, prefill=False, sq=3, sk=5):
    gen = torch.Generator().manual_seed(48)
    shapes = dict(
        q=(1, sq, 2, 192 if prefill else 64),
        k=(1, sk, 1, 192 if prefill else 64),
        v=(1, sk, 1, 128 if prefill else 512),
    )
    result = {
        name: torch.randn(shape, generator=gen).to(dtype) / 4 for name, shape in shapes.items()
    }
    result["qv"] = None if prefill else torch.randn(1, sq, 2, 512, generator=gen).to(dtype) / 4
    return result


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    monkeypatch.setenv("GOLDEN_CACHE_REFRESH", "0")
    return tmp_path


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("prefill", (False, True))
@pytest.mark.parametrize(
    "options,sq,sk",
    [
        ({}, 3, 5),
        ({"causal": True}, 5, 3),
        ({"window_size": (1, 0), "softmax_scale": -0.23}, 3, 5),
        ({"softcap": 0.7}, 3, 5),
        ({}, 3, 0),
        ({"causal": True}, 0, 5),
    ],
)
def test_cold_and_hit_equal_uncached_equations(cache, monkeypatch, dtype, prefill, options, sq, sk):
    data = inputs(dtype, prefill, sq, sk)
    expected = (
        *reference.mla_reference(**data, **options, upcast=True),
        *reference.mla_reference(**data, **options, upcast=False),
    )
    cold = reference.mla_reference_pair(**data, **options)
    monkeypatch.setattr(reference, "mla_reference", lambda **_: pytest.fail("hit recomputed"))
    hit = reference.mla_reference_pair(**data, **options)
    for want, first, second in zip(expected, cold, hit):
        assert torch.equal(want, first) and torch.equal(want, second)


@pytest.mark.parametrize(
    "change",
    (
        "input",
        "stride",
        "offset",
        "dtype",
        "shape",
        "scale",
        "causal",
        "window",
        "softcap",
        "source",
        "threads",
        "runtime",
    ),
)
def test_changed_identity_recomputes(cache, monkeypatch, change):
    data, options = inputs(), {}
    original = reference.mla_reference
    calls = []

    def equation(**kwargs):
        calls.append(kwargs["upcast"])
        return original(**kwargs)

    monkeypatch.setattr(reference, "mla_reference", equation)
    reference.mla_reference_pair(**data)
    if change == "input":
        data["v"].flatten()[0] += 1
    elif change in ("stride", "offset"):
        value = data["q"]
        stride = list(value.stride())
        if change == "stride":
            stride[-2] += 1
            stride[-3] = stride[-2] * value.shape[-2]
        view = torch.empty(value.numel() * 2).to(value.dtype).as_strided(value.shape, stride, 1)
        view.copy_(value)
        data["q"] = view
    elif change == "dtype":
        data = inputs(torch.bfloat16)
    elif change == "shape":
        data = inputs(sq=2)
    elif change in ("scale", "causal", "window", "softcap"):
        key, value = {
            "scale": ("softmax_scale", -0.1),
            "causal": ("causal", True),
            "window": ("window_size", (1, 0)),
            "softcap": ("softcap", 1.0),
        }[change]
        options[key] = value
    elif change == "source":
        monkeypatch.setattr(cache_module.golden_cache, "_source_digest", lambda _: "changed")
    elif change == "threads":
        monkeypatch.setenv("OMP_NUM_THREADS", "changed")
    else:
        monkeypatch.setattr(cache_module.platform, "machine", lambda: "changed")
    reference.mla_reference_pair(**data, **options)
    assert calls == [True, False, True, False]


def test_modes_corruption_errors_and_concurrent_hits(cache, monkeypatch):
    data = inputs()
    expected = reference.mla_reference_pair(**data)
    for path in cache.rglob("case_*.tar.gz"):
        path.write_bytes(b"damaged")
    with pytest.warns(RuntimeWarning, match="recomputing"):
        repaired = reference.mla_reference_pair(**data)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reference.mla_reference_pair(**data), range(4)))
    for result in [repaired, *results]:
        assert all(torch.equal(a, b) for a, b in zip(expected, result))

    def fail(**_):
        raise ValueError("equation failed")

    monkeypatch.setattr(reference, "mla_reference", fail)
    for mode, refresh in (("off", "0"), ("cache", "1")):
        monkeypatch.setenv("GOLDEN_CACHE_MODE", mode)
        monkeypatch.setenv("GOLDEN_CACHE_REFRESH", refresh)
        with pytest.raises(ValueError, match="equation failed"):
            reference.mla_reference_pair(**data)
    monkeypatch.setenv("GOLDEN_CACHE_REFRESH", "0")
    data["q"] = data["q"][..., ::2]
    with pytest.raises(ValueError, match="contiguous last dimension"):
        reference.mla_reference_pair(**data)
