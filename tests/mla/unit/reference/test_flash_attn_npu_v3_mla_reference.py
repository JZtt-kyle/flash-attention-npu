"""CPU acceptance for the MLA oracle; no FA3 extension or NPU is imported.

The scalar oracle and literal visibility matrices deliberately do not reuse
the mapping, masking, or tensor contractions from tests.mla.support.mla_ref.
"""

import math

import pytest
import torch

from tests.mla.support.mla_ref import (
    cann_to_fa3_inputs,
    fa3_to_cann_inputs,
    mla_mask,
    mla_reference,
    mla_reference_pair,
    normalize_fa3_window,
)


def make_components(representation, *, dtype=torch.float64, sq=3, sk=5):
    content_dim, kv_heads = (512, 1) if representation == "compressed" else (128, 2)
    generator = torch.Generator().manual_seed(1907)

    def rand(*shape):
        return (torch.randn(shape, dtype=torch.float64, generator=generator) / 4).to(dtype)

    query = rand(2, sq, 4, content_dim)
    key = rand(2, sk, kv_heads, content_dim)
    value = key if representation == "compressed" else rand(2, sk, kv_heads, 128)
    return dict(
        query=query,
        key=key,
        value=value,
        query_rope=rand(2, sq, 4, 64),
        key_rope=rand(2, sk, kv_heads, 64),
    )


def scalar_reference(query, key, value, query_rope, key_rope, *, scale, visible=None, softcap=0):
    """Small, explicit sum/exp oracle for CANN's two-component equation."""
    batch, sq, heads, _ = query.shape
    sk, kv_heads, dv = value.shape[1:]
    output = torch.zeros((batch, sq, heads, dv), dtype=torch.float64)
    lse = torch.full((batch, heads, sq), math.inf if sk == 0 else -math.inf)
    for b in range(batch):
        for i in range(sq):
            columns = [j for j in range(sk) if visible is None or visible[i][j]]
            for h in range(heads):
                kh = h // (heads // kv_heads)
                scores = []
                for j in columns:
                    content = math.fsum(
                        x * y for x, y in zip(query[b, i, h].tolist(), key[b, j, kh].tolist())
                    )
                    rope = math.fsum(
                        x * y
                        for x, y in zip(query_rope[b, i, h].tolist(), key_rope[b, j, kh].tolist())
                    )
                    score = (content + rope) * scale
                    scores.append(softcap * math.tanh(score / softcap) if softcap else score)
                if not scores:
                    continue
                maximum = max(scores)
                weights = [math.exp(s - maximum) for s in scores]
                denominator = math.fsum(weights)
                lse[b, h, i] = maximum + math.log(denominator)
                for d in range(dv):
                    output[b, i, h, d] = (
                        math.fsum(
                            weight * float(value[b, j, kh, d])
                            for weight, j in zip(weights, columns)
                        )
                        / denominator
                    )
    return output.to(query.dtype), lse


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [None, 0.0, 0.37])
def test_two_terms_scale_and_gqa(representation, dtype, scale):
    components = make_components(representation, dtype=dtype)
    dim = 576 if representation == "compressed" else 192
    expected_scale = dim**-0.5 if scale is None else scale
    fa3 = cann_to_fa3_inputs(**components, scale=expected_scale)
    # Exercise the FA3 default independently of the CANN mapping's scale.
    fa3["softmax_scale"] = scale
    actual, lse = mla_reference(**fa3)
    expected, expected_lse = scalar_reference(**components, scale=expected_scale)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-6, atol=1e-6)
    assert actual.dtype == dtype and lse.dtype == torch.float32
    assert actual.shape == (2, 3, 4, components["value"].shape[-1])
    assert lse.shape == (2, 4, 3)


def test_compressed_needs_both_dot_products():
    # Scores are [2*1 + 3*2, 2*3 + 3*1] = [8, 9], before scale.
    q = torch.zeros(1, 1, 1, 64, dtype=torch.float64)
    k = torch.zeros(1, 2, 1, 64, dtype=torch.float64)
    qv = torch.zeros(1, 1, 1, 512, dtype=torch.float64)
    v = torch.zeros(1, 2, 1, 512, dtype=torch.float64)
    q[..., 0], qv[..., 0] = 2, 3
    k[0, :, 0, 0], v[0, :, 0, 0] = torch.tensor([1, 3]), torch.tensor([2, 1])
    out, lse = mla_reference(q, k, v, qv, softmax_scale=0.5)
    p0 = 1 / (1 + math.exp(0.5))
    assert out[0, 0, 0, 0].item() == pytest.approx(1 + p0)
    assert lse.item() == pytest.approx(4 + math.log1p(math.exp(0.5)))
    assert torch.count_nonzero(out[..., 1:]) == 0
    rope_only, _ = mla_reference(q, k, v, torch.zeros_like(qv), softmax_scale=0.5)
    content_only, _ = mla_reference(torch.zeros_like(q), k, v, qv, softmax_scale=0.5)
    assert not torch.allclose(out, rope_only)
    assert not torch.allclose(out, content_only)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
def test_softcap_follows_sum_and_scale(representation):
    components = make_components(representation)
    actual, lse = mla_reference(**cann_to_fa3_inputs(**components, scale=2.0), softcap=0.7)
    expected, expected_lse = scalar_reference(**components, scale=2.0, softcap=0.7)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(lse, expected_lse)


@pytest.mark.parametrize(
    "representation,split_rope", [("compressed", True), ("prefill", True), ("prefill", False)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mapping_preserves_values_aliases_strides_and_scale(representation, split_rope, dtype):
    source = make_components(representation, dtype=dtype)
    fa3 = cann_to_fa3_inputs(**source, scale=0.23)
    mapped = fa3_to_cann_inputs(**fa3, split_rope=split_rope)
    assert mapped["value"] is source["value"]
    assert mapped["scale"] == 0.23
    if representation == "compressed":
        assert mapped["key"] is mapped["value"]
        assert fa3["q"] is source["query_rope"]
        assert fa3["k"] is source["key_rope"]
        assert fa3["qv"] is source["query"]
    else:
        assert fa3["qv"] is None
        assert not mapped["key"].is_set_to(mapped["value"])
        if split_rope:
            assert mapped["query"].stride(-2) == 192
            assert mapped["query_rope"].stride(-2) == 192
            assert mapped["query_rope"].storage_offset() == 128
            assert mapped["key_rope"].stride(-2) == 192
            for name in ("query", "key", "query_rope", "key_rope"):
                torch.testing.assert_close(mapped[name], source[name], rtol=0, atol=0)
        else:
            assert mapped["query"] is fa3["q"] and mapped["key"] is fa3["k"]
            assert mapped["query_rope"] is None and mapped["key_rope"] is None
    recovered = cann_to_fa3_inputs(**mapped)
    for name in ("q", "k", "v", "qv"):
        if fa3[name] is None:
            assert recovered[name] is None
        else:
            torch.testing.assert_close(recovered[name], fa3[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "representation,expected_scale", [("compressed", 1 / 24), ("prefill", 1 / math.sqrt(192))]
)
def test_mapping_passes_fa3_default_scale_to_cann(representation, expected_scale):
    inputs = cann_to_fa3_inputs(**make_components(representation), scale=1)
    inputs.pop("softmax_scale")
    assert fa3_to_cann_inputs(**inputs)["scale"] == pytest.approx(expected_scale)


@pytest.mark.parametrize("equal_values", [True, False])
def test_independent_compressed_content_key_is_rejected(equal_values):
    components = make_components("compressed")
    components["key"] = components["key"].clone()
    if not equal_values:
        components["key"].add_(1)
    with pytest.raises(ValueError, match="Independent 512-dimensional"):
        cann_to_fa3_inputs(**components, scale=1 / 24)


def test_shared_compressed_kv_allows_the_same_storage_view():
    components = make_components("compressed")
    components["key"] = components["value"].view_as(components["value"])
    assert components["key"] is not components["value"]
    fa3 = cann_to_fa3_inputs(**components, scale=1 / 24)
    assert fa3["v"] is components["value"]


MASK_CASES = [
    pytest.param(2, 5, True, (-1, -1), ["11110", "11111"], id="causal-short-q"),
    pytest.param(
        5, 2, True, (-1, -1), ["00", "00", "00", "10", "11"], id="causal-long-q-empty-rows"
    ),
    pytest.param(4, 4, False, (1, 0), ["1000", "1100", "0110", "0011"], id="local-left"),
    pytest.param(2, 5, False, (0, 0), ["00010", "00001"], id="local-unequal"),
    pytest.param(4, 5, False, (-1, 0), ["11000", "11100", "11110", "11111"], id="unbounded-left"),
    pytest.param(4, 5, False, (0, -1), ["01111", "00111", "00011", "00001"], id="unbounded-right"),
    pytest.param(3, 5, True, (1, 2), ["01100", "00110", "00011"], id="causal-local"),
    pytest.param(1, 3, True, (-1, -1), ["111"], id="single-query-causal"),
    pytest.param(1, 3, True, (0, 0), ["001"], id="single-query-local"),
    pytest.param(2, 3, False, (2, 1), ["111", "111"], id="normalized-full-window"),
]


@pytest.mark.parametrize("sq,sk,causal,window,rows", MASK_CASES)
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
def test_bottom_right_masks_and_empty_rows(representation, sq, sk, causal, window, rows):
    visible = torch.tensor([[c == "1" for c in row] for row in rows])
    torch.testing.assert_close(mla_mask(sq, sk, causal=causal, window_size=window), ~visible)
    components = make_components(representation, sq=sq, sk=sk)
    actual, lse = mla_reference(
        **cann_to_fa3_inputs(**components, scale=0.2), causal=causal, window_size=window
    )
    expected, expected_lse = scalar_reference(**components, scale=0.2, visible=visible.tolist())
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(lse, expected_lse)
    empty = ~visible.any(-1)
    assert torch.count_nonzero(actual[:, empty]) == 0
    assert torch.isneginf(lse[:, :, empty]).all()
    assert not torch.isnan(actual).any() and not torch.isnan(lse).any()


@pytest.mark.parametrize(
    "sq,sk,causal,window,expected",
    [
        (2, 5, False, (4, 1), (-1, -1)),
        (2, 5, False, (3, 0), (3, 0)),
        (2, 5, True, (4, 1), (-1, 0)),
        (1, 5, True, (-1, 0), (-1, -1)),
        (1, 5, True, (0, -1), (0, 0)),
        (5, 2, False, (1, 4), (-1, -1)),
    ],
)
def test_upstream_window_normalization(sq, sk, causal, window, expected):
    assert normalize_fa3_window(sq, sk, causal=causal, window_size=window) == expected


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("sq,sk", [(0, 3), (3, 0), (0, 0)])
def test_globally_empty_tensors(representation, sq, sk):
    components = make_components(representation, sq=sq, sk=sk)
    out, lse = mla_reference(**cann_to_fa3_inputs(**components, scale=0.2), causal=True)
    assert out.shape == (2, sq, 4, components["value"].shape[-1])
    assert lse.shape == (2, 4, sq)
    assert torch.count_nonzero(out) == 0
    assert torch.isposinf(lse).all()


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
def test_noncontiguous_outer_dimensions_and_offsets(representation):
    components = make_components(representation)
    fa3 = cann_to_fa3_inputs(**components, scale=0.2)
    for name in ("q", "k", "v", "qv"):
        tensor = fa3[name]
        if tensor is None:
            continue
        b, s, h, d = tensor.shape
        backing = torch.full((2 * b, 2 * s + 1, 2 * h, d), 9876.0, dtype=tensor.dtype)
        view = backing[::2, 1::2, ::2, :]
        view.copy_(tensor)
        fa3[name] = view
        assert not view.is_contiguous() and view.storage_offset() > 0
    mapped = fa3_to_cann_inputs(**fa3)
    if representation == "prefill":
        assert mapped["query"].stride() == mapped["query_rope"].stride() == fa3["q"].stride()
        assert mapped["key"].stride() == mapped["key_rope"].stride() == fa3["k"].stride()
    actual, lse = mla_reference(**fa3)
    expected, expected_lse = scalar_reference(**components, scale=0.2)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(lse, expected_lse)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
def test_low_precision_reference_pair(dtype, representation):
    components = make_components(representation, dtype=dtype, sq=5, sk=2)
    out_ref, lse_ref, out_pt, lse_pt = mla_reference_pair(
        **cann_to_fa3_inputs(**components, scale=0.2), causal=True
    )
    expected, expected_lse = scalar_reference(
        **components, scale=0.2, visible=[[0, 0], [0, 0], [0, 0], [1, 0], [1, 1]]
    )
    torch.testing.assert_close(out_ref, expected, rtol=0, atol=0)
    torch.testing.assert_close(lse_ref, expected_lse)
    # This bounds rounding in these small CPU examples, not an NPU acceptance
    # tolerance. M02 fixes device accuracy with the repository's dual oracle.
    torch.testing.assert_close(out_pt, expected, rtol=0, atol=torch.finfo(dtype).eps)
    torch.testing.assert_close(lse_pt, expected_lse, rtol=0, atol=torch.finfo(dtype).eps)
    assert out_pt.dtype == dtype and lse_pt.dtype == torch.float32
    assert torch.isneginf(lse_pt[:, :, :3]).all()
    assert torch.count_nonzero(out_pt[:, :3]) == 0


def test_invalid_or_unmodeled_representations_are_rejected():
    fa3 = cann_to_fa3_inputs(**make_components("compressed"), scale=1 / 24)
    with pytest.raises(ValueError, match="prefill requires"):
        mla_reference(**dict(fa3, qv=None))
    with pytest.raises(ValueError, match="qv must have"):
        mla_reference(**dict(fa3, qv=fa3["qv"][:, :1]))
    with pytest.raises(ValueError, match="one KV head"):
        mla_reference(**dict(fa3, k=fa3["k"].repeat(1, 1, 2, 1), v=fa3["v"].repeat(1, 1, 2, 1)))
    with pytest.raises(ValueError, match="contiguous last"):
        mla_reference(**dict(fa3, q=fa3["q"][..., ::2]))
    with pytest.raises(ValueError, match="same dtype"):
        mla_reference(**dict(fa3, q=fa3["q"].float()))
    with pytest.raises(ValueError, match="canonical BSND"):
        mla_reference(**dict(fa3, q=fa3["q"].flatten(0, 1)))
    with pytest.raises(ValueError, match="split RoPE"):
        fa3_to_cann_inputs(**fa3, split_rope=False)
    components = make_components("prefill")
    with pytest.raises(ValueError, match="CANN scale must be explicit"):
        cann_to_fa3_inputs(**components, scale=None)
    with pytest.raises(ValueError, match="both be present"):
        cann_to_fa3_inputs(**dict(components, query_rope=None), scale=0.2)
