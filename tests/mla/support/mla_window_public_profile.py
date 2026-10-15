"""Real band adapter and independent accuracy for the common public timing driver."""

import hashlib
import torch
from tests.mla.support.mla_cann_baseline import CannCall
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_ref import normalize_fa3_window


class WindowCannCall(CannCall):
    def __init__(self, inputs, *, scale=None, kv_lengths=None, spec):
        super().__init__(inputs, scale=scale, kv_lengths=kv_lengths)
        q, k = inputs["q"], inputs["k"]
        left, right = normalize_fa3_window(
            q.shape[1], k.shape[1], causal=spec.get("causal", False), window_size=spec["window"]
        )
        kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
        if kind:
            self.kwargs.update(
                sparse_mode=3 if kind == 1 else 4,
                atten_mask=torch.ones((2048, 2048), dtype=torch.bool, device=q.device).triu(1),
                pre_tokens=2**31 - 1 if kind == 1 else k.shape[1] - 1 if left < 0 else left,
                next_tokens=0 if kind == 1 else q.shape[1] - 1 if right < 0 else right,
            )

    def description(self):
        result = super().description()
        result.update(
            mask={0: "none", 3: "right-down causal", 4: "band"}[self.kwargs["sparse_mode"]],
            pre_tokens=self.kwargs["pre_tokens"],
            next_tokens=self.kwargs["next_tokens"],
        )
        return result


def check_window_accuracy(actual, source, cpu, shape, scale, lengths, spec):
    from tests.mla.support.mla_source_compare import assert_mla_source_close
    from tests.mla.support.components.window_attention import references, normalized_window

    batch, sq, heads, sk = shape
    window = normalized_window(spec)
    full_refs = references(
        dict(
            config=spec,
            shape=shape,
            inputs=cpu,
            lengths=lengths if lengths is not None else [sk] * batch,
            scale=scale,
        )
    )
    reports = []
    for b, length in enumerate(lengths if lengths is not None else [sk] * batch):
        refs = [ref[b : b + 1] for ref in full_refs]
        got = [actual[0].cpu()[b : b + 1], actual[1].cpu()[b : b + 1]]
        native = [source[0].cpu()[b : b + 1], source[1].cpu()[b : b + 1]]
        empty = (
            sq
            if length == 0
            else (
                0
                if window.kind == 0
                else max(0, min(sq, sq - length - (window.right if window.kind == 2 else 0)))
            )
        )
        assert (got[0][:, :empty] == 0).all() and got[1][:, :, :empty].isneginf().all()
        assert (native[0][:, :empty] == 0).all() and native[1][:, :, :empty].isposinf().all()
        native[1][:, :, :empty] = -torch.inf
        if length == 0:
            refs[1].fill_(-torch.inf)
            refs[3].fill_(-torch.inf)
        comparison = {}
        for i, name in enumerate(("out", "lse")):
            if torch.equal(got[i], native[i]):
                comparison[name] = assert_mla_source_close(
                    got[i], native[i], refs[i], refs[i + 2], name=name
                )
            else:
                # Native automatic FD can differ in arithmetic order. Its
                # dispatch is retained below, with independent math checks.
                assert_fa_close(got[i], refs[i], refs[i + 2], name=name)
                assert_fa_close(native[i], refs[i], refs[i + 2], name="native-" + name)
                comparison[name] = dict(source="different", math="passed")
        reports.append(comparison)
    identity = {
        name: dict(
            shape=list(t.shape),
            stride=list(t.stride()),
            offset=t.storage_offset(),
            sha256=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        )
        for name, t in cpu.items()
    }
    return dict(
        shape=shape,
        scale=scale,
        lengths=lengths,
        causal=spec.get("causal", False),
        window=spec["window"],
        inputs=identity,
        comparisons=reports,
    )
