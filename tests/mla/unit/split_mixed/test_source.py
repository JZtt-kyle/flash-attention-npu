"""Actual pinned partial/FD execution mapped into independently allocated mixed spans."""

import json
import pytest
import torch
from tests.mla.support.components import split_kv_merge_merge as source
from tests.mla.support.public_cases.split_mixed_public import prepare, metadata, call, reference
from tests.mla.support.mla_split_mixed_cases import spec
from tests.mla.support.mla_source_compare import assert_mla_source_close

merge_probes = source.merge_probes

SPECS = [
    spec(
        q=(7, 0, 65),
        k=(1025, 0, 511),
        capacity=1057,
        uq=[5, 0, 63],
        h=4,
        window=(530, 1),
        stride="independent",
    ),
    spec(q=(7, 3), k=(3, 0), capacity=17, h=2, causal=True, window=(-1, -1), splits=2),
    spec(q=(3, 1), k=(1537, 17), capacity=1537, h=2, window=(-1, -1), splits=8),
    spec(q=(1,), k=(16384,), capacity=16384, h=2, window=(-1, -1), splits=2),
    spec(q=(3, 1), k=(513, 17), capacity=513, h=2, window=(2, 1), splits=192),
]


@pytest.mark.parametrize("s", SPECS)
def test_fixed_source(merge_probes, s):
    p = merge_probes
    c = prepare(p["dtype"], s)
    md = metadata(c).cpu()
    m, n = int(md[15]), int(md[16])
    qs = s["q"] if s["uq"] is None else s["uq"]
    ks = s["kv"] if s["uk"] is None else s["uk"]
    expected = [
        torch.zeros((sum(s["q"]), s["heads"], 512), dtype=p["dtype"]),
        torch.full((s["heads"], sum(s["q"])), -torch.inf),
    ]
    acc = torch.zeros((s["splits"], s["heads"], sum(s["q"]), 512))
    alse = torch.full(acc.shape[:-1], -torch.inf)
    qstart = kstart = 0
    for b, (q, k) in enumerate(zip(qs, ks)):
        if q:
            native_spec = dict(
                id=f"mixed-{s['splits']}-{sum(s['kv'])}-{b}",
                shape=(1, q, s["heads"], s["kv"][b]),
                lengths=[k],
                m=m,
                n=n,
                splits=s["splits"],
                scale=s.get("scale") or 576**-0.5,
                causal=s.get("causal", False),
                window=s["window"],
            )
            native = source.make_case(p, native_spec)
            for name, t in native["inputs"].items():
                start = qstart if name in ("q", "qv") else kstart
                c["raw"][name][start : start + t.shape[1]] = t[0]
            assert source.launch(p, native, "oracle") == 0
            torch.npu.synchronize()
            source.check_guards(native)
            out, lse = [t.cpu() for t in native["outputs"]["oracle"]]
            expected[0][qstart : qstart + q] = out[0]
            expected[1][:, qstart : qstart + q] = lse[0]
            for group in native["groups"]:
                params = group["params"]
                for row in range(params.rows):
                    qi, hi = divmod(params.row_start + row, params.heads)
                    if not group["valid"][:, row].any():
                        expected[1][hi, qstart + qi] = -torch.inf
                    for split in range(params.splits):
                        if group["valid"][split, row]:
                            acc[split, hi, qstart + qi] = group["raw"][0][split, row]
                            alse[split, hi, qstart + qi] = (
                                group["raw"][1][split, row, 0].log()
                                + group["raw"][2][split, row, 0]
                            )
        qstart += s["q"][b]
        kstart += s["kv"][b]
    for name, t in c["raw"].items():
        c["args"][name].copy_(t.reshape(c["args"][name].shape))
    actual = [t.cpu() for t in call(c["args"], "op")]
    refs = reference(c, (m, n))
    reports = []
    start = 0
    for allocation, q in zip(s["q"], qs):
        for i, name in enumerate(("out", "lse")):
            idx = (slice(start, start + q),) if i == 0 else (slice(None), slice(start, start + q))
            reports.append(
                assert_mla_source_close(
                    actual[i][idx],
                    expected[i][idx],
                    refs[name + "_high"][idx],
                    refs[name + "_low"][idx],
                    name="mixed-source-" + name,
                )
            )
        torch.testing.assert_close(
            actual[2][:, :, start : start + q], acc[:, :, start : start + q], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual[3][:, :, start : start + q], alse[:, :, start : start + q], rtol=1e-6, atol=1e-6
        )
        start += allocation
    path = p["directory"] / f"mixed-{s['splits']}-{sum(s['kv'])}.pt"
    torch.save(
        dict(
            spec=s,
            raw=c["raw"],
            actual=actual,
            source=expected,
            accum=acc,
            accum_lse=alse,
            reference=refs,
        ),
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(reports, indent=2) + "\n")
