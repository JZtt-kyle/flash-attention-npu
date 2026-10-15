"""New production dispatch compared with actual fixed-source partial/FD execution."""

import json
import torch
import pytest
from tests.mla.support.components import split_kv_merge as source
from tests.mla.support.public_cases.split_kv_bsnd_public import make_case, metadata, invoke
from tests.mla.support.mla_split_bsnd_reference import references
from tests.mla.support.mla_source_compare import assert_mla_source_close

merge_probes = source.merge_probes

SPECS = [
    dict(shape=(2, 65, 4, 1057), splits=8, mask="window", strided=True),
    dict(shape=(2, 7, 2, 3), splits=2, mask="causal"),
    dict(shape=(1, 3, 2, 1537), splits=8, mask="none"),
    dict(shape=(1, 1, 2, 16384), splits=2, mask="none"),
    dict(shape=(1, 3, 2, 513), splits=192, mask="window"),
]


@pytest.mark.parametrize("spec", SPECS)
def test_production_fixed_source(merge_probes, spec):
    p = merge_probes
    dtype = "fp16" if p["dtype"] == torch.float16 else "bf16"
    c = make_case(dtype, **spec)
    md = metadata(c).cpu()
    o = c["options"]
    native_spec = dict(
        id=f"production-{spec['mask']}-{spec['splits']}-{spec['shape'][1]}",
        shape=spec["shape"],
        m=int(md[15]),
        n=int(md[16]),
        splits=spec["splits"],
        lengths=c["klens"],
        scale=o["softmax_scale"],
        causal=o["causal"],
        window=o["window_size"],
    )
    native = source.make_case(p, native_spec)
    c["raw"] = native["inputs"]
    for name, value in c["raw"].items():
        c["t"][name].copy_(value)
    actual = [x.cpu() for x in invoke(c)]
    assert source.launch(p, native, "oracle") == 0
    torch.npu.synchronize()
    source.check_guards(native)
    expected = [t.cpu() for t in native["outputs"]["oracle"]]
    accum = torch.zeros_like(actual[2])
    alse = torch.full_like(actual[3], -torch.inf)
    for group in native["groups"]:
        params = group["params"]
        b = params.batch_index
        for row in range(params.rows):
            qi, hi = divmod(params.row_start + row, params.heads)
            if not group["valid"][:, row].any():
                expected[1][b, hi, qi] = -torch.inf
            for split in range(params.splits):
                if group["valid"][split, row]:
                    accum[split, b, hi, qi] = group["raw"][0][split, row]
                    alse[split, b, hi, qi] = (
                        group["raw"][1][split, row, 0].log() + group["raw"][2][split, row, 0]
                    )
    refs = references(
        c["raw"],
        c["qlens"],
        c["klens"],
        spec["splits"],
        native_spec["m"],
        native_spec["n"],
        o["causal"],
        o["window_size"],
        o["softmax_scale"],
    )
    comparisons = []
    for i, name in enumerate(("out", "lse")):
        comparisons.append(
            assert_mla_source_close(
                actual[i],
                expected[i],
                refs[name + "_high"],
                refs[name + "_low"],
                name="production-" + name,
            )
        )
    torch.testing.assert_close(actual[2], accum, rtol=0, atol=0)
    torch.testing.assert_close(actual[3], alse, rtol=1e-6, atol=1e-6)
    path = p["directory"] / (native_spec["id"] + ".pt")
    torch.save(
        dict(
            raw=c["raw"],
            actual=actual,
            source=expected,
            accum=accum,
            accum_lse=alse,
            references=refs,
        ),
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(comparisons, indent=2) + "\n")
