"""Actual public GQA addressing against fixed-source direct/FD instructions."""

import json
import os
from pathlib import Path
import pytest
import torch
from tests.mla.support.components import prefill_attention as probe
from tests.mla.support.public_cases.prefill_paged_bsnd import make_case, metadata, invoke, compare
from tests.mla.support.mla_source_compare import assert_mla_source_close


@pytest.fixture(scope="module", params=["fp16", "bf16"])
def native(request, tmp_path_factory):
    root = (
        Path(
            os.environ.get("MLA_VALIDATION_RESULT", tmp_path_factory.mktemp("public-prefill-paged"))
        )
        / "source"
    )
    return request.param, probe.load(request.param, root)


@pytest.mark.parametrize(
    "mask,splits",
    [("none", 1), ("causal", 1), ("window", 1), ("none", 3), ("causal", 3), ("window", 3)],
)
def test_prefill_source_instructions(native, mask, splits):
    dtype, (libs, directory) = native
    c = make_case(dtype, (2, 17, 8, 1057, 2), splits, mask, strided=True, lengths=[1057, 513])
    record = metadata(c)
    words = record.cpu().tolist()
    spec = dict(
        shape=c["shape"],
        splits=splits,
        m=words[13],
        n=words[14],
        strides=True,
        lengths=c["klens"],
        causal=c["options"]["causal"],
        window=c["options"]["window_size"],
    )
    case = probe.make_case(spec, dtype, libs["cores"])
    # Feed the independently gathered logical pages to the unchanged source
    # numerical fragment harness. Production still reads strided physical PA
    # storage; the pinned public FIA PA route is also checked in profile.
    for key, value in c["raw"].items():
        case["native"][key].copy_(value.npu())
    case["inputs"] = c["raw"]
    actual = [v.cpu() for v in invoke(c, record)]
    refs = compare(c, actual, record)
    raw = [[], [], []]
    for first in range(0, len(case["tasks"]), case["capacity"]):
        tasks = probe.wave(case, first)
        for t, value in zip(case["raw"]["oracle"], (0, 0, -torch.inf)):
            t.fill_(value)
        assert probe.launch_case(libs, case, "oracle", tasks, partial=splits != 1) == 0
        torch.npu.synchronize()
        if splits != 1:
            for dest, value in zip(raw, case["raw"]["oracle"]):
                dest.append(value[: case["params"].cores].cpu())
    if splits == 1:
        source = [v.cpu() for v in case["outputs"]["oracle"]]
        source[1][source[1].isposinf()] = -torch.inf
    else:
        raw = [torch.cat(x) for x in raw]
        merge = probe.prepare_merge(case, dict(oracle=raw, migrated=raw))
        assert probe.launch_merge(libs, merge, "oracle") == 0
        torch.npu.synchronize()
        b, sq, h, _, hk = c["shape"]
        g = h // hk
        out, lse = [v.cpu() for v in merge["outputs"]["oracle"]]
        source = [
            out.reshape(b, hk, sq, g, 128).permute(0, 2, 1, 3, 4).reshape(b, sq, h, 128),
            lse.reshape(b, h, sq),
        ]
        source[1][source[1] == -torch.finfo(torch.float32).max] = -torch.inf
        for i, ((virtual, _, split, _, _), row, visible) in enumerate(case["tasks"]):
            batch, head = divmod(virtual, hk)
            count = visible.shape[0]
            valid = visible.any(-1)
            got = (
                actual[2][split, batch, head * g : (head + 1) * g]
                .transpose(0, 1)
                .reshape(-1, 128)[row : row + count]
            )
            torch.testing.assert_close(got[valid], raw[0][i, :count][valid], rtol=0, atol=0)
            assert (got[~valid] == 0).all()
    checks = []
    for i, name in enumerate(("out", "lse")):
        checks.append(
            assert_mla_source_close(
                actual[i],
                source[i],
                refs[name + "_high"],
                refs[name + "_low"],
                name="public-prefill-paged-" + name,
            )
        )
    evidence = directory / f"public-{mask}-{splits}.pt"
    torch.save(
        dict(
            spec=spec, inputs=c["raw"], actual=actual, source=source, record=record.cpu(), raw=raw
        ),
        evidence,
    )
    evidence.with_suffix(".json").write_text(json.dumps(checks, indent=2) + "\n")
