"""New layout/option dispatch against executed pinned partial and FD instructions."""

import json
import os
from pathlib import Path
import pytest
import torch
from tests.mla.support.components import softcap
from tests.mla.support.components.gqa_options_source import execute
from tests.mla.support.public_cases.gqa_options import make_case, metadata, invoke, reference
from tests.mla.support.mla_source_compare import assert_mla_source_close


@pytest.fixture(
    scope="module", params=[(r, d) for r in ("compressed", "prefill") for d in ("fp16", "bf16")]
)
def source(request, tmp_path_factory):
    representation, dtype = request.param
    directory = (
        Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path_factory.mktemp("gqa-source")))
        / "source"
    )
    return softcap.load(representation, dtype, directory)


@pytest.mark.mla_full_combinations
@pytest.mark.parametrize(
    "layout,pack,margin,splits,mask",
    [
        ("bsnd", False, 0, 3, "window"),
        ("tnd", True, 2, 0, "causal"),
        ("mixed", False, 1, 0, "window"),
        ("paged_bsnd", False, 1, 1, "causal"),
        ("paged_tnd", True, 1, 3, "window"),
        ("bsnd", True, 1, 1, "none"),
        pytest.param("bsnd", False, 1, 0, "window", id="bsnd-auto-head-distribution"),
        pytest.param("tnd", False, 1, 0, "window", id="tnd-auto-head-distribution"),
    ],
)
def test_fixed_source(source, layout, pack, margin, splits, mask):
    distribution = layout in ("bsnd", "tnd") and not pack and splits == 0
    c = make_case(
        layout,
        source["representation"],
        source["dtype"],
        pack=pack,
        margin=margin,
        splits=splits,
        mask=mask,
        q=(3, 1) if distribution else (7, 3),
        k=(4097, 1057) if distribution else (1057, 65),
        strided=True,
        heads=8 if distribution else 4,
        **(dict(window=(256, 2)) if distribution else {}),
    )
    record = metadata(c, geometry=layout == "tnd")
    actual = [x.cpu() for x in invoke(c, None if layout == "tnd" else record, entry="raw")]
    refs = reference(c, record)
    expected, raw = execute(source, c, record)
    checks = [
        assert_mla_source_close(
            actual[i], expected[i], refs[n + "_high"], refs[n + "_low"], name="gqa-source-" + n
        )
        for i, n in enumerate(("out", "lse"))
    ]
    if splits != 1:
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(actual[3], expected[3], rtol=1e-6, atol=1e-6)
    path = source["directory"] / f"gqa-{layout}-{pack}-{margin}-{splits}-{mask}.pt"
    torch.save(
        dict(inputs=c["raw"], actual=actual, source=expected, partials=raw, record=record.cpu()),
        path,
    )
    path.with_suffix(".json").write_text(json.dumps(checks, indent=2) + "\n")
