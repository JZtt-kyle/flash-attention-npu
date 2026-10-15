"""Public PA split timing with native page/cache conversions inside every call."""

import json
import torch
from tests.common.compare import assert_fa_close
from tests.mla.support.public_cases.auto_split_paged_bsnd_public import (
    prepare as prepare_base,
    call,
    reference,
    metadata,
)
from tests.mla.support.mla_auto_split_paged_bsnd_cases import PERFORMANCE_CASES
from tests.mla.support.mla_paged_profile import CannPagedCall
from tests.mla.support.mla_ref import normalize_fa3_window


def prepare(dtype, s):
    c = prepare_base(dtype, s)
    c["spec"] = json.loads(json.dumps(s))
    c["raw"]["page_table"] = c["table"]
    return c


def generate(args):
    q, k, table = args["q"], args["k"], args["page_table"]
    s = dict(
        q=q.shape[1],
        heads=q.shape[2],
        capacity=table.shape[1] * k.shape[1],
        page=k.shape[1],
        causal=args["causal"],
        window=args["window_size"],
        splits=args["num_splits"],
    )
    return metadata(dict(args=args, spec=s))


class Native(CannPagedCall):
    def __init__(self, c):
        s = c["spec"]
        super().__init__(
            c["args"],
            s["lengths"],
            576**-0.5 if s["scale"] is None else s["scale"],
            [s["q"]] * len(s["lengths"]),
        )
        left, right = normalize_fa3_window(
            s["q"], s["capacity"], causal=s["causal"], window_size=s["window"]
        )
        kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
        self.kwargs["sparse_mode"] = 0
        if kind:
            self.kwargs.update(
                sparse_mode=3 if kind == 1 else 4,
                atten_mask=torch.ones((2048, 2048), dtype=torch.bool, device="npu").triu(1),
                pre_tokens=2**31 - 1 if kind == 1 else s["capacity"] - 1 if left < 0 else left,
                next_tokens=0 if kind == 1 else s["q"] - 1 if right < 0 else right,
            )

    def description(self):
        return {n: self.kwargs[n] for n in ("sparse_mode", "pre_tokens", "next_tokens")}


def accuracy(c, actual, native):
    refs = reference(c)
    for label, results in [("migrated", actual), ("native", native)]:
        for i, name in enumerate(("out", "lse")):
            got = results[i].cpu().clone()
            if label == "native" and i == 1:
                got[got.isposinf()] = -torch.inf
            assert_fa_close(got, refs[name + "_high"], refs[name + "_low"], name=label + "-" + name)
    return dict(migrated_math="passed", native_math="passed")


if __name__ == "__main__":
    import sys
    from tests.mla.support.mla_causal_mixed_profile import profile

    profile(
        *sys.argv[1:],
        specs=PERFORMANCE_CASES,
        prepare_case=prepare,
        call_forward=call,
        generate_metadata=generate,
        native_class=Native,
        check_accuracy=accuracy,
        mask="window",
        metadata_name="ComputeMlaAutoSplitPagedBsndMetadata",
        production_name=(
            "mla_auto_split_paged_bsnd_init_",
            "mla_auto_split_paged_bsnd_partial_",
            "mla_auto_split_paged_bsnd_merge_",
        ),
        automatic_split=True,
    )
