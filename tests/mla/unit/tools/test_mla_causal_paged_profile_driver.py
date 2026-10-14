"""Native causal PA inputs preserve mask, cumulative TND Q and KV lengths."""

from tests.mla.paths import SUPPORT

import ast
import itertools

import pytest
import torch


def native_class():
    # Execute the exact adapter classes without importing NPU-only test modules.
    # Native execution and dispatch are separately exercised by profile suites.
    namespace = dict(torch=torch, itertools=itertools)
    for filename, name in (
        ("mla_paged_profile.py", "CannPagedCall"),
        ("mla_causal_paged_profile.py", "CausalPagedCall"),
    ):
        path = SUPPORT / filename
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["CausalPagedCall"]


@pytest.mark.parametrize("tnd", [False, True])
def test_native_mask_lengths_and_per_call_materialization(tnd):
    query = (5, 2) if tnd else (2, 3, 2)
    args = {
        name: torch.zeros((*query, size * 2))[..., ::2] for name, size in (("q", 64), ("qv", 512))
    }
    args.update(
        k=torch.zeros(3, 16, 1, 128)[..., ::2],
        v=torch.zeros(3, 16, 1, 1024)[..., ::2],
        page_table=torch.tensor([[2, 0], [1, 0]], dtype=torch.int32),
    )
    mask = torch.ones(8, 8, dtype=torch.bool).triu(1)
    call = native_class()(
        dict(
            args=args,
            config=dict(lengths=[17, 0], q=[3, 2] if tnd else [3, 3]),
            scale=0.125,
            mask=mask,
        )
    )
    assert call.kwargs["input_layout"] == ("TND" if tnd else "BSND")
    assert call.kwargs["actual_seq_kvlen"] == [17, 0]
    assert call.kwargs.get("actual_seq_qlen") == ([3, 5] if tnd else None)
    assert call.kwargs["atten_mask"] is mask
    assert call.kwargs["sparse_mode"] == 3 and call.kwargs["next_tokens"] == 0
    first = call.pack()
    args["q"].fill_(2)
    second = call.pack()
    assert first["q"].eq(0).all() and second["q"].eq(2).all()
    assert first["q"].data_ptr() != second["q"].data_ptr()
    assert all(first[n].is_contiguous() for n in ("q", "qv", "k", "v", "page_table"))
