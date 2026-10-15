"""Pinned PA FD partial and merge on physical pages; production and dual math."""

import ctypes as ct
import json
from pathlib import Path
import pytest
import torch
from tests.mla.unit.split_kv_merge import test_merge as source
from tests.mla.unit.split_kv_attention import test_attention as partial
from tests.mla.support.mla_split_kv_attention_reference import fragments
from tests.mla.support.mla_split_paged_bsnd_source_build import build_paged_partial, sha256
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.e2e.split_paged_bsnd.test_public import prepare, metadata, call, reference, compare
from tests.mla.support.mla_split_paged_bsnd_cases import spec

merge_probes = source.merge_probes


@pytest.fixture(scope="module")
def paged_source(merge_probes):
    library = build_paged_partial(merge_probes["directory"])
    lib = ct.CDLL(str(library))
    lib.launch_paged_partial.argtypes = (
        [ct.c_void_p] * 10 + [ct.c_float] + [ct.c_void_p] * 3 + [ct.c_uint32] * 2 + [ct.c_void_p]
    )
    lib.launch_paged_partial.restype = ct.c_int
    manifest = json.loads((library.parent / "paged-build-manifest.json").read_text())
    loaded = dict(library=str(Path(lib._name).resolve()), sha256=sha256(Path(lib._name)))
    assert loaded == {key: manifest[key] for key in loaded}
    (library.parent / "paged-loaded.json").write_text(json.dumps(loaded, indent=2) + "\n")
    yield merge_probes, lib
    assert sha256(library) == loaded["sha256"]
    assert all(sha256(Path(name)) == digest for name, digest in manifest["inputs"].items())


SPECS = [
    spec(page=16, q=7, lengths=(3, 33), heads=2, splits=2, causal=True, window=(-1, -1)),
    spec(
        page=112,
        q=65,
        lengths=(1025, 511),
        heads=4,
        splits=8,
        window=(530, 1),
        used=[63, 1],
        stride=True,
    ),
    spec(page=512, q=3, lengths=(1025,), heads=2, splits=2, window=(-1, -1)),
    spec(page=528, q=257, lengths=(1057,), heads=1, splits=192, window=(513, 7)),
    spec(page=1024, q=2, lengths=(16384,), heads=2, splits=2, window=(-1, -1)),
    spec(page=112, q=7, lengths=(449,), heads=2, splits=256, window=(2, 1)),
]


@pytest.mark.parametrize("s", SPECS)
def test_fixed_paged_source(paged_source, s):
    p, lib = paged_source
    c = prepare(p["dtype"], s)
    md = metadata(c).cpu()
    m, n = int(md[15]), int(md[16])
    qs = [s["q"]] * len(s["lengths"]) if s["used"] is None else s["used"]
    expected = [
        torch.zeros((len(qs), s["q"], s["heads"], 512), dtype=p["dtype"]),
        torch.full((len(qs), s["heads"], s["q"]), -torch.inf),
    ]
    acc = torch.zeros((s["splits"], len(qs), s["heads"], s["q"], 512))
    alse = torch.full(acc.shape[:-1], -torch.inf)
    all_raw = []
    for b, (q, length) in enumerate(zip(qs, s["lengths"])):
        ns = dict(
            shape=(1, q, s["heads"], s["capacity"]),
            lengths=[length],
            m=m,
            n=n,
            splits=s["splits"],
            scale=576**-0.5,
            causal=s["causal"],
            window=s["window"],
        )
        tasks = list(fragments(ns, m, n))
        chunks = [[], [], []]
        first = 0
        qr = c["raw"]["q"][b : b + 1, :q].contiguous().npu()
        qc = c["raw"]["qv"][b : b + 1, :q].contiguous().npu()
        kr = c["raw"]["k"].contiguous().npu()
        kv = c["raw"]["v"].contiguous().npu()
        table = c["table"][b : b + 1].contiguous().npu()
        desc = torch.tensor(
            [24, 1 << 32, 0xFFFFFFFF, kv.data_ptr()], dtype=torch.int64, device="npu"
        )
        while first < len(tasks):
            item = partial.make_case(p["partial"], ns, first)
            pointers = [
                qc,
                qr,
                kv,
                kr,
                desc,
                *item["outputs"]["oracle"],
                item["workspace"]["oracle"],
            ]
            assert (
                lib.launch_paged_partial(
                    torch.npu.current_stream().npu_stream,
                    *(x.data_ptr() for x in pointers),
                    ct.byref(item["schedule"]),
                    item["scale"],
                    item["device_lengths"]["oracle"].data_ptr(),
                    item["mask"].data_ptr(),
                    ct.byref(item["oracle_partial"]),
                    s["page"],
                    s["capacity"] // s["page"],
                    table.data_ptr(),
                )
                == 0
            )
            torch.npu.synchronize()
            for values, raw in zip(chunks, partial.raw_results(item, "oracle")):
                values.append(raw)
            ws = item["ws_storage"]["oracle"]
            assert (ws[:512].cpu() == 205).all() and (ws[-512:].cpu() == 205).all()
            first += item["schedule"].cores
        raw = [torch.cat(v) for v in chunks]
        all_raw.append(raw)
        # Both dictionary entries deliberately refer to the same source data:
        # this harness invokes original FD, not the migrated partial probe.
        native = source.allocate_case(p, ns, tasks, dict(oracle=raw, migrated=raw))
        if s["splits"] <= 192:
            assert source.launch(p, native, "oracle") == 0
            torch.npu.synchronize()
            source.check_guards(native)
            expected[0][b, :q] = native["outputs"]["oracle"][0].cpu()[0]
            expected[1][b, :, :q] = native["outputs"]["oracle"][1].cpu()[0]
        else:
            assert source.launch(p, native, "oracle") == -2  # fixed source FD UB limit
        for group in native["groups"]:
            params = group["params"]
            for row in range(params.rows):
                qi, hi = divmod(params.row_start + row, params.heads)
                if not group["valid"][:, row].any():
                    expected[1][b, hi, qi] = -torch.inf
                for split in range(s["splits"]):
                    if group["valid"][split, row]:
                        acc[split, b, hi, qi] = group["raw"][0][split, row]
                        alse[split, b, hi, qi] = (
                            group["raw"][1][split, row, 0].log() + group["raw"][2][split, row, 0]
                        )
    actual = [x.cpu() for x in call(c["args"], "op")]
    refs = reference(c)
    if s["splits"] > 192:
        compare(c, actual, refs)
    reports = []
    for b, q in enumerate(qs):
        if s["splits"] <= 192:
            for i, name in enumerate(("out", "lse")):
                idx = (b, slice(q)) if i == 0 else (b, slice(None), slice(q))
                reports.append(
                    assert_mla_source_close(
                        actual[i][idx],
                        expected[i][idx],
                        refs[name + "_high"][idx],
                        refs[name + "_low"][idx],
                        name="paged-" + name,
                    )
                )
        torch.testing.assert_close(actual[2][:, b, :, :q], acc[:, b, :, :q], rtol=0, atol=0)
        torch.testing.assert_close(actual[3][:, b, :, :q], alse[:, b, :, :q], rtol=1e-6, atol=1e-6)
    path = p["directory"] / f"pa-{s['page']}-{s['splits']}.pt"
    torch.save(
        dict(
            spec=s,
            inputs=c["raw"],
            table=c["table"],
            partials=all_raw,
            actual=actual,
            source=expected,
            accum=acc,
            accum_lse=alse,
            reference=refs,
        ),
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(dict(comparisons=reports, source_merge_supported=s["splits"] <= 192), indent=2)
        + "\n"
    )
