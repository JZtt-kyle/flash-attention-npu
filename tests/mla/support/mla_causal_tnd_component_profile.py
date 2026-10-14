"""Representative TND production kernels versus identical non-FD source schedules."""

import ctypes as ct
import json
import os
from pathlib import Path
import statistics

import torch

from tests.mla.support.mla_causal_bsnd_used_profile import build_source
from tests.mla.support.mla_causal_tnd_profile import sample_kernels
from tests.mla.unit.causal_tnd.test_flash_attn_npu_v3_mla_causal_tnd_metadata import request_for
from tests.mla.e2e.causal_tnd.test_flash_attn_npu_v3_mla_causal_tnd_public import (
    prepare,
    spec,
    call,
    source_tile,
    effective,
    compare,
)
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_attention import make_case
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_dense import Schedule


CASES = [
    spec(q=(7,), k=(513,), uq=[5], uk=[257]),
    spec(q=(257,), k=(129,), uq=[129], uk=[65], h=3, stride="independent"),
    spec(q=(3,), k=(16384,), h=8),
]


def profile(probes, metadata_driver, used_oracle, output):
    output = Path(output)
    output.mkdir(parents=True)
    source = build_source(probes)
    cases = []
    for s in CASES:
        case = prepare(probes[0]["dtype"], s)
        tile = source_tile(probes, case, used_oracle)
        uq, uk = effective(s)
        src = make_case(
            probes,
            dict(shape=(1, s["q"][0], s["heads"], s["kv"][0]), lengths=uk, m=tile[0], n=tile[1]),
        )
        for name in case["raw"]:
            src["device"][name].copy_(case["raw"][name].unsqueeze(0))
        cores = int(src["details"][2])
        words = metadata_driver.evaluate(
            [request_for(s, 1 if probes[0]["dtype"] == torch.float16 else 2, cores)]
        )[0]
        assert words[5] == 0 and words[15:17] == tile
        schedule = Schedule()
        for field, value in zip(
            (
                "batch",
                "sq",
                "heads",
                "sk",
                "m_base",
                "s2_base",
                "cores",
                "block_dim",
                "ratio",
                "r1",
                "r2",
            ),
            (
                1,
                s["q"][0],
                s["heads"],
                s["kv"][0],
                words[15],
                words[16],
                words[20],
                words[21],
                2,
                words[367],
                words[369],
            ),
        ):
            setattr(schedule, field, value)
        for axis, start in (("b_end", 23), ("m_end", 49), ("s_end", 75)):
            getattr(schedule, axis)[:] = words[start : start + 26]
        case.update(
            source=src,
            schedule=schedule,
            used64=torch.tensor(uq, dtype=torch.int64, device="npu"),
            workspace=torch.empty(
                2 * schedule.block_dim * (6 * schedule.r1 + 8 * schedule.r2),
                dtype=torch.uint8,
                device="npu",
            ),
        )
        actual = call(case["args"])
        compare(probes, case, [actual], tile)
        cases.append(case)
    (output / "schedules.json").write_text(
        json.dumps(
            [
                dict(
                    spec=c["spec"],
                    words=metadata_driver.evaluate(
                        [
                            request_for(
                                c["spec"],
                                1 if probes[0]["dtype"] == torch.float16 else 2,
                                int(c["source"]["details"][2]),
                            )
                        ]
                    )[0],
                )
                for c in cases
            ],
            indent=2,
        )
        + "\n"
    )

    def run(index, name):
        case = cases[index]
        if name == "production":
            return call(case["args"])
        src = case["source"]
        d = src["device"]
        ptrs = [
            d["qv"],
            d["q"],
            d["v"],
            d["k"],
            src["desc"],
            *src["outputs"]["oracle"],
            case["workspace"],
        ]
        assert (
            source.launch_causal_used(
                torch.npu.current_stream().npu_stream,
                *(t.data_ptr() for t in ptrs),
                ct.byref(case["schedule"]),
                src["scale"],
                src["device_lengths"]["oracle"].data_ptr(),
                probes[0]["mask"].data_ptr(),
                case["used64"].data_ptr(),
            )
            == 0
        )
        return src["outputs"]["oracle"]

    names = ("production", "oracle")
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    try:
        for i, c in enumerate(cases):
            for _ in range(5):
                actual, expected = (run(i, n) for n in names)
                torch.npu.synchronize()
            q, k = (v[0] for v in effective(c["spec"]))
            torch.testing.assert_close(actual[0][:q], expected[0][0, :q], rtol=0, atol=0)
            empty = max(0, q - k)
            torch.testing.assert_close(
                actual[1][:, empty:q], expected[1][0, :, empty:q], rtol=0, atol=0
            )
        samples, paths = sample_kernels(
            run, len(cases), output, names, "mla_tnd_used_", "cann_causal_used"
        )
        report = dict(warmup=5, rounds=5, profiler_iterations=10, **paths, results=[])
        for i, c in enumerate(cases):
            values = {
                name: statistics.median(
                    r["kernel_us"] for r in samples if r["case"] == i and r["name"] == name
                )
                for name in names
            }
            report["results"].append(
                dict(spec=c["spec"], **values, ratio=values["production"] / values["oracle"])
            )
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        os.sched_setaffinity(0, affinity)
