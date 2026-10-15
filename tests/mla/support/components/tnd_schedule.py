"""Shared helpers extracted from tests/mla/unit/tnd/test_flash_attn_npu_v3_mla_tnd_schedule.py.

Compact TND CPU parser/schedule, independently checked against pinned CANN.

No public/device TND entry is enabled. MLA_TND_SCHEDULE_ARTIFACT_DIR retains
inputs, outputs, source blobs and manifests; MLA_SCHEDULE_SANITIZE=1 instruments
both drivers. The source executable never includes/links the migrated parser.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import pytest
from tests.mla.support.mla_cpu_driver import evaluate_cpu
from tests.mla.support.components.schedule import (
    CANN_COMMIT,
    FIA,
    FUNCTIONS,
    MLA,
    ROOT,
    SOURCE_FILES,
    case,
    source_function,
)

PARSER = "attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp"


@pytest.fixture(scope="session")
def tnd_artifacts(tmp_path_factory):
    target = os.environ.get("MLA_TND_SCHEDULE_ARTIFACT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-tnd-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def compile_driver(
    directory, oracle, *, mixed=False, used=False, paged=False, migrated_split=False
):
    name = "oracle" if oracle else "schedule"
    if mixed:
        name += "-mixed"
    if used:
        name += "-used"
    if paged:
        name += "-paged"
    if migrated_split:
        name += "-migrated-split"
    binary = directory / name
    command = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O2",
        "-g",
        "-D_GLIBCXX_ASSERTIONS",
        "-DBSND_KV_LENGTHS",
        "-DTND_LENGTHS",
    ]
    if mixed:
        command += ["-DTND_BSND_KV"]
    if used:
        command += ["-DUSED_LENGTHS"]
    if paged:
        command += ["-DPAGED_KV"]
    if os.environ.get("MLA_SCHEDULE_SANITIZE") == "1":
        command += [
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
        ]
    if oracle:
        command += ["-DSOURCE_ORACLE", "-I" + str(directory)]
        if migrated_split:
            command += [
                "-DMLA_MIGRATED_SPLIT",
                "-I" + str(MLA),
                str(MLA / "detail/cann_split_core.cpp"),
            ]
        else:
            command += [str(directory / "split_core.cpp")]
    else:
        command += ["-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(MLA)]
        command += [
            str(MLA / f)
            for f in ("mla_tnd_schedule.cpp", "mla_schedule.cpp", "detail/cann_split_core.cpp")
        ]
    command += [str(ROOT / "tests/mla/support/mla_schedule_driver.cpp"), "-o", str(binary)]
    (directory / f"{name}-compile.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (directory / f"{name}-compile.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.fixture(scope="session")
def tnd_oracle(tnd_artifacts):
    return build_oracle(tnd_artifacts)


def build_oracle(tnd_artifacts, *, mixed=False, used=False, paged=False):
    source_root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    blobs = {
        name: subprocess.check_output(
            ["git", "-C", str(source_root), "show", CANN_COMMIT + ":" + name], timeout=30
        )
        for name in [*SOURCE_FILES, PARSER]
    }
    for name, blob in blobs.items():
        (tnd_artifacts / Path(name).name).write_bytes(blob)
    functions = [*FUNCTIONS, "DealSameSeqEachBatch", "CreateSplitInput"]
    source = blobs[FIA + "fia_tiling_nonquant_mla.cpp"].decode()
    bodies = ["template <typename T>\n" + source_function(source, "inline auto Align(")]
    bodies += [source_function(source, "FiaTilingNonQuantMla::" + f + "(") for f in functions]
    (tnd_artifacts / "source_functions.inc").write_text("\n".join(bodies))
    parser_functions = ["ProcessKVActualSeqLoop"]
    if paged:
        parser_functions += ["GetS2SizeForPageAttention"]
    (tnd_artifacts / "source_parser.inc").write_text(
        "\n".join(
            source_function(blobs[PARSER].decode(), "FiaInfoParser::" + name + "(")
            for name in parser_functions
        )
    )
    (tnd_artifacts / "log").mkdir(exist_ok=True)
    (tnd_artifacts / "log/log.h").write_text(
        "#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n"
    )
    manifest = dict(
        commit=CANN_COMMIT,
        kv_layout="per input" if used else "BSND" if mixed else "TND",
        used_adapter=used,
        paged_adapter=paged,
        functions=functions,
        parser_functions=parser_functions,
        split_core_functions=["GetS1SeqSize", "GetS2SeqSize", "SplitCore"],
        files={name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
    )
    (tnd_artifacts / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return compile_driver(tnd_artifacts, True, mixed=mixed, used=used, paged=paged)


def tnd_case(
    qlens=(3, 0, 1),
    klens=(17, 0, 33),
    *,
    cq=None,
    ck=None,
    qcount=None,
    kcount=None,
    has_q=True,
    has_k=True,
    total_q=None,
    total_k=None,
    **shape,
):
    cq = list(itertools.accumulate(qlens, initial=0)) if cq is None else list(cq)
    ck = list(itertools.accumulate(klens, initial=0)) if ck is None else list(ck)
    defaults = dict(batch=len(qlens), sq=max(qlens, default=0), sk=max(klens, default=0))
    defaults.update(shape)
    return case(**defaults) + [
        int(has_k),
        len(ck) if kcount is None else kcount,
        len(ck),
        *ck,
        cq[-1] if total_q is None else total_q,
        ck[-1] if total_k is None else total_k,
        int(has_q),
        len(cq) if qcount is None else qcount,
        len(cq),
        *cq,
    ]


def prefixes(item):
    ck = item[16 : 16 + item[15]]
    start = 16 + item[15]
    cq = item[start + 5 :]
    return cq, ck


def evaluate(binary, cases, *, native=False, parse=False):
    outputs = evaluate_cpu(
        binary,
        cases,
        options={
            "MLA_ORACLE_AUTOMATIC_FD": "1" if native else None,
            "MLA_TND_PARSE_ONLY": "1" if parse else None,
        },
    )
    assert len(outputs) == len(cases)
    return outputs


def expected_spans(item):
    cq, ck = prefixes(item)
    spans = [(q, k, qend - q, kend - k) for q, k, qend, kend in zip(cq, ck, cq[1:], ck[1:])]
    return [
        max((s[2] for s in spans), default=0),
        max((s[3] for s in spans), default=0),
        *itertools.chain.from_iterable(spans),
    ]


def check_tasks(item, result):
    # Independent scalar enumeration, with no SplitCore/parser helpers.
    cq, ck = prefixes(item)
    mbase, nbase, cores = result[1:4]
    coords, costs = [], []
    for b in range(item[0]):
        rows = (cq[b + 1] - cq[b]) * item[2]
        kv = ck[b + 1] - ck[b]
        for m, start in enumerate(range(0, rows, mbase)):
            coords.append((b, m))
            costs.append(
                sum(
                    6 * ((min(mbase, rows - start) + 15) // 16)
                    + 10 * ((min(nbase, kv - k) + 63) // 64)
                    for k in range(0, kv, nbase)
                )
            )
    ends = [(result[18 + i], result[44 + i]) for i in range(cores)]
    assert all(a < b for a, b in zip([(0, 0), *ends[:-1]], ends))
    assert ends[-1] <= (item[0], 0)
    assert result[5:8] == [0, 0, 0] and result[70:96] == [0] * 26
    begin, loads, visited = (0, 0), [], []
    for end in ends:
        selected = [i for i, coord in enumerate(coords) if begin <= coord < end]
        visited += selected
        loads.append(sum(costs[i] for i in selected))
        begin = end
    assert len(visited) == len(set(visited))
    assert {i for i in visited if costs[i]} == {i for i, cost in enumerate(costs) if cost}
    assert result[4] == max(loads)
    assert result[356:] == expected_spans(item)
