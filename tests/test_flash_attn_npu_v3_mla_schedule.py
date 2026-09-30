"""CPU differential checks against compiled functions from the pinned CANN Git blobs.

MLA_CANN_SOURCE_ROOT selects a read-only ops-transformer checkout; only git show is
used. The oracle cannot include/link the migrated schedule. No NPU or torch needed.
Compare SplitCore vectors directly: upstream SetSplitOutput reads usedCoreNum*2
FD endpoints even for C1V1, where the vectors only have aicNum entries.
MLA_SCHEDULE_ARTIFACT_DIR keeps binaries, source manifests and case digests.
MLA_SCHEDULE_SANITIZE=1 additionally enables ASan/UBSan for both executables.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
MLA = ROOT / "csrc/ascend910/flash_attn_npu_3/mla"
CANN_COMMIT = "ca18aee610729ca5270a44e2d84c751bc687fbf0"
FIA = "attention/fused_infer_attention_score/op_host/arch22/"
SOURCE_FILES = [
    "attention/common/op_host/split_core.h",
    "attention/common/op_host/split_core.cpp",
    FIA + "fia_tiling_nonquant_mla.cpp",
    FIA + "fia_tiling_nonquant_mla_arch22.h",
]
FUNCTIONS = [
    "CalcInnerSize",
    "CalcMBaseSize",
    "CalcMmResSize",
    "FillTilingWorkspaceParams",
    "CalcNormalWorkspaceSize",
    "CalcFlashDecodeParamNums",
    "CalcFlashDecodeWorkspace",
    "CalcWorkspaceSize",
    "CalcNumBlocks",
    "CalcScheduleMode",
]


def source_function(source, token):
    """Extract a complete function definition, preserving its body byte for byte."""
    start = source.rfind("\n", 0, source.index(token)) + 1
    brace = source.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + "\n"


def run(command, **kwargs):
    return subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=120, **kwargs
    )


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    target = os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-schedule")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def compile_driver(directory, oracle):
    binary = directory / ("oracle" if oracle else "schedule")
    command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-g", "-D_GLIBCXX_ASSERTIONS"]
    if os.environ.get("MLA_SCHEDULE_SANITIZE") == "1":
        command += [
            "-fsanitize=address,undefined",
            "-fno-omit-frame-pointer",
            "-fno-sanitize-recover=all",
        ]
    if oracle:
        command += ["-DSOURCE_ORACLE", "-I" + str(directory), str(directory / "split_core.cpp")]
    else:
        command += ["-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(MLA)]
        command += [str(MLA / "mla_schedule.cpp"), str(MLA / "detail/cann_split_core.cpp")]
    command += [str(ROOT / "tests/mla_schedule_driver.cpp"), "-o", str(binary)]
    (directory / (binary.name + "-compile.json")).write_text(json.dumps(command, indent=2) + "\n")
    result = run(command)
    (directory / (binary.name + "-compile.log")).write_text(result.stdout + result.stderr)
    return binary


@pytest.fixture(scope="session")
def scheduler(artifact_dir):
    return compile_driver(artifact_dir, oracle=False)


@pytest.fixture(scope="session")
def oracle(artifact_dir):
    source_root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    if not source_root.is_dir():
        pytest.skip("set MLA_CANN_SOURCE_ROOT to a checkout containing the pinned CANN commit")
    blobs = {}
    for name in SOURCE_FILES:
        blobs[name] = subprocess.check_output(
            ["git", "-C", str(source_root), "show", CANN_COMMIT + ":" + name], timeout=30
        )
    manifest = {
        "commit": CANN_COMMIT,
        "functions": FUNCTIONS,
        "files": {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()},
    }
    (artifact_dir / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, blob in blobs.items():
        (artifact_dir / Path(name).name).write_bytes(blob)
    source = blobs[FIA + "fia_tiling_nonquant_mla.cpp"].decode()
    header = blobs[FIA + "fia_tiling_nonquant_mla_arch22.h"].decode()
    for default in ["mBaseSize_ = 256", "mFdBaseSize_ = 8", "splitKVFlag_ = false"]:
        assert default in header  # Match the isolated class shim's initial state.
    bodies = [source_function(source, "inline auto Align(")]
    bodies[0] = "template <typename T>\n" + bodies[0]
    bodies += [source_function(source, "FiaTilingNonQuantMla::" + name + "(") for name in FUNCTIONS]
    (artifact_dir / "source_functions.inc").write_text("\n".join(bodies))
    log_dir = artifact_dir / "log"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "log.h").write_text(
        "#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n"
    )
    return compile_driver(artifact_dir, oracle=True)


def case(batch=1, sq=1, heads=128, sk=512, aic=20, ratio=2, **overrides):
    values = dict(
        batch=batch,
        sq=sq,
        heads=heads,
        sk=sk,
        aic=aic,
        aiv=aic * ratio,
        l1=516096,
        ub=178176 if ratio == 1 else 174080,
        l0a=65536,
        l0b=65536,
        l0c=131072,
        libapi=16777216,
        block_dim=0,
    )
    values.update(overrides)
    return list(values.values())


def evaluate(binary, cases):
    text = "".join(" ".join(map(str, item)) + "\n" for item in cases)
    result = run([str(binary)], input=text)
    assert not result.stderr, result.stderr
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(outputs) == len(cases)
    return outputs


def compare(scheduler, oracle, cases, artifact_dir, name):
    actual = evaluate(scheduler, cases)
    expected = evaluate(oracle, cases)
    for shape, got, want in zip(cases, actual, expected):
        assert got == want, (shape, got, want)
        assert got[0] == 0
        assert got[5:8] == [0, 0, 0]  # No FD, including its two auxiliary counts.
        assert len(got) == 18 + 9 * 26 + 2 * 52
    report = dict(
        cases=len(cases),
        input_sha256=hashlib.sha256(json.dumps(cases).encode()).hexdigest(),
        output_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest(),
        all_fields_equal=True,
    )
    (artifact_dir / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    return actual


@pytest.mark.parametrize("aic,ratio", itertools.product(range(1, 27), (1, 2)))
def test_pinned_schedule_boundaries(scheduler, oracle, artifact_dir, aic, ratio):
    batches = sorted({1, max(1, aic - 1), aic, aic + 1, 2 * aic - 1, 2 * aic, 2 * aic + 1})
    rows = (1, 15, 16, 17, 127, 128, 129, 255, 256)
    kv_lengths = (1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512)
    cases = [
        case(b, 1, m, n, aic, ratio) for b, m, n in itertools.product(batches, rows, kv_lengths)
    ]
    compare(scheduler, oracle, cases, artifact_dir, f"boundary-{aic}-{ratio}")


@pytest.mark.parametrize("ratio", (1, 2))
def test_multi_query_and_reuse(scheduler, oracle, artifact_dir, ratio):
    cases = [
        case(b, sq, m // sq, n, c, ratio)
        for c, b, m, n in itertools.product(
            (2, 20, 24, 26), (1, 3, 17, 41), (16, 128, 256), (1, 127, 512)
        )
        for sq in (2, 4, m)
    ]
    # A process handles many changing shapes, then repeats them in reverse order.
    results = compare(scheduler, oracle, cases + cases[::-1], artifact_dir, f"multi-query-{ratio}")
    assert results[: len(cases)] == results[len(cases) :][::-1]


def test_platform_policy_and_workspace_fields(scheduler, oracle, artifact_dir):
    cases = [case(batch=3, aic=20, block_dim=20), case(batch=10, heads=256)]
    results = compare(scheduler, oracle, cases, artifact_dir, "platform-and-workspace")
    assert results[0][16] == 20 and results[0][3] < 20  # Real blockDim is supplied, not guessed.
    assert results[1][1] == 128  # Source M adjustment: 10 batches, 20 cores, 256 rows.
    assert all(item[11] > 0 and item[12] > 0 for item in results)  # FD size fields stay nonzero.
    budget = results[0][13]
    max_u64 = 2**64 - 1
    assert evaluate(scheduler, [case(batch=3, libapi=max_u64 - budget)])[0][15] == max_u64
    assert evaluate(scheduler, [case(batch=3, libapi=max_u64 - budget + 1)]) == [[5]]


def test_equal_cost_keeps_first_plan(scheduler, oracle, artifact_dir):
    result = compare(scheduler, oracle, [case(batch=4, aic=3)], artifact_dir, "equal-cost")[0]
    # Two and three cores both have a slow core with two blocks. Source retains two.
    assert result[3] == 2
    assert result[18:20] == [2, 4]


@pytest.mark.parametrize("ratio", (1, 2))
def test_s1g_multiblock_source_partitions(scheduler, oracle, artifact_dir, ratio):
    cases = [
        case(b, sq, h, sk, aic, ratio)
        for aic in (1, 2, 3, 20, 24, 26)
        for b in sorted({1, aic - 1 or 1, aic, aic + 1, 2 * aic + 1})
        for sq, h in ((257, 1), (7, 73), (4, 128), (3, 171), (17, 257), (129, 128))
        for sk in (1, 33, 257, 511, 512)
    ]
    compare(scheduler, oracle, cases, artifact_dir, f"multiblock-{ratio}")
    # Exercise every source-adjusted base, including odd values, rather than
    # assuming that all M blocks are 128 or 256 rows.
    cases = [case(10, 2, m, 65, 20, ratio) for m in range(128, 257)]
    results = compare(scheduler, oracle, cases, artifact_dir, f"adjusted-m-{ratio}")
    assert [row[1] for row in results] == list(range(128, 257))


@pytest.mark.parametrize("ratio", (1, 2))
def test_local_memory_boundaries(scheduler, ratio):
    good = case(ratio=ratio)
    assert evaluate(scheduler, [good])[0][0] == 0
    cases = []
    for index in range(6, 11):
        bad = good.copy()
        bad[index] -= 1
        cases.append(bad)
    assert evaluate(scheduler, cases) == [[4]] * 5


@pytest.mark.parametrize(
    "overrides,status",
    [
        ({"batch": 0}, 1),
        ({"batch": -1}, 1),
        ({"sq": 0}, 1),
        ({"sq": -1}, 1),
        ({"heads": 0}, 1),
        ({"heads": -1}, 1),
        ({"sk": 0}, 1),
        ({"sk": -1}, 1),
        ({"heads": 2**23}, 2),
        ({"sq": 2**30 + 1, "heads": 1}, 2),
        ({"sk": 513}, 1),
        ({"batch": 2**32}, 2),
        ({"sq": 2**63 - 1}, 2),
        ({"heads": 2**32}, 2),
        ({"sk": 2**32}, 2),
        ({"sq": 2**32 - 1, "heads": 2**32 - 1}, 2),
        ({"sq": 2**30 - 63, "heads": 4}, 2),
        ({"batch": 2**24, "sq": 2**30, "heads": 1}, 2),
        ({"batch": 2**31, "sq": 513, "heads": 1, "aic": 1}, 2),
        ({"batch": 2**32 - 1}, 2),  # Source uint32 lcm would overflow before allocating vectors.
        ({"batch": 2**31, "heads": 256, "aic": 3}, 2),
        ({"aic": 0, "aiv": 0}, 3),
        ({"aic": 27, "aiv": 54}, 3),
        ({"aiv": 0}, 3),
        ({"aiv": 19}, 3),
        ({"aiv": 21}, 3),
        ({"aiv": 60}, 3),
        ({"aic": 2**32 - 1, "aiv": 2**32 - 1}, 3),
        ({"block_dim": 21}, 3),
        ({"batch": 40, "block_dim": 1}, 3),
    ],
)
def test_rejected_inputs_preserve_output(scheduler, overrides, status):
    assert evaluate(scheduler, [case(**overrides)]) == [[status]]
