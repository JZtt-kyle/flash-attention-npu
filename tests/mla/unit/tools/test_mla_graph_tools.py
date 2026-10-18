"""Graph profiling phases, lifecycle boundaries and driver ownership."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
from tests.mla.support.mla_graph_profile_rows import (
    PERFORMANCE,
    selected,
    partition_kernels,
    select_model_rows,
)
from tests.mla.support import mla_validation as driver


def rows(names):
    return [
        dict(Name=name, Device_id="0", **{"Start Time(us)": str(i * 3), "Duration(us)": "2"})
        for i, name in enumerate(names)
    ]


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_graph_phase_boundaries(representation, dtype):
    dt = "Fp16" if dtype == "fp16" else "Bf16"
    spec = dict(PERFORMANCE[1], layout="tnd")
    names = ["ComputeMlaGqaMetadata", "MlaGqa" + dt + "Attention"]
    labels = [dict(case=0, name="fa3", round=0, iteration=0)]
    result = partition_kernels(rows(names), labels, [spec], dtype, representation)
    assert result[0]["kernel_us"] == result[0]["adaptation_us"] == 2
    for broken in (names[1:], names[:1], names + names[-1:], ["MlaGqa" + dt + "Init", *names]):
        with pytest.raises(ValueError):
            partition_kernels(rows(broken), labels, [spec], dtype, representation)
    spec["layout"] = "bsnd"
    with pytest.raises(ValueError, match="lifecycle"):
        partition_kernels(rows(names), labels, [spec], dtype, representation)


@pytest.mark.parametrize("layout", ["bsnd", "tnd", "mixed", "paged_bsnd", "paged_tnd"])
@pytest.mark.parametrize("representation", ["compressed", "prefill"])
def test_legacy_graph_phases(layout, representation):
    spec = dict(PERFORMANCE[0], layout=layout)
    names = ["ComputeMlaMetadata"] if layout == "tnd" else []
    if representation == "prefill":
        names += ["MlaPrefillTndFp16Init"] if layout != "bsnd" else []
        names += ["MlaPrefillBsndFp16Attention"]
    else:
        prefix = dict(
            bsnd="dense",
            tnd="tnd",
            mixed="tnd_bsnd",
            paged_bsnd="paged_bsnd",
            paged_tnd="paged_tnd",
        )[layout]
        names += ["mla_" + prefix + "_fp16"]
    result = partition_kernels(
        rows(names), [dict(case=0, name="fa3")], [spec], "fp16", representation
    )
    assert result[0]["kernel_us"] == 2
    source = rows(["Transpose", "native_FusedInferAttentionScore"])
    result = partition_kernels(
        source, [dict(case=0, name="native_cann")], [spec], "fp16", representation
    )
    assert result[0]["kernel_us"] == result[0]["adaptation_us"] == 2


def test_selection_and_resume_boundary():
    assert len(PERFORMANCE) * 4 <= 100
    assert {(s["layout"], s["splits"]) for s in selected(True)} == {
        (s["layout"], s["splits"]) for s in PERFORMANCE
    }
    assert {"graph-profile", "graph-profile-check"} <= driver.PERFORMANCE_SUITES
    assert "graph-tools" in driver.CPU_SUITES
    assert driver.SUITES["graph-profile"] == driver.SUITES["graph-profile-check"]


def test_graph_subprocess_and_helper_dependencies():
    from tests.mla.support.mla_validation_dependencies import suite_sources

    sources = suite_sources(driver.ROOT, driver.SUITES["graph-profile"], driver.sources())
    assert {
        "tests/mla/support/mla_graph_profile.py",
        "tests/mla/support/mla_graph_profile_rows.py",
        "tests/mla/support/public_cases/graph.py",
        "tests/mla/support/mla_gqa_options_profile.py",
        "tests/mla/support/native_package.py",
    } <= sources.keys()


def test_native_failure_propagates(monkeypatch, tmp_path):
    path = Path(driver.ROOT) / "tests/mla/e2e/graph/public_profile.py"
    spec = importlib.util.spec_from_file_location("graph_profile_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "native_vendor", lambda: tmp_path)
    monkeypatch.setenv("MLA_VALIDATION_RESULT", str(tmp_path))
    monkeypatch.setenv("MLA_GRAPH_PROFILE_CHECK", "1")
    observed = []

    def fail(command, **kwargs):
        observed.append(copy.deepcopy(command))
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.test_native_public_profile()
    assert len(observed) == 1 and "--check" in observed[0]


@pytest.mark.parametrize("profile_probe", [False, True])
def test_graph_worker_captures_only_inside_profiler(monkeypatch, tmp_path, profile_probe):
    from tests.mla.support import mla_graph_profile as profile

    specs = selected(True)
    proof = tmp_path / "dispatch.json"
    proof.write_text(
        json.dumps(dict(dtype="fp16", representation="compressed", specs=specs, artifacts={}))
    )
    monkeypatch.setenv("ASCEND_GLOBAL_LOG_LEVEL", "3")
    monkeypatch.setattr(profile, "make_case", lambda **kwargs: kwargs)
    monkeypatch.setattr(profile, "metadata", lambda case: None)
    prepared = []

    def eager(case, phase):
        prepared.append((case, phase))
        return case

    monkeypatch.setattr(profile, "native_call", lambda case: lambda: eager(case, "native_cann"))
    monkeypatch.setattr(profile, "invoke", lambda case, record: eager(case, "fa3"))
    monkeypatch.setattr(profile.torch.npu, "synchronize", lambda: None)
    recording, captured = False, []

    class Graph:
        def replay(self):
            assert recording

    def capture(call, *, warmup):
        assert recording, "Wall/preflight graph captured before profiler RECORD"
        assert warmup is False
        result = call()
        captured.append(result)
        return Graph(), result

    def sample(call, selected_specs, directory, dtype, **kwargs):
        nonlocal recording
        assert len(prepared) == 2 * len(selected_specs)
        assert [phase for _, phase in prepared] == ["fa3", "native_cann"] * len(selected_specs)
        call(0, "fa3")  # Existing sampler's WARMUP call must remain eager.
        assert not captured
        recording = True
        call(0, "fa3")
        assert len(captured) == 2 * len(selected_specs)

    monkeypatch.setattr(profile, "capture", capture)
    monkeypatch.setattr(profile, "sample_kernels", sample)
    profile.run(
        "compressed",
        "fp16",
        tmp_path / "result",
        proof,
        True,
        profile_probe=profile_probe,
        graph_kernels=not profile_probe,
    )
    assert len(captured) == (6 if profile_probe else 2 * len(specs))


def test_graph_worker_requires_dispatch(tmp_path):
    from tests.mla.support.mla_graph_profile import run

    with pytest.raises(ValueError, match="verified native dispatch"):
        run("compressed", "fp16", tmp_path, graph_kernels=True)


def test_wall_samples_exclude_previous_output_release(monkeypatch):
    from tests.mla.support import mla_graph_profile as profile

    ticks, measured, eager_calls = 0, False, 0
    released, retained = [], {}

    class Output:
        def __del__(self):
            nonlocal ticks
            # Simulate expensive output deallocation independently of the
            # current call. Graph outputs remain owned by their captured graph.
            ticks += 200_000
            released.append(measured)

    def call(i, name, mode):
        nonlocal ticks, eager_calls
        ticks += 7_000  # Includes this invocation's output allocation.
        if mode == "graph":
            key = (i, name)
            if key not in retained:
                retained[key] = Output()
            return retained[key]
        eager_calls += 1
        return Output()

    def sync():
        nonlocal ticks
        ticks += 3_000

    def clock():
        nonlocal measured
        measured = not measured
        return ticks

    monkeypatch.setattr(profile.torch.npu, "synchronize", sync)
    monkeypatch.setattr(profile.time, "perf_counter_ns", clock)
    specs = [PERFORMANCE[0], PERFORMANCE[2]]  # Cache and varlen entry/R sets.
    samples = profile.sample_wall(call, specs)
    assert len(samples) == sum(len(profile.modes(s)) for s in specs) * 2 * 5 * 20
    assert {s["e2e_us"] for s in samples} == {10}
    assert len(released) == eager_calls and not any(released)
    assert {(s["mode"], s["round"]) for s in samples} == {
        (mode, round_) for mode in ("eager", "graph") for round_ in range(5)
    }


@pytest.mark.parametrize("failure", ["call", "sync"])
def test_wall_sample_failure_propagates(monkeypatch, failure):
    from tests.mla.support import mla_graph_profile as profile

    calls = 0

    def call(*args):
        nonlocal calls
        calls += 1
        if calls == 11 and failure == "call":
            raise RuntimeError("wall call failed")

    def sync():
        if calls == 11 and failure == "sync":
            raise RuntimeError("wall sync failed")

    monkeypatch.setattr(profile, "modes", lambda spec: ["fa3"])
    monkeypatch.setattr(profile.torch.npu, "synchronize", sync)
    with pytest.raises(RuntimeError, match="wall " + failure + " failed"):
        profile.sample_wall(call, [{}])
    assert calls == 11  # Ten warmup calls, then the first measured invocation.


def test_graph_worker_failure_stops_wall_sampling(monkeypatch, tmp_path):
    path = Path(driver.ROOT) / "tests/mla/e2e/graph/public_profile.py"
    spec = importlib.util.spec_from_file_location("graph_profile_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "native_vendor", lambda: tmp_path)
    monkeypatch.setattr(module, "parse_dispatch", lambda *args, **kwargs: {})
    monkeypatch.setenv("MLA_VALIDATION_RESULT", str(tmp_path))
    monkeypatch.delenv("MLA_GRAPH_PROFILE_CHECK", raising=False)
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: {0, 1})
    affinities, commands = [], []
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda pid, cpus: affinities.append(cpus))

    def execute(command, **kwargs):
        commands.append(command)
        assert kwargs["check"] is True
        if "--dispatch" in command:
            assert "--graph-kernels" in command
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(module.subprocess, "run", execute)
    with pytest.raises(subprocess.CalledProcessError):
        module.test_native_public_profile()
    assert len(commands) == 5  # Four native checks, then a fresh graph worker.
    assert affinities == [{0}, {0, 1}]


def test_profiler_capture_is_lazy_and_reused(monkeypatch):
    from tests.mla.support import mla_graph_profile as profile

    captured, replays, eager_calls = [], [], []

    class Graph:
        def __init__(self, result):
            self.result = result

        def replay(self):
            replays.append(self.result)

    def capture(call, *, warmup):
        assert warmup is False
        result = call()
        captured.append(result)
        return Graph(result), result

    monkeypatch.setattr(profile, "capture", capture)

    def eager(i, name):
        eager_calls.append((i, name))
        return i, name

    call = profile.profiler_replay(eager, [{}, {}])
    assert not captured
    assert call(0, "fa3") == (0, "fa3")
    assert not captured and not replays  # Profiler WARMUP.
    assert eager_calls == [(0, "fa3")]
    assert call(0, "fa3") == (0, "fa3")  # First RECORD call captures.
    assert captured == [(i, name) for i in range(2) for name in ("fa3", "native_cann")]
    assert call(1, "native_cann") == (1, "native_cann")
    assert len(captured) == 4 and replays == [(0, "fa3"), (1, "native_cann")]


def test_model_task_selection_and_identity():
    names = ["Fill", "mla_dense_fp16", "Fill", "mla_dense_fp16"]
    selected = select_model_rows(
        rows(names), [{"Model ID": i} for i in (0xFFFFFFFF, 3, 0xFFFFFFFF, 3)]
    )
    labels = [dict(case=0, name="fa3")] * 2
    result = partition_kernels(selected, labels, [PERFORMANCE[0]], "fp16", "compressed")
    assert len(result) == 2 and all(r["kernel_us"] == 2 for r in result)
    with pytest.raises(ValueError, match="No captured model"):
        select_model_rows(rows(["Fill"]), [{"Model ID": 0xFFFFFFFF}])
    with pytest.raises(ValueError, match="Missing raw"):
        select_model_rows(rows(["Fill"]), [])
    selected[-1]["model_id"] = 4
    with pytest.raises(ValueError, match="changed captured model"):
        partition_kernels(selected, labels, [PERFORMANCE[0]], "fp16", "compressed")
