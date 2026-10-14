"""The new CPU suites retain fixed-source identity and isolated execution."""

import hashlib

import pytest

from tests.mla.support import mla_validation as runner
from tests.mla.support import mla_window_build as window


def test_window_source_missing_or_changed_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("MLA_FA3_SOURCE_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        window.fa3_blob()
    (tmp_path / "flash_api.cpp").write_text("wrong revision")
    with pytest.raises(RuntimeError, match="pinned"):
        window.fa3_blob()


@pytest.mark.parametrize("suite", ["window-cpu", "window-sanitize"])
def test_window_dependency_identity_includes_both_sources(monkeypatch, suite):
    monkeypatch.setattr(window, "fa3_blob", lambda: b"checked fixed FA3 source")
    monkeypatch.setattr(
        window,
        "toolchain_identity",
        lambda sanitize: {"sanitize": sanitize, "source_blobs": {"CANN": "fixed"}},
    )
    result = runner.suite_dependencies(suite)
    assert result == dict(
        sanitize=suite.endswith("sanitize"),
        source_blobs={"CANN": "fixed"},
        fa3_commit=window.FA3_COMMIT,
        fa3_source_sha256=hashlib.sha256(b"checked fixed FA3 source").hexdigest(),
    )


def test_window_normal_and_sanitizer_select_same_cases():
    assert runner.SUITES["window-cpu"] == runner.SUITES["window-sanitize"] == [runner.WINDOW_CPU]
    assert runner.SUITES["causal-cpu"] == runner.SUITES["causal-sanitize"] == [runner.CAUSAL_CPU]
    assert {"window-cpu", "window-sanitize", "window-tools"} <= runner.CPU_SUITES


def test_window_extracts_exact_both_official_normalizers():
    blob = window.fa3_blob()
    source = blob.decode()
    oracle = window.normalization_oracle(blob)
    for start, end in [
        (
            "    if (window_size_left >= max_seqlen_k - 1)",
            "    params.attention_chunk = attention_chunk;",
        ),
        ("    if (window_size_left >= seqlen_k - 1)", "    if (!is_varlen_q)"),
        ("    // Causal is the special case", "    params.attention_chunk = attention_chunk;"),
    ]:
        fragment = source[source.index(start) :]
        fragment = fragment[: fragment.index(end)]
        assert fragment in oracle
