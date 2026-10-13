"""Dispatch evidence must name the executed fixed-package binary and key."""

import json

import pytest

from tests.mla_native_dispatch import parse_dispatch, sha256


@pytest.fixture
def native_log(tmp_path):
    root = tmp_path / "vendor/op_impl/ai_core/tbe/kernel"
    binary = root / "ascend910b/fused_infer_attention_score/FusedInferAttentionScore_abcd.o"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"inert test binary")
    metadata = binary.with_suffix(".json")
    # Fixed ca18aee GenTilingKey: non-PA/PA, each without/with FD.
    keys = [105000000010000001, 105000000010100001, 105000000010200001, 105000000010300001]
    metadata.write_text(
        json.dumps(
            dict(
                binFileName=binary.stem,
                binFileSuffix=".o",
                sha256=sha256(binary),
                kernelList=[dict(tilingKey=k) for k in keys],
            )
        )
    )
    config = root / "config/ascend910b/fused_infer_attention_score.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            dict(
                binList=[
                    dict(
                        inputs=[dict(dtype="float16")],
                        outputs=[dict(dtype="float16")],
                        binInfo=dict(jsonFilePath=str(metadata.relative_to(root))),
                    )
                ]
            )
        )
    )
    lines = [f"Register Kernel succ bin path: {binary}\n"]
    for index, key in enumerate(keys):
        lines += [
            f"MLA_STRIDE_NATIVE_BEGIN fp16 {index}\n",
            "CreateNewKernel: kernel_name=FusedInferAttentionScore_abcd_999,\n",
            f"LaunchKernel: kernel info : kernel_name=FusedInferAttentionScore_abcd_{key},\n",
            f"MLA_STRIDE_NATIVE_END fp16 {index}\n",
        ]
    log = tmp_path / "native.log"
    log.write_text("".join(lines))
    return log, tmp_path / "vendor", binary


def test_actual_launch_and_fd_classification(native_log):
    log, vendor, _ = native_log
    result = parse_dispatch(log, "fp16", 4, vendor)
    assert [x["native_fd"] for x in result["cases"]] == [False, True, False, True]
    assert [x["e2e_applicable"] for x in result["cases"]] == [True, False, True, False]
    assert all(x["native_template"] == 5 for x in result["cases"])
    assert result["artifacts"][str(log)] == sha256(log)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-launch",
        "duplicate",
        "registration",
        "foreign",
        "key",
        "unfinished",
        "order",
        "binary",
    ],
)
def test_reject_incomplete_or_stale_dispatch(native_log, fault):
    log, vendor, binary = native_log
    text = log.read_text()
    launch = "LaunchKernel: kernel info : kernel_name=FusedInferAttentionScore_abcd_105000000010000001,\n"
    if fault == "binary":
        binary.write_bytes(b"changed")
    else:
        text = {
            "missing-launch": lambda: text.replace(launch, ""),
            "duplicate": lambda: text.replace(launch, launch * 2),
            "registration": lambda: text.split("\n", 1)[1],
            "foreign": lambda: text.replace(
                "kernel_name=FusedInferAttentionScore_abcd_105",
                "kernel_name=FusedInferAttentionScore_ffff_105",
            ),
            "key": lambda: text.replace("105000000010000001", "105000000019000001"),
            "unfinished": lambda: text.replace("MLA_STRIDE_NATIVE_END fp16 1\n", ""),
            "order": lambda: text.replace(
                "MLA_STRIDE_NATIVE_BEGIN fp16 0", "MLA_STRIDE_NATIVE_BEGIN fp16 1"
            ),
        }[fault]()
        log.write_text(text)
    with pytest.raises(RuntimeError):
        parse_dispatch(log, "fp16", 4, vendor)
