"""CPU checks that incomplete profiler evidence cannot become a baseline."""

import csv
from decimal import Decimal

import pytest

from tests.mla_cann_profile import parse_kernel_details


FIA = "aclnnFusedInferAttentionScoreV4_FusedInferAttentionScore_FusedInferAttentionScore"
SLICE = "aclnnFusedInferAttentionScoreV4_SliceAiCore_Slice"


def write_capture(tmp_path, calls):
    rows = []
    start = Decimal("1790225174683160.877")
    for kernels in calls:
        for name, duration in kernels:
            rows.append(
                {
                    "Step Id": "1",  # Deliberately unreliable, as in torch_npu 2.7.1.
                    "Device_id": "7",
                    "Name": name,
                    "Start Time(us)": str(start) + "\t",
                    "Duration(us)": str(duration),
                }
            )
            start += Decimal("100.001")  # Dispatch gaps are not kernel duration.
    path = tmp_path / "kernel_details.csv"
    write_rows(path, rows)
    return path, rows


def write_rows(path, rows):
    headers = ["Step Id", "Device_id", "Name", "Start Time(us)", "Duration(us)"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_kernel_and_conversion_costs_are_separate(tmp_path):
    calls = [[(FIA, duration)] for duration in (10, 12, 14, 16)]
    calls += [[(SLICE, 2)] * 4 + [(FIA, duration)] for duration in (20, 22, 24, 26)]
    path, rows = write_capture(tmp_path, calls)
    write_rows(path, list(reversed(rows)))
    dense, split = parse_kernel_details(path, conversion_counts=[0, 4], iterations=2, rounds=2)
    assert dense["median_fia_kernel_us"] == 13
    assert dense["median_conversion_kernel_us"] == 0
    assert split["median_fia_kernel_us"] == 23
    assert split["median_conversion_kernel_us"] == 8
    assert split["median_device_kernel_sum_us"] == 31
    assert split["round_means"] == [
        dict(fia_kernel_us=21, conversion_kernel_us=8),
        dict(fia_kernel_us=25, conversion_kernel_us=8),
    ]
    assert split["kernel_counts"] == {SLICE: 16, FIA: 4}


@pytest.mark.parametrize(
    "damage",
    [
        "missing_fia",
        "extra_fia",
        "missing_copy",
        "trailing_copy",
        "unknown",
        "nan",
        "zero",
        "device",
    ],
)
def test_rejects_incomplete_or_unattributable_capture(tmp_path, damage):
    path, rows = write_capture(tmp_path, [[(SLICE, 2)] * 4 + [(FIA, 20)]])
    if damage == "missing_fia":
        rows.pop()
    elif damage == "extra_fia":
        rows.append(rows[-1].copy())
    elif damage == "missing_copy":
        rows.pop(0)
    elif damage == "trailing_copy":
        rows.append(rows[-1] | {"Name": SLICE, "Start Time(us)": "1790225174684000"})
    elif damage == "unknown":
        rows[0]["Name"] = "aclnnUnrelated_Slice"
    elif damage == "nan":
        rows[0]["Duration(us)"] = "nan"
    elif damage == "zero":
        rows[0]["Duration(us)"] = "0"
    else:
        rows[0]["Device_id"] = "6"
    write_rows(path, rows)
    with pytest.raises(ValueError):
        parse_kernel_details(path, conversion_counts=[4], iterations=1, rounds=1)


@pytest.mark.parametrize("empty", [True, False])
def test_rejects_missing_profiler_data(tmp_path, empty):
    path, _ = write_capture(tmp_path, [])
    if not empty:
        path.write_text("Name,Duration(us)\nFIA,12\n")
    with pytest.raises(ValueError):
        parse_kernel_details(path, conversion_counts=[0], iterations=1, rounds=1)
