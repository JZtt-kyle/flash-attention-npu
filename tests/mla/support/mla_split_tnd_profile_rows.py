"""Validate the producer/partial/merge task boundary and retain each phase."""

from decimal import Decimal
import re


def production_sample(rows, metadata_name, stages):
    if len(rows) != 3 or metadata_name not in rows[0]["Op Name"]:
        raise ValueError("Missing TND split metadata or kernel")

    def matches(name, symbol):
        # Ascend's profiler can emit the C++ ABI name, including its argument
        # suffix. Match the full function identifier and its namespace.
        return name == symbol or re.fullmatch(
            rf"_ZN10flash_attn3mla{len(symbol)}{re.escape(symbol)}E.+", name
        )

    if not any(
        all(matches(row["Op Name"], stage + dtype) for row, stage in zip(rows[1:], stages))
        for dtype in ("fp16", "bf16")
    ):
        raise ValueError("Wrong TND split stage or dtype")
    for first, second in zip(rows, rows[1:]):
        end = Decimal(first["Task Start Time(us)"]) + Decimal(first["Task Duration(us)"])
        if first["Device_id"] != second["Device_id"] or end > Decimal(
            second["Task Start Time(us)"]
        ):
            raise ValueError("TND split phases overlap or use different devices")
    return dict(
        rows[-1],
        **{"Task Duration(us)": str(sum(Decimal(r["Task Duration(us)"]) for r in rows[1:]))},
    )
