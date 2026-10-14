"""Validate and pair the stream-ordered partial/merge profiler rows."""

from decimal import Decimal


def pair_split_kernels(rows, labels, dtype, *, automatic=False):
    index = 0
    paired = []
    for label in labels:
        count = (3 if automatic else 2) if label["name"] == "fa3" else 1
        group = rows[index : index + count]
        if len(group) != count:
            raise ValueError("Missing split profiler kernel")
        if count > 1:
            stages = ("init", "partial", "merge") if automatic else ("partial", "merge")
            prefix = "mla_auto_split_bsnd" if automatic else "mla_split_bsnd"
            for row, stage in zip(group, stages):
                if f"{prefix}_{stage}_{dtype}" not in row["Name"]:
                    raise ValueError("Wrong split profiler kernel order or dtype")
            for first, second in zip(group, group[1:]):
                first_end = Decimal(first["Start Time(us)"]) + Decimal(first["Duration(us)"])
                if first["Device_id"] != second["Device_id"] or first_end > Decimal(
                    second["Start Time(us)"]
                ):
                    raise ValueError("Split phases overlap or run on different devices")
            row = dict(group[0])
            row["Duration(us)"] = str(sum(Decimal(r["Duration(us)"]) for r in group))
            row["Split End Time(us)"] = str(
                Decimal(group[-1]["Start Time(us)"]) + Decimal(group[-1]["Duration(us)"])
            )
            row["Split Segments"] = [dict(r) for r in group]
        else:
            row = group[0]
            if "FusedInferAttentionScore" not in row["Name"]:
                raise ValueError("Missing native profiler kernel")
        paired.append(row)
        index += count
    if index != len(rows):
        raise ValueError("Unmatched split profiler kernels")
    return paired
