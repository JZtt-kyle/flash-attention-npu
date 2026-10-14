"""Validate and pair the stream-ordered partial/merge profiler rows."""

from decimal import Decimal


def pair_split_kernels(rows, labels, dtype):
    index = 0
    paired = []
    for label in labels:
        count = 2 if label["name"] == "fa3" else 1
        group = rows[index : index + count]
        if len(group) != count:
            raise ValueError("Missing split profiler kernel")
        if count == 2:
            for row, stage in zip(group, ("partial", "merge")):
                if f"mla_split_bsnd_{stage}_{dtype}" not in row["Name"]:
                    raise ValueError("Wrong split profiler kernel order or dtype")
            first_end = Decimal(group[0]["Start Time(us)"]) + Decimal(group[0]["Duration(us)"])
            if group[0]["Device_id"] != group[1]["Device_id"] or first_end > Decimal(
                group[1]["Start Time(us)"]
            ):
                raise ValueError("Split phases overlap or run on different devices")
            row = dict(group[0])
            row["Duration(us)"] = str(sum(Decimal(r["Duration(us)"]) for r in group))
            row["Split End Time(us)"] = str(
                Decimal(group[1]["Start Time(us)"]) + Decimal(group[1]["Duration(us)"])
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
