"""Parse complete prefill calls, preserving initialization and native copies."""

from decimal import Decimal


def partition_kernels(rows, labels, specs, dtype):
    pos, result = 0, []
    prefix = "MlaPrefillBsnd" + {"fp16": "Fp16", "bf16": "Bf16"}[dtype]
    for label in labels:
        ours = label["name"] == "fa3"
        stages = [prefix + "Attention"] + (
            [] if specs[label["case"]]["splits"] == 1 else [prefix + "Merge"]
        )
        group = []
        while pos < len(rows):
            row = rows[pos]
            pos += 1
            group.append(row)
            terminal = (
                stages[-1] in row["Name"]
                if ours
                else row["Name"].endswith("_FusedInferAttentionScore")
            )
            if terminal:
                break
        else:
            raise ValueError("Missing prefill/native terminal task")
        if ours:
            kernels = [r for r in group if prefix in r["Name"]]
            if len(kernels) != len(stages) or not all(
                stage in row["Name"] for stage, row in zip(stages, kernels)
            ):
                raise ValueError("Missing prefill attention/merge phase")
        else:
            kernels = [group[-1]]
        if any("ComputeMla" in row["Name"] for row in group):
            raise ValueError("Explicit reuse unexpectedly regenerated metadata")
        for first, second in zip(group, group[1:]):
            if first["Device_id"] != second["Device_id"] or Decimal(
                first["Start Time(us)"]
            ) + Decimal(first["Duration(us)"]) > Decimal(second["Start Time(us)"]):
                raise ValueError("Prefill phases overlap or change device")
        result.append(
            dict(
                **label,
                tasks=group,
                kernel_us=sum(float(row["Duration(us)"]) for row in kernels),
                adaptation_us=sum(
                    float(row["Duration(us)"]) for row in group if row not in kernels
                ),
            )
        )
    if pos != len(rows):
        raise ValueError("Unclaimed prefill profiler tasks")
    return result
