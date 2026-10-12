"""Reuse the two CPU equations after the caller's independent logical mapping.

Mapping, source/device execution and comparisons still run on every test. The
key describes the actual canonical inputs, not a seed, shape or test node.
"""

import os
from pathlib import Path
import platform
import sys

import torch

from tests.common import golden_cache


def reference_pair(equation, inputs, options):
    names = ("out64", "lse64", "out_low", "lse_low")

    def compute():
        high = equation(**inputs, **options, upcast=True)
        low = equation(**inputs, **options, upcast=False)
        return dict(zip(names, (*high, *low)))

    metadata = dict(
        options=options,
        layout={
            name: None
            if tensor is None
            else dict(stride=list(tensor.stride()), offset=tensor.storage_offset())
            for name, tensor in inputs.items()
        },
        python=sys.version,
        machine=platform.machine(),
        torch_config=torch.__config__.show(),
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        default_dtype=str(torch.get_default_dtype()),
        environment={
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    )
    values = golden_cache.get_or_compute_golden(
        nodeid="mla-logical-reference",
        metadata=metadata,
        inputs=inputs,
        compute_fn=compute,
        expected_keys=names,
        source_files=[__file__, str(Path(__file__).with_name("mla_ref.py")), golden_cache.__file__],
    )
    return tuple(values[name] for name in names)
