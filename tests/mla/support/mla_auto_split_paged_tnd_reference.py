"""Fixed-source native endpoints for internal Q-used numerical references."""

from functools import lru_cache
import itertools
import os
from pathlib import Path
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_split_kv_build import compile_drivers
from tests.mla.support.mla_ref import normalize_fa3_window


@lru_cache(maxsize=1)
def native_binary():
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "native-pa-reference"
    return compile_drivers(directory)[1]


def native_schedule(s, m, n, cores, driver=None):
    left, right = normalize_fa3_window(
        s["max_q"], s["capacity"], causal=s["causal"], window_size=s["window"]
    )
    kind = 3 if left < 0 and right == 0 else 4 if left >= 0 or right >= 0 else 0
    left = 2**31 - 1 if kind == 3 else s["capacity"] - 1 if left < 0 else left
    right = 0 if kind == 3 else s["max_q"] - 1 if right < 0 else right
    qs = s["q"] if s["used"] is None else s["used"]
    query = [
        1,
        kind,
        left,
        right,
        s["heads"],
        m,
        n,
        cores,
        2,
        len(qs),
        *itertools.chain.from_iterable(zip(qs, s["lengths"])),
    ]
    if driver is not None:
        return driver.evaluate([query])[0]
    with CpuDriver(native_binary()) as source:
        return source.evaluate([query])[0]
