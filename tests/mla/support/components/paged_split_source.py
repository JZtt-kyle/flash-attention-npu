"""Pinned PA FD numerical fixture shared by layout acceptance modules."""

import ctypes as ct
import json
from pathlib import Path
import pytest
from tests.mla.support.mla_split_paged_bsnd_source_build import build_paged_partial, sha256


@pytest.fixture(scope="module")
def paged_source(merge_probes):
    library = build_paged_partial(merge_probes["directory"])
    lib = ct.CDLL(str(library))
    lib.launch_paged_partial.argtypes = (
        [ct.c_void_p] * 10 + [ct.c_float] + [ct.c_void_p] * 3 + [ct.c_uint32] * 2 + [ct.c_void_p]
    )
    lib.launch_paged_partial.restype = ct.c_int
    manifest = json.loads((library.parent / "paged-build-manifest.json").read_text())
    loaded = dict(library=str(Path(lib._name).resolve()), sha256=sha256(Path(lib._name)))
    assert loaded == {key: manifest[key] for key in loaded}
    (library.parent / "paged-loaded.json").write_text(json.dumps(loaded, indent=2) + "\n")
    yield merge_probes, lib
    assert sha256(library) == loaded["sha256"]
    assert all(sha256(Path(name)) == digest for name, digest in manifest["inputs"].items())
