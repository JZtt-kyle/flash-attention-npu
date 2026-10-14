"""Fresh partial and FD probes; original pinned headers, independent manifests."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from tests.mla.support.mla_dense_build import ROOT, build_probes, sha256


def build_merge(directory, dtype):
    directory = Path(directory)
    partial_dir = directory / "partial"
    partial = build_probes(partial_dir, dtype, split_kv=True)
    manifest = json.loads((partial_dir / "build-manifest.json").read_text())
    probe = ROOT / "tests/mla/support/mla_split_kv_merge_probe.cpp"

    def compile_one(name):
        command = json.loads((partial_dir / f"{name}-command.json").read_text())
        command[command.index(str(ROOT / "tests/mla/support/mla_split_kv_attention_probe.cpp"))] = (
            str(probe)
        )
        command[command.index("-MF") + 1] = str(directory / f"{name}.d")
        command[command.index("-o") + 1] = str(directory / f"{name}.so")
        (directory / f"{name}-command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (directory / f"{name}-compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        deps = [
            arg
            for arg in command[: command.index("-MMD")]
            if arg != "-shared" and not arg.startswith(("-l", "-L", "-Wl,"))
        ]
        deps += ["--offload-device-only", "-M", "-MF", str(directory / f"{name}-device.d")]
        (directory / f"{name}-deps-command.json").write_text(json.dumps(deps, indent=2) + "\n")
        with (directory / f"{name}-deps.log").open("w") as log:
            subprocess.run(deps, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
        text = (directory / f"{name}-device.d").read_text()
        assert ("oracle-source" in text) == (name == "oracle")
        assert (
            "fia_block_vec_flashdecode_arch22.h" if name == "oracle" else "mla_split_kv_merge.hpp"
        ) in text
        return str(directory / f"{name}.so")

    with ThreadPoolExecutor(max_workers=2) as pool:
        libraries = dict(zip(("oracle", "migrated"), pool.map(compile_one, ("oracle", "migrated"))))
    libraries.update({f"partial_{name}": path for name, path in partial.items()})
    for p in (
        probe,
        Path(__file__).resolve(),
        ROOT / "csrc/ascend910/flash_attn_npu_3/mla/mla_split_kv_merge_params.h",
    ):
        manifest["sources"][str(p.relative_to(ROOT))] = sha256(p)
    for depfile in directory.glob("*-device.d"):
        for item in depfile.read_text().replace("\\\n", " ").split()[1:]:
            p = Path(item).resolve()
            manifest["dependencies"][str(p)] = sha256(p)
    manifest["split_kv_merge"] = True
    manifest["libraries"] = {
        name: dict(path=path, sha256=sha256(Path(path))) for name, path in libraries.items()
    }
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return libraries
