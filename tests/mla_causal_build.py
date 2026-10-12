"""Pinned CPU oracle and compiler identity for causal scheduling acceptance."""

import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import time

from tests.test_flash_attn_npu_v3_mla_schedule import CANN_COMMIT, MLA, ROOT, run

SOURCE_FILES = ("attention/common/op_host/split_core.h", "attention/common/op_host/split_core.cpp")


def source_blobs():
    root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    return {
        name: subprocess.check_output(
            ["git", "-C", str(root), "show", CANN_COMMIT + ":" + name], timeout=30
        )
        for name in SOURCE_FILES
    }


def toolchain_identity(sanitize):
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if not compiler:
        raise RuntimeError("Causal CPU validation requires a C++ compiler")
    paths = [Path(compiler).resolve()]
    for option in ("-print-prog-name=cc1plus", "-print-file-name=libstdc++.so"):
        paths.append(Path(run([compiler, option]).stdout.strip()).resolve())
    if sanitize:
        for library in ("libasan.so", "libubsan.so"):
            paths.append(
                Path(run([compiler, "-print-file-name=" + library]).stdout.strip()).resolve()
            )
    return dict(
        compiler=compiler,
        version=run([compiler, "--version"]).stdout,
        binaries={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        source_commit=CANN_COMMIT,
        source_blobs={
            name: hashlib.sha256(blob).hexdigest() for name, blob in source_blobs().items()
        },
        sanitize=bool(sanitize),
    )


def compile_drivers(directory):
    directory.mkdir(parents=True, exist_ok=True)
    sanitize = os.environ.get("MLA_SCHEDULE_SANITIZE") == "1"
    (directory / "toolchain.json").write_text(
        json.dumps(toolchain_identity(sanitize), indent=2) + "\n"
    )
    for name, blob in source_blobs().items():
        (directory / Path(name).name).write_bytes(blob)
    (directory / "log").mkdir(exist_ok=True)
    (directory / "log/log.h").write_text(
        "#define OP_LOGI(...) ((void)0)\n#define OP_LOGD(...) ((void)0)\n"
    )
    binaries = []
    for kind in ("adapter", "source", "migrated-source"):
        binary = directory / kind
        command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-g", "-D_GLIBCXX_ASSERTIONS"]
        if sanitize:
            command += [
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
                "-fno-sanitize-recover=all",
            ]
        if kind == "adapter":
            command += ["-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(MLA)]
            command += [
                str(MLA / p)
                for p in (
                    "mla_causal.cpp",
                    "mla_schedule.cpp",
                    "mla_tnd_schedule.cpp",
                    "detail/cann_split_core.cpp",
                )
            ]
            command += [str(ROOT / "tests/mla_causal_schedule_driver.cpp")]
        else:
            if kind == "source":
                command += ["-I" + str(directory), str(directory / "split_core.cpp")]
            else:
                command += [
                    "-DMIGRATED_SPLIT",
                    "-I" + str(MLA),
                    str(MLA / "detail/cann_split_core.cpp"),
                ]
            command += [str(ROOT / "tests/mla_causal_source_driver.cpp")]
        command += ["-o", str(binary)]
        (directory / f"{kind}-compile.json").write_text(json.dumps(command, indent=2) + "\n")
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (directory / f"{kind}-compile.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        binaries.append(binary)
    return binaries


class CpuDriver:
    """One checked process per executable; retain complete request/response logs.

    In particular, LeakSanitizer still runs at exit, once for the entire corpus.
    EOF, malformed output, nonzero exit and sanitizer stderr all fail acceptance.
    """

    def __init__(self, binary):
        self.binary = binary
        self.started = time.monotonic()
        self.requests = 0
        self.stderr_path = binary.with_name(binary.name + ".stderr")
        self.stderr = self.stderr_path.open("w")
        self.inputs = binary.with_name(binary.name + ".stdin").open("w")
        self.outputs = binary.with_name(binary.name + ".stdout").open("w")
        self.process = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def __enter__(self):
        return self

    def evaluate(self, cases):
        values = []
        for item in cases:
            line = " ".join(map(str, item)) + "\n"
            self.inputs.write(line)
            self.inputs.flush()
            self.process.stdin.write(line)
            self.process.stdin.flush()
            if not self.selector.select(timeout=120):
                raise RuntimeError(f"CPU driver response timed out: {self.binary}")
            response = self.process.stdout.readline()
            if not response:
                raise RuntimeError(
                    f"CPU driver exited before a response: {self.stderr_path.read_text()}"
                )
            self.outputs.write(response)
            self.outputs.flush()
            values.append(json.loads(response))
            self.requests += 1
        return values

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.process.stdin.close()
            try:
                code = self.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise
            extra = self.process.stdout.read()
        finally:
            self.selector.close()
            self.process.stdout.close()
            self.stderr.close()
            self.inputs.close()
            self.outputs.close()
            self.binary.with_name(self.binary.name + "-run.json").write_text(
                json.dumps(
                    dict(
                        returncode=self.process.returncode,
                        requests=self.requests,
                        seconds=time.monotonic() - self.started,
                    ),
                    indent=2,
                )
                + "\n"
            )
        if code or extra or self.stderr_path.read_text():
            raise RuntimeError(
                f"CPU driver failed at exit ({code}): {self.stderr_path.read_text()}; extra={extra!r}"
            )
