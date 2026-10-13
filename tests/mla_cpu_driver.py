"""Checked line-oriented CPU processes, shared by MLA oracle/metadata tests."""

from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time


class CpuDriver:
    """One checked process per executable; retain complete request/response logs.

    In particular, LeakSanitizer still runs at exit, once for the entire corpus.
    EOF, malformed output, nonzero exit and sanitizer stderr all fail acceptance.
    """

    def __init__(self, binary, *, args=(), env=None, log_base=None):
        self.binary = Path(binary)
        binary = self.binary
        self.command = [str(binary), *map(str, args)]
        self.log_base = Path(log_base) if log_base else binary
        self.started = time.monotonic()
        self.requests = 0
        self.stderr_path = self.log_base.with_name(self.log_base.name + ".stderr")
        self.stderr = self.stderr_path.open("w")
        self.inputs = self.log_base.with_name(self.log_base.name + ".stdin").open("w")
        self.outputs = self.log_base.with_name(self.log_base.name + ".stdout").open("w")
        self.process = subprocess.Popen(
            self.command,
            env=env,
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
            self.log_base.with_name(self.log_base.name + "-run.json").write_text(
                json.dumps(
                    dict(
                        command=self.command,
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


class CpuDriverPool(ExitStack):
    """Reuse only identical executables, arguments and process environments."""

    def __init__(self):
        super().__init__()
        self.drivers = {}

    def evaluate(self, binary, cases, *, args=(), env=None):
        binary = Path(binary).resolve()
        env = dict(os.environ if env is None else env)
        # These pytest/reference labels do not affect a CPU executable, and
        # change for each case. Keep every actual driver/sanitizer option.
        for name in ("PYTEST_CURRENT_TEST", "GOLDEN_CACHE_NODEID", "GOLDEN_CACHE_TEST_FILE"):
            env.pop(name, None)
        stat = binary.stat()
        identity = (
            str(binary),
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            tuple(map(str, args)),
            sorted(env.items()),
        )
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        if key not in self.drivers:
            self.drivers[key] = self.enter_context(
                CpuDriver(
                    binary,
                    args=args,
                    env=env,
                    log_base=binary.with_name(binary.name + "-" + key[:16]),
                )
            )
        return self.drivers[key].evaluate(cases)


_pool = None


@contextmanager
def cpu_driver_session():
    global _pool
    previous = _pool
    try:
        with CpuDriverPool() as pool:
            _pool = pool
            yield pool
    finally:
        _pool = previous


def evaluate_cpu(binary, cases, *, args=(), options=None):
    env = os.environ.copy()
    for name, value in (options or {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    if _pool is not None:
        return _pool.evaluate(binary, cases, args=args, env=env)
    with CpuDriver(binary, args=args, env=env) as driver:
        return driver.evaluate(cases)
