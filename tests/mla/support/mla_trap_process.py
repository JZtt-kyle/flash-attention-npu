"""Run each release Trap in a fresh interpreter and device context."""

from pathlib import Path
import subprocess


def run_isolated(command, log, timeout=120):
    """Preserve logs, timeouts/signals and reap the exact child before returning.

    Package import initializes NPU, so a fork of the pytest parent is unsafe.
    A fresh interpreter never inherits a healthy or failed accelerator context.
    """
    with Path(log).open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        expired = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired = True
            process.kill()
            process.wait()
    return dict(pid=process.pid, returncode=process.returncode, timed_out=expired)
