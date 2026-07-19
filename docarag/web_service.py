"""Start, stop and inspect the web UI as a managed service.

The server runs detached with its pid and log kept under the runs root, so
one command starts it, one stops it, and one says whether it is up and what
it is serving -- no pkill, no orphaned terminals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

DEFAULT_PORT = 8765


def _service_dir() -> Path:
    root = Path(os.environ.get("DOCARAG_RUNS_DIR", "runs")).expanduser().resolve()
    directory = root / "_web"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def _pid_file() -> Path:
    return _service_dir() / "web.pid"


def _log_file() -> Path:
    return _service_dir() / "web.log"


def _running_pid() -> int | None:
    try:
        payload = json.loads(_pid_file().read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/health", timeout=3
        ) as response:
            return json.load(response)
    except Exception:
        return None


def start(port: int, run_dir: str | None) -> int:
    if _running_pid() is not None:
        payload = json.loads(_pid_file().read_text(encoding="utf-8"))
        print(f"web UI already running: http://127.0.0.1:{payload['port']}")
        return 0
    repo = Path(__file__).resolve().parent.parent
    command = [sys.executable, str(repo / "Scripts" / "web.py"), "--port", str(port)]
    if run_dir:
        command.insert(2, run_dir)
    log = open(_log_file(), "ab")
    process = subprocess.Popen(
        command,
        cwd=str(repo),
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    _pid_file().write_text(
        json.dumps({"pid": process.pid, "port": port}), encoding="utf-8"
    )
    for _ in range(30):
        if _health(port):
            print(f"web UI started: http://127.0.0.1:{port}  (pid {process.pid})")
            print(f"log: {_log_file()}")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.5)
    print(f"web UI failed to start; see {_log_file()}", file=sys.stderr)
    _pid_file().unlink(missing_ok=True)
    return 1


def stop() -> int:
    pid = _running_pid()
    if pid is None:
        print("web UI is not running")
        _pid_file().unlink(missing_ok=True)
        return 0
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.25)
    else:
        os.kill(pid, signal.SIGKILL)
    _pid_file().unlink(missing_ok=True)
    print(f"web UI stopped (pid {pid})")
    return 0


def status() -> int:
    pid = _running_pid()
    if pid is None:
        print("web UI: stopped")
        return 1
    payload = json.loads(_pid_file().read_text(encoding="utf-8"))
    health = _health(int(payload["port"]))
    if health is None:
        print(f"web UI: pid {pid} alive but not answering on port {payload['port']}")
        return 1
    serving = Path(str(health.get("runDir", ""))).name[:12]
    print(
        f"web UI: running  http://127.0.0.1:{payload['port']}  "
        f"(pid {pid}, serving {serving}…)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the web UI service")
    parser.add_argument("action", choices=("start", "stop", "status", "restart"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--run", help="run directory to serve initially")
    args = parser.parse_args(argv)
    if args.action == "start":
        return start(args.port, args.run)
    if args.action == "stop":
        return stop()
    if args.action == "restart":
        stop()
        return start(args.port, args.run)
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
