"""Small dependency-free sequential HTTP latency check."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _call(url: str) -> dict:
    request = urllib.request.Request(
        url,
        data=b'{"user_id":"A"}',
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read())


def main(argv: list[str] | None = None) -> int:
    """Start a server, warm it, and print p50/p95/p99 latency."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args(argv)
    project = Path(__file__).parents[1]
    port = _port()
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(project / "src"),
            "RECO_HOME": str((project / "var").resolve()),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "reco.serving.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=project,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/v1/recommendations"
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                _call(url)
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("uvicorn did not become ready")
        for _ in range(args.warmup):
            _call(url)
        durations: list[float] = []
        for _ in range(args.requests):
            started = time.perf_counter()
            _call(url)
            durations.append((time.perf_counter() - started) * 1000)
        result = {
            "requests": args.requests,
            "p50_ms": round(float(np.percentile(durations, 50)), 3),
            "p95_ms": round(float(np.percentile(durations, 95)), 3),
            "p99_ms": round(float(np.percentile(durations, 99)), 3),
        }
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
