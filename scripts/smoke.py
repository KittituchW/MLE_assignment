"""Start a real uvicorn process and verify the assignment request over HTTP."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> int:
    """Run the smoke check."""
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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                status, _ = _request(f"http://127.0.0.1:{port}/readyz")
                if status == 200:
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            raise RuntimeError("uvicorn did not become ready")
        status, body = _request(f"http://127.0.0.1:{port}/v1/recommendations", {"user_id": "A"})
        if status != 200 or len(body.get("items", [])) != 10:
            raise RuntimeError(f"smoke request failed: status={status} body={body}")
        print(json.dumps({"status": "ok", "strategy": body["strategy"], "model_version": body["model_version"]}))
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
