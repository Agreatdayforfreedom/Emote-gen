"""Minimal ComfyUI HTTP API client (stdlib only): upload image, queue workflow, wait, fetch images.

Workflows are stored in ComfyUI *API format* JSON with `{{placeholders}}` (e.g. {{prompt}}, {{seed}}, {{image}})
that `fill_workflow` substitutes, so seed and prompt are fixed and reproducible from config.
"""
from __future__ import annotations

import io
import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .sprite import EmoteError

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def fill_workflow(workflow: Any, params: dict[str, Any]) -> Any:
    """Recursively replace {{name}} placeholders. A string that is exactly one placeholder keeps the
    parameter's type (so {{seed}} becomes an int); placeholders inside longer strings are stringified."""
    if isinstance(workflow, dict):
        return {k: fill_workflow(v, params) for k, v in workflow.items()}
    if isinstance(workflow, list):
        return [fill_workflow(v, params) for v in workflow]
    if isinstance(workflow, str):
        whole = _PLACEHOLDER.fullmatch(workflow)
        if whole:
            return _lookup(params, whole.group(1))
        return _PLACEHOLDER.sub(lambda m: str(_lookup(params, m.group(1))), workflow)
    return workflow


def _lookup(params: dict[str, Any], name: str) -> Any:
    if name not in params:
        raise EmoteError(f"Workflow placeholder {{{{{name}}}}} has no value (known: {sorted(params)})")
    return params[name]


class ComfyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8188, timeout: float = 600.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    # -- low level -------------------------------------------------------------------------------
    def _request(self, path: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise EmoteError(f"ComfyUI returned HTTP {exc.code} for {path}: {body[:1500]}") from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            raise EmoteError(f"Cannot reach ComfyUI at {self.base} ({exc}). Start it or set ai.comfy_dir so "
                             "the pipeline can launch it.") from exc

    def alive(self) -> bool:
        # A refused connection to a closed port takes ~2 s on Windows (connect retries), so probe the socket
        # with a short timeout first.
        parsed = urllib.parse.urlparse(self.base)
        try:
            socket.create_connection((parsed.hostname, parsed.port), timeout=0.4).close()
        except OSError:
            return False
        try:
            self._request("/system_stats")
            return True
        except EmoteError:
            return False

    def system_stats(self) -> dict:
        return json.loads(self._request("/system_stats"))

    # -- workflow --------------------------------------------------------------------------------
    def upload_image(self, path: str | Path, name: str | None = None) -> str:
        p = Path(path)
        boundary = uuid.uuid4().hex
        name = name or p.name
        body = io.BytesIO()
        for field, value in (("overwrite", "true"), ("type", "input")):
            body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
        body.write(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
                   "Content-Type: image/png\r\n\r\n".encode())
        body.write(p.read_bytes())
        body.write(f"\r\n--{boundary}--\r\n".encode())
        res = json.loads(self._request("/upload/image", body.getvalue(),
                                       {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
        return res["name"]

    def queue(self, workflow: dict) -> str:
        payload = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        res = json.loads(self._request("/prompt", payload, {"Content-Type": "application/json"}))
        if res.get("node_errors"):
            raise EmoteError(f"ComfyUI rejected the workflow: {json.dumps(res['node_errors'])[:1500]}")
        return res["prompt_id"]

    def wait(self, prompt_id: str, poll: float = 1.0) -> dict:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            hist = json.loads(self._request(f"/history/{prompt_id}"))
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    raise EmoteError(f"ComfyUI execution failed: {json.dumps(msgs)[:1500]}")
                return entry
            time.sleep(poll)
        raise EmoteError(f"Timed out after {self.timeout:.0f}s waiting for ComfyUI job {prompt_id}")

    def fetch_images(self, entry: dict, node_id: str | None = None) -> list[np.ndarray]:
        """RGBA arrays of every image output (optionally only from one node)."""
        out = []
        for nid, o in entry.get("outputs", {}).items():
            if node_id is not None and nid != node_id:
                continue
            for img in o.get("images", []):
                q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                            "type": img.get("type", "output")})
                data = self._request(f"/view?{q}")
                out.append(np.array(Image.open(io.BytesIO(data)).convert("RGBA")))
        if not out:
            raise EmoteError("ComfyUI job finished but produced no images")
        return out

    def run(self, workflow: dict, node_id: str | None = None) -> list[np.ndarray]:
        return self.fetch_images(self.wait(self.queue(workflow)), node_id)


def launch_server(comfy_dir: str | Path, host: str, port: int, wait_s: float = 180.0) -> subprocess.Popen:
    """Start the ComfyUI portable build (python_embeded\\python.exe ComfyUI\\main.py) and wait until it answers."""
    root = Path(comfy_dir).resolve()  # absolute: the child process runs with cwd=root
    py = root / "python_embeded" / "python.exe"
    main = root / "ComfyUI" / "main.py"
    if not py.is_file() or not main.is_file():
        raise EmoteError(f"ai.comfy_dir={root} is not a ComfyUI portable folder (missing python_embeded or ComfyUI\\main.py)")
    proc = subprocess.Popen([str(py), "-s", str(main), "--windows-standalone-build", "--listen", host,
                             "--port", str(port), "--disable-auto-launch"], cwd=root,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = ComfyClient(host, port)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            raise EmoteError(f"ComfyUI exited early (code {proc.returncode}); run it manually once to see the error")
        if client.alive():
            return proc
        time.sleep(2)
    proc.terminate()
    raise EmoteError(f"ComfyUI did not come up within {wait_s:.0f}s")
