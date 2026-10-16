import base64
import contextlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request
import zipfile
import io
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from emote import aigen, ui
from emote.sprite import EmoteError, extract_palette, load_rgba
from test_aigen import FakeClient


def data_url(path: str | Path, mime="png") -> str:
    return f"data:image/{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


@pytest.fixture
def server(tmp_path, monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(aigen, "server", lambda cfg: contextlib.nullcontext(fake))
    monkeypatch.setattr(aigen, "check_models", lambda cfg: None)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    app = ui.App("config.yaml", tmp_path / "jobs", port)
    app.work.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app, "ensure_comfy", lambda: None)
    monkeypatch.setattr(app, "comfy_alive", lambda: True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), ui.make_handler(app, b"<html>ok</html>"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}", app, fake
    httpd.shutdown(); httpd.server_close()


def call(base, path, body=None, headers=None, host=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    if host:
        h["Host"] = host
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), headers=h,
                                 method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


GOOD = {"character": "a pixel art pigeon, grey head, orange beak", "facing": "side view facing left", "seed": "7",
        "fps": 8, "poses": ["wings up", "wings half", "wings down", "wings half"]}


def test_index_and_state(server):
    base, app, _ = server
    code, body, _ = call(base, "/")
    assert code == 200 and b"ok" in body
    code, body, _ = call(base, "/api/state")
    state = json.loads(body)
    assert code == 200 and "Flap wings" in state["presets"] and state["comfy"] is True


def test_inspect_cleans_noisy_image_and_reports_palette(server):
    base, *_ = server
    code, body, _ = call(base, "/api/inspect", {"image": data_url("assets/original.png")})
    info = json.loads(body)
    assert code == 200 and info["colors"] <= 16 and any("Cleaned" in n for n in info["notes"])
    assert info["chroma"] in ([0, 255, 0], [255, 0, 255], [0, 0, 255]) and info["palette"][0].startswith("#")


def test_inspect_strips_flat_background_on_clean_sprite_without_alpha(tmp_path):
    a = np.zeros((12, 12, 4), np.uint8); a[..., 3] = 255; a[...] = (255, 255, 255, 255)
    a[3:9, 3:9] = (200, 30, 30, 255); a[5, 5] = (10, 10, 10, 255)
    p = tmp_path / "flat.png"; Image.fromarray(a).save(p)
    sprite, notes = ui.prepare_sprite(load_rgba(p))
    assert any("flat background" in n for n in notes)
    assert set(extract_palette(sprite)) == {(200, 30, 30), (10, 10, 10)}


def test_pick_chroma_avoids_colors_in_the_sprite():
    assert ui.pick_chroma([(0, 250, 0), (20, 20, 20)]) != (0, 255, 0)
    assert ui.pick_chroma([(173, 140, 120), (53, 49, 73)]) == (0, 255, 0)


def test_security_rejects_foreign_host_and_non_json_and_bad_input(server):
    base, *_ = server
    assert call(base, "/api/state", host="evil.example.com")[0] == 403
    req = urllib.request.Request(base + "/api/jobs", data=b"x=1", headers={"Content-Type": "text/plain"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 415
    assert call(base, "/api/jobs", {**GOOD, "image": "data:text/html;base64,AAAA"})[0] == 400
    assert call(base, "/api/jobs", {**GOOD, "image": data_url("assets/base.png"), "poses": ["a", "b"]})[0] == 400
    assert call(base, "/api/jobs", {**GOOD, "image": data_url("assets/base.png"), "fps": 500})[0] == 400
    assert call(base, "/api/jobs", {**GOOD, "image": data_url("assets/base.png"), "seed": "abc"})[0] == 400
    assert call(base, "/api/jobs", {**GOOD, "image": data_url("assets/base.png"), "facing": "upside down"})[0] == 400


def test_path_traversal_is_blocked(server):
    base, app, _ = server
    jid = "abcdef0123"
    (app.work / jid).mkdir()
    (app.work / "secret.txt").write_text("nope")
    app.jobs[jid] = {"id": jid, "stage": "done", "created": 0, "opts": {}, "files": {}, "stage_index": 5, "error": None}
    for rel in ("../secret.txt", "..%2Fsecret.txt", "%2e%2e/secret.txt"):
        code, body, _ = call(base, f"/jobs/{jid}/{rel}")
        assert code == 404 and b"nope" not in body


def test_full_job_through_http_produces_palette_only_outputs(server):
    base, app, fake = server
    code, body, _ = call(base, "/api/jobs", {**GOOD, "image": data_url("assets/base.png")})
    assert code == 200
    jid = json.loads(body)["id"]
    for _ in range(200):
        job = json.loads(call(base, f"/api/jobs/{jid}")[1])
        if job["stage"] == "done" or job["error"]:
            break
        time.sleep(0.1)
    assert job["error"] is None and job["stage"] == "done" and job["frames"] == 4
    assert fake.calls[0]["17"]["inputs"]["noise_seed"] == 7  # the seed from the form
    prompt = fake.calls[0]["10"]["inputs"]["text"]
    assert "a pixel art pigeon" in prompt and "Frame 3 (bottom-left): wings down" in prompt
    sheet = load_rgba(app.work / jid / job["files"]["spritesheet"])
    assert {tuple(int(v) for v in p[:3]) for p in sheet.reshape(-1, 4) if p[3]} <= set(
        extract_palette(load_rgba("assets/base.png")))
    code, zbody, hdr = call(base, f"/jobs/{jid}/download.zip")
    assert code == 200 and "sprite_sheet.png" in zipfile.ZipFile(io.BytesIO(zbody)).namelist()
    assert call(base, f"/jobs/{jid}/{job['files']['gif_112']}")[0] == 200


def test_bad_sprite_reports_clear_error_in_job(server):
    base, app, _ = server
    a = np.zeros((300, 300, 4), np.uint8); a[50:250, 50:250] = (10, 10, 10, 255)
    Image.fromarray(a).save(app.work / "big.png")
    jid = json.loads(call(base, "/api/jobs", {**GOOD, "image": data_url(app.work / "big.png")})[1])["id"]
    for _ in range(100):
        job = json.loads(call(base, f"/api/jobs/{jid}")[1])
        if job["error"] or job["stage"] == "done":
            break
        time.sleep(0.1)
    assert "limit" in job["error"]


def test_pick_canvas_leaves_headroom():
    # each of the 4 sheet cells must have ~1.6x the sprite's size, or the model shrinks the character
    assert ui.pick_canvas((27, 29)) == 128 and ui.pick_canvas((16, 16)) == 64 and ui.pick_canvas((80, 70)) == 256
    with pytest.raises(EmoteError):
        ui.pick_canvas((100, 10))


def test_inspect_returns_cleaned_preview_and_restore_box_validation(server):
    base, *_ = server
    info = json.loads(call(base, "/api/inspect", {"image": data_url("assets/base.png")})[1])
    prev = load_upload_bytes(info["preview"])
    assert prev.shape[:2] == (info["height"], info["width"])
    good = {**GOOD, "image": data_url("assets/base.png")}
    for bad in ([1, 2, 3], [5, 5, 2, 9], ["a", 1, 2, 3], [-1, 0, 4, 4], True):
        assert call(base, "/api/jobs", {**good, "restore_box": bad})[0] == 400
    assert call(base, "/api/jobs", {**good, "restore_box": [9, 5, 14, 10]})[0] == 200


def load_upload_bytes(data_url_str):
    return ui.load_upload(ui.decode_data_url(data_url_str))
