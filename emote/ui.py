"""Local web UI: upload a sprite, describe it, pick four poses, generate -> GIF + spritesheet.

Pure stdlib (http.server). Binds to 127.0.0.1 only, rejects foreign Host headers and non-JSON POSTs (so a web page
on another origin cannot drive it), and serves only files inside a job's own folder. One worker thread runs the
jobs one at a time (they share the GPU); ComfyUI is started on first use and kept warm until the UI exits.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import math
import mimetypes
import queue
import random
import re
import threading
import time
import traceback
import urllib.parse
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from . import aigen
from .clean import _flood_background, clean_sprite
from .color import delta_e, rgb_to_lab
from .comfy import ComfyClient, launch_server
from .config import load_config
from .pipeline import process
from .sprite import EmoteError, extract_palette, save_png

MAX_UPLOAD = 8 * 1024 * 1024
MAX_SPRITE = 80           # px; in sheet mode the reference canvas must be ~3.2x the sprite (max canvas 256)
NOISY_COLORS = 64         # more unique colors than this = not a clean pixel sprite
CHROMA_CANDIDATES = [(0, 255, 0), (255, 0, 255), (0, 0, 255)]

PRESETS = {
    "Flap wings": ["the character flaps its wings: wings fully raised straight up",
                   "the character flaps its wings: wings halfway up, angled up at 45 degrees",
                   "the character flaps its wings: wings folded down along the sides of its body",
                   "the character flaps its wings: wings halfway up, angled up at 45 degrees"],
    "Walk": ["the character walks: left foot stepping forward, body slightly low",
             "the character walks: feet passing each other under the body, body slightly raised",
             "the character walks: right foot stepping forward, body slightly low",
             "the character walks: feet passing each other under the body, body slightly raised"],
    "Jump": ["the character crouches down, knees bent, getting ready to jump",
             "the character launches upward, body stretched, feet leaving the ground",
             "the character is at the top of the jump in mid-air, tucked up",
             "the character lands, crouching slightly, absorbing the impact"],
    "Idle bounce": ["the character stands still in a relaxed neutral pose",
                    "the character squashes down slightly, body a little shorter and wider",
                    "the character stands still in a relaxed neutral pose",
                    "the character stretches up slightly, body a little taller and narrower"],
    "Wave": ["the character raises one arm or wing up high to wave hello",
             "the character waves, arm or wing tilted to one side",
             "the character raises one arm or wing up high to wave hello",
             "the character waves, arm or wing tilted to the other side"],
}
FACINGS = ["side view facing left", "side view facing right", "front view", "three-quarter view"]

GENERIC_PROMPTS = {
    "prompt": ("{character}. Edit the image: {pose}. Keep this exact same character: the same colors, the same "
               "proportions, the same face and details, the same outline, the same chunky pixel size, {facing}. "
               "Only change the pose. The character keeps a complete, unbroken dark outline all the way around its "
               "silhouette, exactly one pixel thick like the reference. "
               "Flat solid {chroma} background, no shadow, no anti-aliasing."),
    "sheet_prompt": ("A 2x2 sprite sheet grid on a flat solid {chroma} background: four animation frames of "
                     "{character}. Every frame shows the exact same character with identical colors, proportions, "
                     "details, outline and chunky pixel size, {facing}, same pixel art style as the reference "
                     "image, no shadow, no anti-aliasing. Every figure has a complete, unbroken dark outline all "
                     "the way around its silhouette, exactly one pixel thick like the reference. {cells}"),
}
STAGES = ["queued", "preparing", "starting ComfyUI", "generating", "processing", "done"]


# ---------------------------------------------------------------- image helpers
def decode_data_url(data: str) -> bytes:
    m = re.fullmatch(r"data:image/(?:png|jpeg|gif|webp);base64,([A-Za-z0-9+/=\s]+)", data or "")
    if not m:
        raise EmoteError("Upload a PNG, JPEG, GIF or WebP image")
    try:
        raw = base64.b64decode(m.group(1), validate=False)
    except binascii.Error as exc:
        raise EmoteError("The image data is corrupt") from exc
    if len(raw) > MAX_UPLOAD:
        raise EmoteError(f"Image is larger than {MAX_UPLOAD // (1024 * 1024)} MB")
    return raw


def load_upload(raw: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            return np.array(im.convert("RGBA"), dtype=np.uint8)
    except Exception as exc:  # PIL raises many types for bad data
        raise EmoteError(f"Cannot read that image: {exc}") from exc


def strip_flat_background(rgba: np.ndarray) -> np.ndarray:
    """No-transparency sprite on a flat colour: make the border-connected exact-colour region transparent."""
    corner = rgba[0, 0, :3]
    same = (rgba[..., :3] == corner).all(axis=-1)
    bg = _flood_background(same)
    out = rgba.copy()
    out[bg] = 0
    return out


def pick_chroma(palette: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """The key colour that is perceptually farthest from every colour in the sprite."""
    pal = rgb_to_lab(np.array(palette, np.uint8))
    best = max(CHROMA_CANDIDATES, key=lambda c: float(delta_e(pal, rgb_to_lab(np.array(c))).min()))
    return best


def prepare_sprite(rgba: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return a clean native-resolution RGBA sprite on a transparent background + notes about what was done."""
    notes: list[str] = []
    n_colors = len(np.unique(rgba[rgba[..., 3] >= 128][:, :3], axis=0))
    has_alpha = bool((rgba[..., 3] < 128).any())
    if n_colors > NOISY_COLORS:
        sprite, grid = clean_sprite(rgba[..., :3], colors=16, pad=1)
        notes.append(f"Cleaned a noisy image ({n_colors} colors) down to its pixel grid "
                     f"(block {grid.period_x:.1f}px) and a small palette")
    elif not has_alpha:
        sprite = strip_flat_background(rgba)
        notes.append("Removed the flat background color")
    else:
        sprite = rgba.copy()
        sprite[sprite[..., 3] < 128] = 0
        sprite[..., 3] = np.where(sprite[..., 3] >= 128, 255, 0)
    ys, xs = np.nonzero(sprite[..., 3])
    if xs.size == 0:
        raise EmoteError("Nothing is left after removing the background")
    y0, y1, x0, x1 = max(ys.min() - 1, 0), ys.max() + 2, max(xs.min() - 1, 0), xs.max() + 2
    sprite = sprite[y0:y1, x0:x1]
    if max(sprite.shape[:2]) > MAX_SPRITE:
        raise EmoteError(f"Sprite is {sprite.shape[1]}x{sprite.shape[0]}px; the limit is {MAX_SPRITE}px. "
                         "Downscale it to its true pixel size first")
    return sprite, notes


def inspect_upload(raw: bytes) -> dict:
    sprite, notes = prepare_sprite(load_upload(raw))
    pal = extract_palette(sprite)
    hexes = ["#%02x%02x%02x" % c for c in pal]
    desc = ("a pixel art character (describe it here: what it is, its main parts and colors). Palette: "
            + ", ".join(hexes[:8]))
    buf = io.BytesIO()
    Image.fromarray(sprite, "RGBA").save(buf, format="PNG")
    return {"width": int(sprite.shape[1]), "height": int(sprite.shape[0]), "colors": len(pal), "palette": hexes,
            "notes": notes, "suggested_character": desc, "chroma": list(pick_chroma(pal)),
            "preview": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


def pick_canvas(sprite_hw: tuple[int, int]) -> int:
    """Smallest reference canvas (divides 1024) where each of the 4 sheet cells has ~1.6x the sprite's size.

    Too small a canvas makes the model shrink the character to fit its cell (fewer pixels, lost detail)."""
    need = math.ceil(max(sprite_hw) * 3.2)
    for n in (64, 128, 256):
        if n >= need:
            return n
    raise EmoteError(f"Sprite too large for the reference canvas ({max(sprite_hw)}px; limit {MAX_SPRITE}px)")


# ---------------------------------------------------------------- job model
def validate_request(req: dict) -> dict:
    poses = req.get("poses")
    if not (isinstance(poses, list) and len(poses) == 4 and all(isinstance(p, str) and p.strip() for p in poses)):
        raise EmoteError("Provide exactly four non-empty pose descriptions")
    if any(len(p) > 300 for p in poses):
        raise EmoteError("Each pose description must be at most 300 characters")
    character = str(req.get("character", "")).strip()
    if not 5 <= len(character) <= 1200:
        raise EmoteError("Describe the character (5 to 1200 characters)")
    facing = str(req.get("facing", FACINGS[0]))
    if facing not in FACINGS:
        raise EmoteError(f"facing must be one of {FACINGS}")
    seed_raw = req.get("seed")
    try:
        seed = random.randint(0, 2**31 - 1) if seed_raw in (None, "") else int(seed_raw)
    except (TypeError, ValueError) as exc:
        raise EmoteError("Seed must be a whole number (or empty for random)") from exc
    if not 0 <= seed < 2**53:
        raise EmoteError("Seed must be between 0 and 2^53")
    try:
        fps = float(req.get("fps", 8))
    except (TypeError, ValueError) as exc:
        raise EmoteError("fps must be a number") from exc
    if not 1 <= fps <= 30:
        raise EmoteError("fps must be between 1 and 30")
    box = req.get("restore_box")
    if box is not None:
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) and not isinstance(v, bool)
                                                                  for v in box)):
            raise EmoteError("restore_box must be four whole numbers [x0, y0, x1, y1]")
        if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
            raise EmoteError("restore_box must have x0 < x1 and y0 < y1, both non-negative")
    return {"poses": [p.strip() for p in poses], "character": character, "facing": facing, "seed": seed, "fps": fps,
            "restore_box": box}


class App:
    def __init__(self, config_path: str, work_dir: Path, port: int):
        self.config_path = config_path
        self.work = Path(work_dir).resolve()
        self.port = port
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.comfy_proc = None
        self.base_cfg = load_config(config_path)
        threading.Thread(target=self._worker, daemon=True).start()

    # -- ComfyUI management
    def comfy_client(self) -> ComfyClient:
        ac = self.base_cfg["ai"]
        return ComfyClient(ac["host"], ac["port"], timeout=ac["timeout_s"])

    def comfy_alive(self) -> bool:
        return self.comfy_client().alive()

    def ensure_comfy(self) -> None:
        if self.comfy_alive():
            return
        ac = self.base_cfg["ai"]
        self.comfy_proc = launch_server(ac["comfy_dir"], ac["host"], ac["port"])

    def shutdown(self) -> None:
        if self.comfy_proc is not None:
            self.comfy_proc.terminate()

    # -- jobs
    def submit(self, req: dict) -> str:
        opts = validate_request(req)
        raw = decode_data_url(req.get("image", ""))
        load_upload(raw)  # fail fast on a bad image
        jid = uuid.uuid4().hex[:10]
        folder = self.work / jid
        folder.mkdir(parents=True)
        (folder / "upload.bin").write_bytes(raw)
        self.jobs[jid] = {"id": jid, "stage": "queued", "stage_index": 0, "error": None, "files": {}, "opts": opts,
                          "created": time.time(), "notes": []}
        self.q.put(jid)
        return jid

    def _set(self, jid: str, stage: str) -> None:
        self.jobs[jid]["stage"] = stage
        self.jobs[jid]["stage_index"] = STAGES.index(stage)

    def _worker(self) -> None:
        while True:
            jid = self.q.get()
            try:
                self._run(jid)
            except EmoteError as exc:
                self.jobs[jid]["error"] = str(exc)
            except Exception as exc:  # unexpected: show a message, keep the traceback in the console
                traceback.print_exc()
                self.jobs[jid]["error"] = f"Unexpected error: {exc}"

    def _run(self, jid: str) -> None:
        job, folder = self.jobs[jid], self.work / jid
        o = job["opts"]
        self._set(jid, "preparing")
        sprite, notes = prepare_sprite(load_upload((folder / "upload.bin").read_bytes()))
        job["notes"] = notes
        base_path = save_png(sprite, folder / "base.png")
        pal = extract_palette(sprite)
        cfg = load_config(self.config_path)
        chroma = pick_chroma(pal)
        cfg["paths"]["base"] = str(base_path)
        cfg["paths"]["out_dir"] = str(folder / "out")
        cfg["prep"]["chroma_color"] = list(chroma)
        cfg["ai"].update(mode="sheet", poses=o["poses"], character=o["character"], facing=o["facing"],
                         seed=o["seed"], canvas=pick_canvas(sprite.shape[:2]), out=str(folder / "frames"),
                         prompt=GENERIC_PROMPTS["prompt"], sheet_prompt=GENERIC_PROMPTS["sheet_prompt"])
        cfg["export"].update(fps=o["fps"], name="sprite")
        h, w = sprite.shape[:2]
        box = o["restore_box"]
        if box and (box[2] > w or box[3] > h):
            raise EmoteError(f"The kept-exact region {box} is outside the {w}x{h} sprite")
        cfg["restore"]["regions"] = {"detail": box} if box else {}  # replaces the pigeon defaults from config.yaml
        self._set(jid, "starting ComfyUI")
        with self.lock:
            self.ensure_comfy()
            self._set(jid, "generating")
            frames_dir = aigen.generate(cfg, folder / "frames")
        self._set(jid, "processing")
        res = process(cfg, frames_dir, out_dir=folder / "out")
        job["warnings"] = res["warnings"]
        job["restore_notes"] = res["notes"]
        job["frames"] = res["frames"]
        job["fps_effective"] = round(res["fps_gif_effective"], 2)
        job["files"] = {
            "reference": "frames/raw/reference.png", "sheet_raw": "frames/raw/sheet.png",
            "gif_112": "out/sprite_112.gif", "gif_56": "out/sprite_56.gif", "gif_28": "out/sprite_28.gif",
            "gif_native": "out/sprite_native.gif", "spritesheet": "out/sprite_sheet.png",
            "atlas": "out/sprite_sheet.json", "preview": "out/sprite_preview.png",
            "apng_112": "out/sprite_112.png"}
        job["sizes"] = {k: (folder / v).stat().st_size for k, v in job["files"].items() if (folder / v).is_file()}
        self._set(jid, "done")

    def zip_job(self, jid: str) -> bytes:
        folder = self.work / jid / "out"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(folder).as_posix())
        return buf.getvalue()

    def public_job(self, jid: str) -> dict:
        j = self.jobs[jid]
        return {k: v for k, v in j.items() if k != "created"} | {"stages": STAGES}


# ---------------------------------------------------------------- http
def make_handler(app: App, index_html: bytes):
    allowed_hosts = {f"127.0.0.1:{app.port}", f"localhost:{app.port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "emote-ui"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _host_ok(self) -> bool:
            if self.headers.get("Host", "") in allowed_hosts:
                return True
            self._json(403, {"error": "Forbidden host"})
            return False

        def do_GET(self):
            if not self._host_ok():
                return
            path = urllib.parse.urlparse(self.path).path
            if path == "/":
                page = index_html() if callable(index_html) else index_html
                return self._send(200, page, "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(200, {"comfy": app.comfy_alive(), "presets": PRESETS, "facings": FACINGS,
                                        "comfy_url": f"http://{app.base_cfg['ai']['host']}:{app.base_cfg['ai']['port']}",
                                        "jobs": [app.public_job(j) for j in sorted(
                                            app.jobs, key=lambda k: app.jobs[k]["created"], reverse=True)[:12]]})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{10})", path)
            if m:
                return self._json(200, app.public_job(m.group(1))) if m.group(1) in app.jobs \
                    else self._json(404, {"error": "Unknown job"})
            m = re.fullmatch(r"/jobs/([0-9a-f]{10})/(.+)", path)
            if m and m.group(1) in app.jobs:
                jid, rel = m.group(1), urllib.parse.unquote(m.group(2))
                if rel == "download.zip":
                    if app.jobs[jid]["stage"] != "done":
                        return self._json(409, {"error": "Job not finished"})
                    return self._send(200, app.zip_job(jid), "application/zip",
                                      {"Content-Disposition": f'attachment; filename="sprite_{jid}.zip"'})
                root = (app.work / jid).resolve()
                target = (root / rel).resolve()
                if root not in target.parents or not target.is_file():
                    return self._json(404, {"error": "Not found"})
                return self._send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0]
                                  or "application/octet-stream")
            self._json(404, {"error": "Not found"})

        def do_POST(self):
            if not self._host_ok():
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                return self._json(415, {"error": "Content-Type must be application/json"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"error": "Bad Content-Length"})
            if n <= 0 or n > MAX_UPLOAD * 2:
                return self._json(413, {"error": "Request too large"})
            try:
                req = json.loads(self.rfile.read(n))
                if not isinstance(req, dict):
                    raise ValueError
            except ValueError:
                return self._json(400, {"error": "Body must be a JSON object"})
            try:
                path = urllib.parse.urlparse(self.path).path
                if path == "/api/inspect":
                    return self._json(200, inspect_upload(decode_data_url(req.get("image", ""))))
                if path == "/api/jobs":
                    return self._json(200, {"id": app.submit(req)})
                if path == "/api/comfy/start":
                    threading.Thread(target=app.ensure_comfy, daemon=True).start()
                    return self._json(200, {"starting": True})
            except EmoteError as exc:
                return self._json(400, {"error": str(exc)})
            self._json(404, {"error": "Not found"})

    return Handler


def serve(config_path: str = "config.yaml", port: int = 8765, work_dir: str = "ui_jobs",
          open_browser: bool = True) -> None:
    page = Path(__file__).parent / "ui_page.html"
    app = App(config_path, Path(work_dir), port)
    app.work.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, page.read_bytes))  # re-read per request
    url = f"http://127.0.0.1:{port}"
    print(f"Sprite animator UI: {url}   (Ctrl+C to stop; jobs are saved in {app.work})")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.shutdown()
