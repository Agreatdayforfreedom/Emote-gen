"""Stage 2b: AI pose frames via a local ComfyUI server (FLUX.2 klein 4B image edit).

For every pose in `ai.poses` the sprite is sent as a reference image (integer-upscaled onto a flat chroma
background, on a square canvas that scales exactly to ~1 MP) with a fixed prompt template and a fixed seed.
Raw outputs are kept in <out>/raw_XX.png; `process` then removes the background, votes each block down to the
native grid, snaps to the base palette and aligns the frames. Everything is deterministic given the config
(workflow, models, prompt, seed) on the same ComfyUI build.
"""
from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from .comfy import ComfyClient, fill_workflow, launch_server
from .split import split_sheet
from .sprite import EmoteError, load_rgba, on_background, save_png, upscale_nn


def reference_image(cfg: dict) -> tuple[np.ndarray, int]:
    """Base sprite centred on an N x N chroma-key canvas, upscaled by an integer so the result is ~1 MP."""
    ac = cfg["ai"]
    n = ac["canvas"]
    base = load_rgba(cfg["paths"]["base"])
    h, w = base.shape[:2]
    if max(h, w) > n:
        raise EmoteError(f"ai.canvas={n} is smaller than the sprite ({w}x{h}); raise it")
    side = int(round((ac["megapixels"] * 1024 * 1024) ** 0.5))  # ComfyUI counts 1 MP as 1024 x 1024
    if side % n:
        raise EmoteError(f"ai.canvas={n} must divide the working size {side}px (megapixels={ac['megapixels']}) "
                         "so the reference scales by a whole number; try 32, 64 or 128 with megapixels 1.0")
    canvas = np.zeros((n, n, 4), np.uint8)
    y0, x0 = n - h - (n - h) // 4, (n - w) // 2  # bottom-weighted: leaves headroom for raised wings
    canvas[y0:y0 + h, x0:x0 + w] = base
    k = side // n
    return on_background(upscale_nn(canvas, k), tuple(cfg["prep"]["chroma_color"])), k


@contextmanager
def server(cfg: dict) -> Iterator[ComfyClient]:
    ac = cfg["ai"]
    client = ComfyClient(ac["host"], ac["port"], timeout=ac["timeout_s"])
    proc = None
    if not client.alive():
        if not ac["autostart"]:
            raise EmoteError(f"ComfyUI is not running at {client.base} and ai.autostart is false")
        print(f"  starting ComfyUI from {ac['comfy_dir']} ...")
        proc = launch_server(ac["comfy_dir"], ac["host"], ac["port"])
    try:
        yield client
    finally:
        if proc is not None:
            proc.terminate()


def check_models(cfg: dict) -> None:
    root = Path(cfg["ai"]["comfy_dir"]) / "ComfyUI" / "models"
    wanted = {"diffusion_models": cfg["ai"]["unet"], "text_encoders": cfg["ai"]["clip"], "vae": cfg["ai"]["vae"]}
    if cfg["ai"]["lora"]:
        wanted["loras"] = cfg["ai"]["lora"]
    missing = [f"{d}/{f}" for d, f in wanted.items() if not (root / d / f).is_file()]
    if missing:
        raise EmoteError(f"Missing ComfyUI model files under {root}: {', '.join(missing)}")


CELL_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]


CHROMA_NAMES = {(0, 255, 0): "bright green (#00FF00)", (255, 0, 255): "bright magenta (#FF00FF)",
                (0, 0, 255): "bright blue (#0000FF)"}


def chroma_name(cfg: dict) -> str:
    c = tuple(int(v) for v in cfg["prep"]["chroma_color"])
    return CHROMA_NAMES.get(c, "flat color #%02X%02X%02X" % c)


def pose_prompt(cfg: dict, pose: str) -> str:
    ac = cfg["ai"]
    return ac["prompt"].format(pose=pose, character=ac["character"], facing=ac["facing"], chroma=chroma_name(cfg))


def sheet_prompt(cfg: dict) -> str:
    ac = cfg["ai"]
    if len(ac["poses"]) != 4 or ac["sheet_rows"] != 2 or ac["sheet_cols"] != 2:
        raise EmoteError("ai.mode 'sheet' currently supports exactly 4 poses on a 2x2 grid")
    cells = " ".join(f"Frame {i + 1} ({CELL_NAMES[i]}): {p}." for i, p in enumerate(ac["poses"]))
    return ac["sheet_prompt"].format(cells=cells, character=ac["character"], facing=ac["facing"],
                                     chroma=chroma_name(cfg))


def _params(cfg: dict, image: str, prompt: str, seed: int) -> dict:
    ac = cfg["ai"]
    return {"image": image, "prompt": prompt, "seed": seed, "steps": ac["steps"], "megapixels": ac["megapixels"],
            "lora_name": ac["lora"] or "none", "lora_strength": ac["lora_strength"] if ac["lora"] else 0.0}


def _run(client: ComfyClient, wf: dict, params: dict, lora: bool) -> np.ndarray:
    wfi = fill_workflow(wf, params)
    if not lora:  # bypass the LoRA node entirely
        wfi["16"]["inputs"]["model"] = ["1", 0]
        del wfi["15"]
    return client.run(wfi, node_id="20")[0]


def generate(cfg: dict, out_dir: str | Path | None = None) -> Path:
    """Generate the pose frames into `out_dir` (default ai.out) and write the meta.json for `process`."""
    ac = cfg["ai"]
    check_models(cfg)
    wf = json.loads(Path(ac["workflow"]).read_text(encoding="utf-8"))
    out = Path(out_dir or ac["out"])
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"  # unprocessed inputs/outputs kept out of the folder `process` reads
    for old in [*out.glob("frame_*.png"), *raw.glob("*.png")]:
        old.unlink()
    seed0 = int(ac["seed"] if ac["seed"] is not None else cfg["seed"])

    ref, ref_scale = reference_image(cfg)
    ref_path = save_png(ref, raw / "reference.png")
    frames: list[np.ndarray] = []
    with server(cfg) as client:
        name = client.upload_image(ref_path, "emote_reference.png")
        if ac["mode"] == "sheet":
            prompt = sheet_prompt(cfg)
            print(f"  one 2x2 sheet (seed {seed0})")
            sheet = _run(client, wf, _params(cfg, name, prompt, seed0), bool(ac["lora"]))
            save_png(sheet, raw / "sheet.png")
            frames = split_sheet(sheet, ac["sheet_rows"], ac["sheet_cols"])
        elif ac["mode"] == "frames":
            cache: dict[tuple[str, int], np.ndarray] = {}  # identical pose + seed -> identical frame
            for i, pose in enumerate(ac["poses"]):
                seed = seed0 + i * ac["seed_step"]
                key = (pose, seed)
                if key not in cache:
                    print(f"  [{i + 1}/{len(ac['poses'])}] {pose}  (seed {seed})")
                    cache[key] = _run(client, wf, _params(cfg, name, pose_prompt(cfg, pose), seed), bool(ac["lora"]))
                else:
                    print(f"  [{i + 1}/{len(ac['poses'])}] {pose}  (same as an earlier frame, reused)")
                frames.append(cache[key])
        else:
            raise EmoteError(f"ai.mode must be 'frames' or 'sheet', got {ac['mode']!r}")
    for i, img in enumerate(frames):
        save_png(img, out / f"frame_{i:02d}.png")
    scale = ac["source_scale"] or ref_scale  # the model copies the reference's pixel-block size
    print(f"  pixel-block size used to snap back to the grid: {scale}px")
    (out / "meta.json").write_text(json.dumps({
        "source_scale": scale, "bg_mode": "chroma", "anchor": ac["anchor"],
        "canvas_pad": "auto", "downscale": "vote", "despeckle": ac["despeckle"],
        "thin_outline": ac["thin_outline"], "flatten": ac["flatten"]}, indent=2), encoding="utf-8")
    return out
