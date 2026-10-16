"""Single config.yaml loader with defaults. No hard-coded paths outside defaults."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .sprite import EmoteError

DEFAULTS: dict[str, Any] = {
    "paths": {
        "base": "assets/base.png",
        "regions": "assets/regions.json",
        "frames_dir": "frames",
        "out_dir": "out",
        "specs": "twitch_specs.json",
    },
    "process": {
        "source_scale": 8,      # frames are this many times larger than the native sprite
        "sheet": None,          # optional sprite-sheet path (split with rows x cols)
        "rows": 2,
        "cols": 2,
        "bg": {
            "mode": "chroma",   # chroma | ai | none
            "key": None,        # None = prep.chroma_color
            "tolerance": 12.0,  # Lab dE
            "alpha_cutoff": 128,
            "ai_model": "birefnet-general",
        },
        "snap": {
            "anchor": "feet",   # feet | bbox-bottom | none
            "feet_rows": 2,
            "canvas_pad": 0,    # int: transparent border (native px) around native size; "auto": fit frames
            "crop": True,       # crop all frames to their union bounding box (keeps registration)
            "crop_pad": 0,
            "despeckle": 0,     # drop opaque islands smaller than this many pixels (0 = off)
            "thin_outline": False,  # shave outlines thicker than 1 px (AI frames; the ai backend turns it on)
            "flatten": False,       # flatten dithered / speckled shading (AI frames; the ai backend turns it on)
        },
    },
    "restore": {  # paste exact pixels of the ORIGINAL sprite back onto AI frames (needs process.snap downscale)
        "regions": {},          # {"eye": [x0, y0, x1, y1]} in native sprite pixels, x1/y1 exclusive; {} = off
        "min_score": 0.7,       # fraction of the region's pixels that must already match; below = left untouched
        "tolerance": 12.0,      # Lab distance counted as "matching"
    },
    "run": {
        "backend": "procedural",  # procedural | ai (ComfyUI, optional phase 5)
    },
    "ai": {  # stage 2b: ComfyUI + FLUX.2 klein 4B image edit (optional backend)
        "comfy_dir": "../comfyui/app/ComfyUI_windows_portable",
        "host": "127.0.0.1",
        "port": 8188,
        "autostart": True,          # launch ComfyUI if it is not already running, stop it afterwards
        "timeout_s": 600,
        "workflow": "workflows/klein_edit.json",
        "unet": "flux-2-klein-4b-fp8.safetensors",
        "clip": "qwen_3_4b.safetensors",
        "vae": "flux2-vae.safetensors",
        "lora": None,               # e.g. flux-spritesheet-lora.safetensors
        "lora_strength": 1.0,
        # The sprite is centred on a canvas x canvas grid, scaled by (1024 / canvas). In sheet mode each pose
        # only gets a quarter of the image, so the canvas must be ~3.2x the sprite or the model shrinks the
        # character (and loses detail such as the eye highlight). 128 = 8 px blocks for a ~29 px sprite.
        "canvas": 128,
        "megapixels": 1.0,
        "steps": 4,                 # klein 4B distilled
        "seed": None,               # None = top-level seed
        "seed_step": 0,             # added per frame; 0 = same seed for every frame (best consistency)
        "anchor": "bbox-bottom",    # feet | bbox-bottom | none
        "mode": "sheet",            # sheet = one 2x2 sheet, 4 poses in one pass (consistent character);
                                    # frames = one edit per pose (character drifts more)
        "sheet_rows": 2,
        "sheet_cols": 2,
        "source_scale": None,       # model pixel-block size in px; None = the reference's scale (1024 / canvas)
        "despeckle": 4,             # drop stray AI pixel islands smaller than this (px)
        "thin_outline": True,       # the model draws ~2 px outlines; shave them to 1 px like pixel art
        "flatten": True,            # flatten the model's speckled / dithered shading to flat colours
        # Name every palette colour AND say where orange must not appear: without the last sentence the model
        # paints an orange chest, which the palette snap then turns coral.
        "character": ("a round, chubby pixel art pigeon with a muted grey-brown head (#a89993), a dusty "
                      "tan-brown chest (#ad8c78), cream-white belly and wings (#f2e9e7), dark navy outline "
                      "(#353149), a big round black eye with a white highlight. Only the small beak and the feet "
                      "are coral-orange (#dc6e62); the chest and body have NO orange or red"),
        "facing": "side view facing left",   # how the character is shown; e.g. "front view", "side view facing right"
        "prompt": ("{character}. Edit the image: {pose}. Keep this exact same character: the same colors, "
                   "the same round chubby proportions, the same big eye, the same orange beak, the same dark navy "
                   "outline, the same chunky pixel size, {facing}. Only change the pose. The character keeps a "
                   "complete, unbroken dark outline all the way around its silhouette, exactly one pixel thick "
                   "like the reference. "
                   "Flat solid {chroma} background, no shadow, no anti-aliasing."),
        "sheet_prompt": ("A 2x2 sprite sheet grid on a flat solid {chroma} background: four animation frames of "
                         "{character}. Every frame shows the exact same character with identical colors, "
                         "proportions, eye, beak, outline and chunky pixel size, {facing}, "
                         "same pixel art style as the reference image, no shadow, no anti-aliasing. Every figure has "
                         "a complete, unbroken dark outline all the way around its silhouette, exactly one pixel "
                         "thick like the reference. {cells}"),
        # up -> half -> folded down -> half: the only phrasing that got a real folded-wings frame (see README)
        "poses": [
            "the pigeon flaps its wings: wings fully raised straight up",
            "the pigeon flaps its wings: wings halfway up, angled up at 45 degrees",
            "the pigeon flaps its wings: wings folded down along the sides of its body",
            "the pigeon flaps its wings: wings halfway up, angled up at 45 degrees",
        ],
        "out": "frames/ai",
        "name": "emote_fly",        # output file base name for run.backend=ai (keeps the walk exports intact)
    },
    "animate": {
        "motion": "walk",       # walk | sleep
        "sleep": {              # closed eyes + breathing + Zzz (needs an 'eye' region in regions.json)
            "frames": 12,
            "breaths": 1,       # breaths per loop
            "breath_amp": 1,    # px the body settles down on the out-breath
            "head_nod": 1,      # extra px the head droops at the deepest point
            "zzz": True,
            "zzz_color": [220, 110, 98],    # must be a colour of the sprite; coral reads on dark AND light chat
            "margin": 2,
            "fps": 6,           # sleep is slow; `run` uses this instead of export.fps
        },
        "frames": 4,            # frames in the loop (phase i/N, so the loop is seamless)
        "facing": "left",
        "head_amp": 1,          # head thrust forward at foot contact, px
        "body_dip": 1,          # body dip on contact frames, px
        "leg_swing": 1,
        "leg_lift": 1,
        "margin": 2,            # transparent border so shifted pixels never clip
        "out": "frames/walk",
    },
    "export": {
        "name": "emote",
        "fps": 8,
        "sizes": [112, 56, 28],
        "integer_upscale": True,
        "spritesheet": {
            "enabled": True,        # <name>_sheet.png + <name>_sheet.json atlas (game-engine style)
            "cols": 0,              # 0 = all frames in one row
            "scale": 1,             # integer nearest-neighbor upscale of the sheet (1 = native pixels)
        },
    },
    "prep": {
        "upscale": 8,
        "chroma_color": [0, 255, 0],
        "alpha_cutoff": 128,
    },
    "clean": {
        "block": None,          # pixel-block size in source px; None = auto-detect
        "offset": None,         # [x, y] grid offset; None = auto-detect
        "colors": 16,           # max k-means clusters before merging
        "merge_delta_e": 6.0,   # merge clusters closer than this (Lab)
        "bg_tolerance": 5.0,    # background flood-fill tolerance (Lab dE)
        "remove_shadow": True,  # drop light blue/purple ground-shadow color
        "pad": 1,
    },
    "seed": 1234,
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise EmoteError(f"Bad --set '{item}'; use KEY=VALUE, e.g. animate.head_amp=2")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise EmoteError(f"--set {key}: '{part}' is not a config section")
            node = node[part]
        if parts[-1] not in node:
            raise EmoteError(f"--set {key}: unknown config key '{parts[-1]}'")
        node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path = "config.yaml", overrides: list[str] | None = None) -> dict[str, Any]:
    p = Path(path)
    data: dict = {}
    if p.is_file():
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise EmoteError(f"Invalid YAML in {p}: {exc}") from exc
        if not isinstance(data, dict):
            raise EmoteError(f"{p} must contain a mapping at the top level")
    cfg = _merge(copy.deepcopy(DEFAULTS), data)  # deep copy: overrides must never leak into DEFAULTS
    _apply_overrides(cfg, overrides or [])
    return cfg
