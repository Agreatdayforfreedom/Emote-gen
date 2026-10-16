"""Check outputs against twitch_specs.json and produce warnings."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .sprite import EmoteError


def load_specs(path: str | Path = "twitch_specs.json") -> dict:
    p = Path(path)
    if not p.is_file():
        raise EmoteError(f"{p} not found; it records Twitch limits with source and date")
    return json.loads(p.read_text(encoding="utf-8"))


def check_gif(path: Path, size_px: int, specs: dict) -> list[str]:
    a = specs["animated"]
    warns: list[str] = []
    nbytes = path.stat().st_size
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        w, h = im.size
    if w != h:
        warns.append(f"{path.name}: not square ({w}x{h})")
    if size_px in a["manual_sizes_px"] and (w, h) != (size_px, size_px):
        warns.append(f"{path.name}: expected {size_px}x{size_px}, got {w}x{h}")
    if nbytes > a["max_file_bytes_manual_each"]:
        warns.append(f"{path.name}: {nbytes / 1024:.0f} KB exceeds {a['max_file_bytes_manual_each'] // 1024} KB per-size limit")
    if n > a["max_frames"]:
        warns.append(f"{path.name}: {n} frames exceeds max {a['max_frames']}")
    return warns

