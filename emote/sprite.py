"""Sprite loading, palette extraction and nearest-neighbor upscaling (no interpolation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

RGB = tuple[int, int, int]


class EmoteError(Exception):
    """User-facing error with a clear message."""


def load_rgba(path: str | Path) -> np.ndarray:
    """Load an image as an (H, W, 4) uint8 RGBA array."""
    p = Path(path)
    if not p.is_file():
        raise EmoteError(f"Image not found: {p}")
    try:
        with Image.open(p) as im:
            return np.array(im.convert("RGBA"), dtype=np.uint8)
    except OSError as exc:
        raise EmoteError(f"Cannot read image {p}: {exc}") from exc


def extract_palette(rgba: np.ndarray, alpha_cutoff: int = 128) -> list[RGB]:
    """Unique colors of opaque pixels, ordered by descending pixel count."""
    opaque = rgba[rgba[..., 3] >= alpha_cutoff][:, :3]
    if opaque.size == 0:
        raise EmoteError("Sprite has no opaque pixels; is the background fully transparent?")
    colors, counts = np.unique(opaque, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return [tuple(int(v) for v in colors[i]) for i in order]


def palette_counts(rgba: np.ndarray, alpha_cutoff: int = 128) -> list[tuple[RGB, int]]:
    opaque = rgba[rgba[..., 3] >= alpha_cutoff][:, :3]
    colors, counts = np.unique(opaque, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return [(tuple(int(v) for v in colors[i]), int(counts[i])) for i in order]


def semi_transparent_pixels(rgba: np.ndarray) -> int:
    a = rgba[..., 3]
    return int(np.count_nonzero((a > 0) & (a < 255)))


def bounding_box(rgba: np.ndarray, alpha_cutoff: int = 128) -> tuple[int, int, int, int] | None:
    """Tight (x0, y0, x1, y1) box (x1/y1 exclusive) around opaque pixels."""
    ys, xs = np.nonzero(rgba[..., 3] >= alpha_cutoff)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def upscale_nn(rgba: np.ndarray, factor: int) -> np.ndarray:
    """Integer nearest-neighbor upscale by pixel repetition."""
    if factor < 1:
        raise EmoteError(f"Upscale factor must be >= 1, got {factor}")
    return np.repeat(np.repeat(rgba, factor, axis=0), factor, axis=1)


def on_background(rgba: np.ndarray, color: RGB) -> np.ndarray:
    """Composite onto a flat opaque background using binary alpha (no blending)."""
    out = np.empty_like(rgba)
    out[...] = (*color, 255)
    mask = rgba[..., 3] >= 128
    out[mask] = rgba[mask]
    out[mask, 3] = 255
    return out


def save_png(rgba: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(p)
    return p
