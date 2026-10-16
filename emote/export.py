"""Stage 6: resizing (nearest, centred, no stretching), APNG / GIF writing, contact sheet."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .sprite import EmoteError, RGB, upscale_nn


def fit_square(frame: np.ndarray, size: int, integer_upscale: bool = True) -> np.ndarray:
    """Fit into size x size, aspect preserved, centred on transparent padding. Nearest-neighbor only.

    Upscales use the largest integer factor (crisp pixels) when `integer_upscale`; downscales
    (e.g. 29 px -> 28 px) necessarily use nearest sampling at a non-integer ratio.
    """
    h, w = frame.shape[:2]
    if integer_upscale and size >= max(w, h):
        k = size // max(w, h)
        scaled = upscale_nn(frame, k)
    else:
        s = size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        scaled = np.array(Image.fromarray(frame, "RGBA").resize((nw, nh), Image.NEAREST))
    sh, sw = scaled.shape[:2]
    canvas = np.zeros((size, size, 4), np.uint8)
    y0, x0 = (size - sh) // 2, (size - sw) // 2
    canvas[y0:y0 + sh, x0:x0 + sw] = scaled
    return canvas


def gif_duration_ms(fps: float) -> int:
    """GIF delays are whole centiseconds; returns the nearest representable frame time in ms."""
    return max(20, int(round(100.0 / fps)) * 10)  # browsers clamp delays < 20 ms


def save_apng(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(f, "RGBA") for f in frames]
    ims[0].save(path, format="PNG", save_all=True, append_images=ims[1:], duration=round(1000 / fps),
                loop=0, disposal=1, blend=0)
    return path


def save_gif(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    """Transparent GIF using the exact colors present (no dithering, no palette drift)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.unique(np.concatenate([f[f[..., 3] > 0][:, :3] for f in frames]), axis=0)
    if len(colors) > 255:
        raise EmoteError(f"GIF needs <= 255 colors, frames contain {len(colors)}")
    lut = {tuple(int(v) for v in c): i + 1 for i, c in enumerate(colors)}  # index 0 = transparent
    palette = [0, 0, 0] + [int(v) for c in colors for v in c]
    ims = []
    for f in frames:
        idx = np.zeros(f.shape[:2], np.uint8)
        ys, xs = np.nonzero(f[..., 3])
        for y, x in zip(ys, xs):
            idx[y, x] = lut[tuple(int(v) for v in f[y, x, :3])]
        im = Image.fromarray(idx, "P")
        im.putpalette(palette)
        ims.append(im)
    ims[0].save(path, format="GIF", save_all=True, append_images=ims[1:], duration=gif_duration_ms(fps),
                loop=0, transparency=0, disposal=2, optimize=False)
    return path


def save_spritesheet(frames: list[np.ndarray], png_path: Path, fps: float, cols: int = 0, scale: int = 1) -> Path:
    """Frames packed row-major into one transparent PNG (+ `<name>.json` atlas next to it).

    `cols` = 0 packs all frames in one row. `scale` is an integer nearest-neighbor upscale (1 = native pixels).
    Cells are the largest frame size; smaller frames are centred, so every cell has identical dimensions.
    """
    import json

    n = len(frames)
    cols = n if cols <= 0 else min(cols, n)
    rows = -(-n // cols)
    cw, ch = max(f.shape[1] for f in frames), max(f.shape[0] for f in frames)
    sheet = np.zeros((rows * ch, cols * cw, 4), np.uint8)
    atlas = []
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = r * ch + (ch - f.shape[0]) // 2, c * cw + (cw - f.shape[1]) // 2
        sheet[y:y + f.shape[0], x:x + f.shape[1]] = f
        atlas.append({"index": i, "x": c * cw * scale, "y": r * ch * scale, "w": cw * scale, "h": ch * scale})
    if scale > 1:
        sheet = upscale_nn(sheet, scale)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, "RGBA").save(png_path)
    png_path.with_suffix(".json").write_text(json.dumps({
        "image": png_path.name, "frame_width": cw * scale, "frame_height": ch * scale, "columns": cols, "rows": rows,
        "frame_count": n, "fps": fps, "loop": True, "frames": atlas}, indent=2), encoding="utf-8")
    return png_path


def _hex_rgb(h: str) -> RGB:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _composite(frame: np.ndarray, bg: RGB) -> np.ndarray:
    out = np.empty(frame.shape[:2] + (3,), np.uint8)
    out[...] = bg
    m = frame[..., 3] > 0
    out[m] = frame[m][:, :3]
    return out


def contact_sheet(frames: list[np.ndarray], path: Path, small: int = 28, big: int | None = None,
                  dark: str = "#18181b", light: str = "#efeff1", zoom: int = 4, gap: int = 8) -> Path:
    """Rows: 28 px on dark, 28 px on light (each with a zoomed copy), then full size on dark."""
    nat_h, nat_w = frames[0].shape[:2]
    big_scale = big or max(1, 112 // max(nat_w, nat_h))
    rows = []
    for bg in (dark, light):
        a = [_composite(fit_square(f, small), _hex_rgb(bg)) for f in frames]
        z = [np.repeat(np.repeat(x, zoom, 0), zoom, 1) for x in a]
        rows.append((bg, a, z))
    full = [_composite(upscale_nn(f, big_scale), _hex_rgb(dark)) for f in frames]

    def strip(imgs: list[np.ndarray], bg: str) -> np.ndarray:
        h = max(i.shape[0] for i in imgs)
        w = sum(i.shape[1] for i in imgs) + gap * (len(imgs) + 1)
        s = np.empty((h + 2 * gap, w, 3), np.uint8)
        s[...] = _hex_rgb(bg)
        x = gap
        for i in imgs:
            s[gap + (h - i.shape[0]) // 2:gap + (h - i.shape[0]) // 2 + i.shape[0], x:x + i.shape[1]] = i
            x += i.shape[1] + gap
        return s

    strips = []
    for bg, a, z in rows:
        strips.append(strip(a + z, bg))
    strips.append(strip(full, dark))
    width = max(s.shape[1] for s in strips)
    sheet = np.zeros((sum(s.shape[0] for s in strips), width, 3), np.uint8)
    y = 0
    for s in strips:
        sheet[y:y + s.shape[0], :s.shape[1]] = s
        sheet[y:y + s.shape[0], s.shape[1]:] = s[0, 0]
        y += s.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, "RGB").save(path)
    return path
