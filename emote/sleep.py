"""Sleep animation from pixel edits only: closed eyes, slow breathing, rising Zzz. No new colors except the
(configurable) Zzz color, which must be a color of the sprite's palette.

Per frame i of N:
  * breathing: body+head settle down by up to `breath_amp` px and rise again `breaths` times per loop (feet stay
    planted, the body slides behind them); at the deepest point the head nods `head_nod` px further.
  * Zzz: three z glyphs (3, 4 and 5 px) travel from beside the head up and to the right, each a third of a loop
    apart, growing as they go and vanishing at the end; glyphs are drawn only on transparent pixels.
Phases are i / N, so frame N would equal frame 0 (seamless loop).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .animate import _paste, split_layers
from .color import rgb_to_lab
from .regions import Box
from .sprite import EmoteError, RGB

GLYPHS = {
    3: ["###", ".#.", "###"],
    4: ["####", "..#.", ".#..", "####"],
    5: ["#####", "...#.", "..#..", ".#...", "#####"],
}


@dataclass
class SleepParams:
    frames: int = 12
    breaths: int = 1           # breaths per loop
    breath_amp: int = 1        # px the body settles down on the out-breath
    head_nod: int = 1          # extra px the head droops at the deepest point
    zzz: bool = True
    zzz_color: RGB = (220, 110, 98)
    margin: int = 2            # transparent border so nothing clips


def close_eyes(base: np.ndarray, eye: Box) -> np.ndarray:
    """Replace the open eye inside `eye` with a closed one: head colour fill + a small dark arc.

    The head colour is the most common non-dark colour in a 1 px ring around the box; the arc uses the darkest
    colour found inside the box (the eye's own colour), spanning the width of its dark pixels.
    """
    x0, y0, x1, y1 = eye
    h, w = base.shape[:2]
    lab = rgb_to_lab(base[..., :3])
    opaque = base[..., 3] > 0
    box_dark = np.zeros((h, w), bool)
    box_dark[y0:y1, x0:x1] = opaque[y0:y1, x0:x1] & (lab[y0:y1, x0:x1, 0] < 32)
    ys, xs = np.nonzero(box_dark)
    if xs.size == 0:
        raise EmoteError(f"The eye box {list(eye)} contains no dark eye pixels; adjust regions.json 'eye'")
    ring = [tuple(int(v) for v in base[y, x, :3])
            for y in range(max(y0 - 1, 0), min(y1 + 1, h)) for x in range(max(x0 - 1, 0), min(x1 + 1, w))
            if not (y0 <= y < y1 and x0 <= x < x1) and opaque[y, x] and lab[y, x, 0] >= 32]
    if not ring:
        raise EmoteError(f"No head colour found around the eye box {list(eye)}")
    fill = Counter(ring).most_common(1)[0][0]
    ink = tuple(int(v) for v in base[ys[0], xs[0], :3])   # the eye's dark colour
    out = base.copy()
    inside = opaque[y0:y1, x0:x1]
    out[y0:y1, x0:x1][inside, :3] = fill
    ax0, ax1 = int(xs.min()), int(xs.max())
    yc = int(round((ys.min() + ys.max()) / 2))
    span = ax1 - ax0 + 1
    for k in range(span):  # a shallow smile-shaped arc: ends one row above the middle
        yy = yc if k in (0, span - 1) else yc + 1
        out[yy, ax0 + k] = (*ink, 255)
    return out


def _raw_breath(i: int, p: SleepParams) -> float:
    return 0.5 - 0.5 * math.cos(2 * math.pi * p.breaths * (i % p.frames) / p.frames)   # 0 = in, 1 = out


def breath_offsets(i: int, p: SleepParams) -> tuple[int, int]:
    """(body_dy, head_dy) for frame i. Periodic in i with period p.frames."""
    raw = _raw_breath(i, p)
    # exact halves round DOWN on both sides of the cycle (plain round() flips on float noise, making it lopsided)
    body = int(math.floor(p.breath_amp * raw + 0.5 - 1e-9))
    head = body + (p.head_nod if raw >= 0.95 else 0)
    return body, head


def zzz_glyphs(i: int, p: SleepParams, head: Box) -> list[tuple[int, int, int]]:
    """[(size, x, y)] top-left corners of the visible z glyphs for frame i, oldest (biggest) first."""
    sx, sy = head[2], head[3] - 4              # starts beside the head, level with its lower half
    ex, ey = head[2] + 5, head[1] - 1          # ends up and to the right, top of the head box
    out = []
    for k in range(3):
        t = ((i % p.frames) / p.frames + k / 3) % 1.0
        if t >= 0.85:
            continue                           # gone: the next one is about to start
        size = 3 if t < 0.3 else 4 if t < 0.6 else 5
        out.append((size, int(round(sx + (ex - sx) * t)), int(round(sy + (ey - sy) * t))))
    return sorted(out, key=lambda g: -g[0])


def _draw_glyph(canvas: np.ndarray, size: int, x: int, y: int, color: RGB, m: int) -> None:
    for gy, row in enumerate(GLYPHS[size]):
        for gx, ch in enumerate(row):
            if ch != "#":
                continue
            cy, cx = y + gy + m, x + gx + m
            if 0 <= cy < canvas.shape[0] and 0 <= cx < canvas.shape[1] and canvas[cy, cx, 3] == 0:
                canvas[cy, cx] = (*color, 255)


def sleep_cycle(base: np.ndarray, head: Box, legs: Box, eye: Box, p: SleepParams) -> list[np.ndarray]:
    if p.frames < 4:
        raise EmoteError("A sleep cycle needs at least 4 frames")
    if p.zzz:
        pal = {tuple(int(v) for v in c) for c in base[base[..., 3] > 0][:, :3]}
        if tuple(p.zzz_color) not in pal:
            raise EmoteError(f"zzz_color {list(p.zzz_color)} is not a color of the sprite; pick one of "
                             f"{sorted(pal)[:6]}...")
    closed = close_eyes(base, eye)
    body, head_l, legs_l = split_layers(closed, head, legs)
    m = p.margin
    h, w = base.shape[:2]
    frames = []
    for i in range(p.frames):
        body_dy, head_dy = breath_offsets(i, p)
        canvas = np.zeros((h + 2 * m, w + 2 * m, 4), np.uint8)
        _paste(canvas, body, 0, body_dy, m)
        _paste(canvas, head_l, 0, head_dy, m)
        _paste(canvas, legs_l, 0, 0, m)
        if p.zzz:
            for size, x, y in zzz_glyphs(i, p, head):
                _draw_glyph(canvas, size, x, y, p.zzz_color, m)
        frames.append(canvas)
    return frames
