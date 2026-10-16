"""Restore: paste exact pixels of the original sprite back onto AI frames.

The AI model redraws the character from scratch, so small details (the eye highlight, the beak, exact head
colors) are only approximated. For a region of the original (e.g. the face), this finds where that region
best matches each frame (a pure translation search, no scaling or rotation) and, if the match is good enough,
overwrites the frame's pixels with the original's. Where the match is poor (head tilted, different size) the
AI pixels are kept and a note is returned, so the step can only improve a frame or leave it untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import rgb_to_lab
from .regions import Box
from .sprite import EmoteError


@dataclass
class Match:
    frame: int
    part: str
    score: float
    offset: tuple[int, int] | None  # (dx, dy) of the template's top-left in the frame; None = not restored
    restored: bool


def _template(base: np.ndarray, box: Box) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box
    t = base[y0:y1, x0:x1]
    if not (t[..., 3] > 0).any():
        raise EmoteError(f"Restore region {list(box)} contains no opaque pixels; adjust the box")
    return t, t[..., 3] > 0


def best_match(frame: np.ndarray, tmpl: np.ndarray, mask: np.ndarray, tol: float) -> tuple[float, tuple[int, int]]:
    """Translation maximising the fraction of template pixels that land on an opaque, similarly coloured pixel."""
    th, tw = mask.shape
    fh, fw = frame.shape[:2]
    if th > fh or tw > fw:
        return 0.0, (0, 0)
    f_lab = rgb_to_lab(frame[..., :3])
    f_op = frame[..., 3] > 0
    t_lab = rgb_to_lab(tmpl[..., :3])[mask]
    n = int(mask.sum())
    best = (-1.0, (0, 0))
    for dy in range(fh - th + 1):
        for dx in range(fw - tw + 1):
            win_op = f_op[dy:dy + th, dx:dx + tw][mask]
            if win_op.sum() < 0.5 * n:  # cheap reject: far too few opaque pixels under the template
                continue
            d = np.linalg.norm(f_lab[dy:dy + th, dx:dx + tw][mask] - t_lab, axis=-1)
            score = float(np.mean(win_op & (d <= tol)))
            if score > best[0]:
                best = (score, (dx, dy))
    return max(best[0], 0.0), best[1]


def restore_frames(frames: list[np.ndarray], base: np.ndarray, parts: dict[str, Box], min_score: float = 0.6,
                   tol: float = 12.0) -> tuple[list[np.ndarray], list[Match]]:
    """Return (restored frames, one Match per frame and part). Frames are native-resolution, palette-snapped."""
    out = [f.copy() for f in frames]
    report: list[Match] = []
    for name, box in parts.items():
        tmpl, mask = _template(base, box)
        th, tw = mask.shape
        for i, f in enumerate(out):
            score, (dx, dy) = best_match(f, tmpl, mask, tol)
            ok = score >= min_score
            if ok:
                region = f[dy:dy + th, dx:dx + tw]
                region[mask] = tmpl[mask]
            report.append(Match(i, name, round(score, 3), (dx, dy) if ok else None, ok))
    return out, report
