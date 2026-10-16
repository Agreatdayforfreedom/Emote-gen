"""Quality metrics for pixel-art frames."""
from __future__ import annotations

import numpy as np

from .color import rgb_to_lab


def dark_mask(frame: np.ndarray, lightness: float = 32.0) -> np.ndarray:
    """Opaque pixels dark enough (Lab L*) to be part of an outline."""
    return (frame[..., 3] > 0) & (rgb_to_lab(frame[..., :3])[..., 0] < lightness)


def outline_runs(frame: np.ndarray, lightness: float = 32.0, max_run: int = 4) -> np.ndarray:
    """For every silhouette-boundary pixel and each open direction, the number of consecutive dark pixels
    going inward from it. 1 = a one-pixel outline, 2+ = a fat outline, 0 = no dark edge there.

    Runs longer than `max_run` are skipped: at a corner the "inward" direction runs along the outline itself, and
    no real outline is that thick."""
    dark, opaque = dark_mask(frame, lightness), frame[..., 3] > 0
    h, w = opaque.shape
    runs: list[int] = []
    for y in range(h):
        for x in range(w):
            if not opaque[y, x]:
                continue
            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and opaque[ny, nx]:
                    continue  # not a boundary in this direction
                r, yy, xx = 0, y, x
                while 0 <= yy < h and 0 <= xx < w and opaque[yy, xx] and dark[yy, xx]:
                    r += 1
                    yy, xx = yy - dy, xx - dx
                if r <= max_run:
                    runs.append(r)
    return np.array(runs, dtype=int)


def outline_summary(frames: list[np.ndarray]) -> dict[str, float]:
    """Mean outline thickness and the share of boundary samples that are exactly 1 px / 2+ px / missing."""
    r = np.concatenate([outline_runs(f) for f in frames]) if frames else np.array([], int)
    if r.size == 0:
        return {"mean": 0.0, "one_px": 0.0, "fat": 0.0, "none": 0.0}
    return {"mean": float(r.mean()), "one_px": float(np.mean(r == 1)), "fat": float(np.mean(r >= 2)),
            "none": float(np.mean(r == 0))}
