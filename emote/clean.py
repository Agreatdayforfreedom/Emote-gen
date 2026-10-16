"""Recover a clean native-resolution pixel sprite from a blurred / re-compressed upscale.

Steps: detect pixel-block grid -> sample one color per block -> cluster to a small
palette (Lab, deterministic) -> flood-fill the flat background -> optional shadow removal.
No interpolation is used anywhere: each output pixel is the median of one block's core.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import delta_e, rgb_to_lab
from .sprite import EmoteError


@dataclass
class Grid:
    period_x: float
    period_y: float
    off_x: float
    off_y: float


def _edge_profile(rgb: np.ndarray, axis: int) -> np.ndarray:
    g = np.abs(np.diff(rgb.astype(np.float64), axis=axis)).sum(axis=(1 - axis, 2))
    return np.concatenate([[0.0], g])


def _best_offset(g: np.ndarray, p: float) -> tuple[float, float]:
    n, mean = len(g), g.mean() + 1e-9
    def at(o: float) -> float:
        return g[np.clip(np.round(np.arange(o, n, p)).astype(int), 1, n - 1)].mean()

    # Contrast between on-grid edges and the half-period-shifted positions: a multiple of
    # the true period has real edges at its half-shift, so it scores low.
    return max(((at(o) - at(o + p / 2)) / mean, float(o)) for o in np.arange(0, p, 0.25))


def detect_grid(rgb: np.ndarray, pmin: float = 2.5, pmax: float = 12.0) -> Grid:
    """Fit one shared (square-pixel) period plus per-axis offsets to the edge energy.

    """
    gy, gx = _edge_profile(rgb, 0), _edge_profile(rgb, 1)
    cands = []
    for p in np.arange(pmin, pmax, 0.02):
        sy, oy = _best_offset(gy, p)
        sx, ox = _best_offset(gx, p)
        cands.append((sx + sy, float(p), ox, oy))
    best = max(cands)
    # A period of k * true_period also scores high; prefer a sub-period that scores nearly as well.
    for d in (2, 3, 4, 5):
        near = [c for c in cands if abs(c[1] - best[1] / d) <= 0.1]
        if near and max(near)[0] >= 0.6 * best[0] and best[1] / d >= pmin:
            best = max(near)
            break
    _, p, ox, oy = best
    return Grid(p, p, ox, oy)


def sample_cells(rgb: np.ndarray, grid: Grid, core: float = 0.5) -> np.ndarray:
    """Median color of the central `core` fraction of every grid cell -> (rows, cols, 3)."""
    h, w = rgb.shape[:2]
    # Cell edges sit at off + k*period; the first partial cell (before off) is kept if wide enough.
    x_edges = _edges(w, grid.period_x, grid.off_x)
    y_edges = _edges(h, grid.period_y, grid.off_y)
    out = np.zeros((len(y_edges) - 1, len(x_edges) - 1, 3), np.uint8)
    for j in range(len(y_edges) - 1):
        for i in range(len(x_edges) - 1):
            x0, x1, y0, y1 = x_edges[i], x_edges[i + 1], y_edges[j], y_edges[j + 1]
            mx, my = (x1 - x0) * (1 - core) / 2, (y1 - y0) * (1 - core) / 2
            xs, xe = int(round(x0 + mx)), max(int(round(x1 - mx)), int(round(x0 + mx)) + 1)
            ys, ye = int(round(y0 + my)), max(int(round(y1 - my)), int(round(y0 + my)) + 1)
            patch = rgb[ys:ye, xs:xe].reshape(-1, 3)
            out[j, i] = np.median(patch, axis=0).astype(np.uint8)
    return out


def _edges(n: int, period: float, off: float) -> list[float]:
    e = list(np.arange(off % period, n, period))
    if e and e[0] >= period * 0.6:
        e.insert(0, 0.0)
    if e[-1] < n - period * 0.6:
        e.append(float(n))
    return [min(v, float(n)) for v in e]


def kmeans_palette(colors_lab: np.ndarray, k: int, iters: int = 50) -> np.ndarray:
    """Deterministic k-means (farthest-point init, most-common-color first). Returns centers in Lab."""
    pts = colors_lab
    uniq, counts = np.unique(np.round(pts, 3), axis=0, return_counts=True)
    centers = [uniq[np.argmax(counts)]]
    while len(centers) < min(k, len(uniq)):
        d = np.min(np.linalg.norm(uniq[:, None] - np.array(centers)[None], axis=-1), axis=1)
        centers.append(uniq[np.argmax(d * np.sqrt(counts))])
    c = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(pts[:, None] - c[None], axis=-1), axis=1)
        new = np.array([pts[lab == i].mean(axis=0) if np.any(lab == i) else c[i] for i in range(len(c))])
        if np.allclose(new, c):
            break
        c = new
    return c


def _flood_background(is_bg: np.ndarray) -> np.ndarray:
    """4-connected fill of `is_bg` cells reachable from the border."""
    h, w = is_bg.shape
    seen = np.zeros_like(is_bg)
    stack = [(y, x) for y in range(h) for x in range(w) if (y in (0, h - 1) or x in (0, w - 1)) and is_bg[y, x]]
    while stack:
        y, x = stack.pop()
        if seen[y, x]:
            continue
        seen[y, x] = True
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and is_bg[ny, nx] and not seen[ny, nx]:
                stack.append((ny, nx))
    return seen


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest 8-connected component of a boolean mask."""
    h, w = mask.shape
    label = np.zeros((h, w), np.int32)
    sizes: list[int] = [0]
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and not label[sy, sx]:
                sizes.append(0)
                lid = len(sizes) - 1
                stack = [(sy, sx)]
                while stack:
                    y, x = stack.pop()
                    if not (0 <= y < h and 0 <= x < w) or not mask[y, x] or label[y, x]:
                        continue
                    label[y, x] = lid
                    sizes[lid] += 1
                    stack.extend((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return label == int(np.argmax(sizes)) if len(sizes) > 1 else mask


def clean_sprite(
    rgb: np.ndarray,
    colors: int = 16,
    merge_de: float = 6.0,
    bg_tolerance: float = 5.0,
    remove_shadow: bool = True,
    pad: int = 1,
    block: float | None = None,
    offset: tuple[float, float] | None = None,
) -> tuple[np.ndarray, Grid]:
    """Return (native RGBA sprite, detected grid)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise EmoteError("clean expects an RGB image")
    grid = detect_grid(rgb)
    if block:
        grid.period_x = grid.period_y = block
    if offset:
        grid.off_x, grid.off_y = offset
    cells = sample_cells(rgb, grid)
    h, w = cells.shape[:2]
    lab = rgb_to_lab(cells.reshape(-1, 3))

    corner = np.median(np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]]), axis=0)
    bg_lab = rgb_to_lab(corner)
    is_bg = _flood_background(delta_e(lab, bg_lab).reshape(h, w) <= bg_tolerance)

    centers = kmeans_palette(lab[~is_bg.reshape(-1)], colors)
    # merge near-duplicate centers
    keep: list[np.ndarray] = []
    for c in centers:
        if all(delta_e(c, k) > merge_de for k in keep):
            keep.append(c)
    centers = np.array(keep)
    idx = np.argmin(np.linalg.norm(lab[:, None] - centers[None], axis=-1), axis=1).reshape(h, w)

    # palette colors in RGB = the most common actual sampled color of each cluster (exact, no drift)
    flat_cells = cells.reshape(-1, 3)
    pal = np.zeros((len(centers), 3), np.uint8)
    for i in range(len(centers)):
        members = flat_cells[(idx.reshape(-1) == i) & ~is_bg.reshape(-1)]
        if len(members):
            u, n = np.unique(members, axis=0, return_counts=True)
            pal[i] = u[np.argmax(n)]

    alpha = np.where(is_bg, 0, 255).astype(np.uint8)
    out = np.dstack([pal[idx], alpha])

    if remove_shadow:
        pal_lab = rgb_to_lab(pal)
        shadow = [i for i in range(len(pal)) if pal_lab[i, 0] > 60 and pal_lab[i, 2] < -6 and pal_lab[i, 1] > -5]
        for i in shadow:
            out[(idx == i) & (alpha > 0), 3] = 0

    out[..., 3] = np.where(_largest_component(out[..., 3] > 0), 255, 0)  # drop stray noise specks

    out[out[..., 3] == 0] = 0  # no stray RGB under transparent pixels

    ys, xs = np.nonzero(out[..., 3])
    if xs.size == 0:
        raise EmoteError("Nothing left after background removal; try a larger bg_tolerance or set block/offset manually")
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + 1 + pad, max(xs.min() - pad, 0), xs.max() + 1 + pad
    cropped = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    src = out[y0:y1, x0:x1]
    cropped[: src.shape[0], : src.shape[1]] = src
    return cropped, grid
