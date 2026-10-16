"""Stage 5: pixel-snap. NN downscale, quantize to the base palette (Lab), anchor alignment."""
from __future__ import annotations

import numpy as np

from .color import rgb_to_lab
from .sprite import EmoteError, RGB

ANCHORS = ("feet", "bbox-bottom", "none")


def downscale_nn(rgba: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor downscale: sample the center pixel of each factor x factor block."""
    if factor < 1:
        raise EmoteError(f"source_scale must be >= 1, got {factor}")
    if factor == 1:
        return rgba.copy()
    h, w = (rgba.shape[0] // factor) * factor, (rgba.shape[1] // factor) * factor
    return rgba[factor // 2:h:factor, factor // 2:w:factor].copy()


def _grid_offset(rgba: np.ndarray, factor: int) -> tuple[int, int]:
    """(oy, ox): integer offset in [0, factor) at which block edges line up best with colour/alpha edges."""
    from .clean import _best_offset

    f = rgba.astype(np.float64)
    gy = np.concatenate([[0.0], np.abs(np.diff(f, axis=0)).sum(axis=(1, 2))])
    gx = np.concatenate([[0.0], np.abs(np.diff(f, axis=1)).sum(axis=(0, 2))])
    # _best_offset returns the position of the first grid line; trimming that many px puts it at 0
    return int(round(_best_offset(gy, float(factor))[1])) % factor, int(round(_best_offset(gx, float(factor))[1])) % factor


def downscale_vote(rgba: np.ndarray, factor: int, palette: list[RGB], align: bool = False) -> np.ndarray:
    """Block-majority downscale for frames that are NOT exactly grid-aligned (e.g. AI output).

    Every opaque source pixel is first mapped to its nearest palette color; each factor x factor block then
    becomes the most common palette color among its opaque pixels, and is transparent if fewer than half
    its pixels are opaque. Output colors are palette colors by construction (no blending).
    """
    if factor < 1:
        raise EmoteError(f"source_scale must be >= 1, got {factor}")
    pal = np.array(palette, np.uint8)
    if align and factor > 1:  # AI frames sit on a grid offset by a few px: find it and trim to it
        oy, ox = _grid_offset(rgba, factor)
        rgba = rgba[oy:, ox:]
    h, w = (rgba.shape[0] // factor) * factor, (rgba.shape[1] // factor) * factor
    src = rgba[:h, :w]
    opaque = src[..., 3] > 0
    idx = np.full((h, w), -1, np.int16)
    if opaque.any():
        px = rgb_to_lab(src[opaque][:, :3])
        idx[opaque] = np.argmin(np.linalg.norm(px[:, None] - rgb_to_lab(pal)[None], axis=-1), axis=1)
    bh, bw = h // factor, w // factor
    blocks = idx.reshape(bh, factor, bw, factor)
    counts = np.stack([(blocks == p).sum(axis=(1, 3)) for p in range(len(pal))], axis=-1)  # (bh, bw, P)
    n_opaque = counts.sum(axis=-1)
    out = np.zeros((bh, bw, 4), np.uint8)
    solid = n_opaque * 2 >= factor * factor
    out[solid, :3] = pal[np.argmax(counts, axis=-1)[solid]]
    out[solid, 3] = 255
    return out


def quantize(rgba: np.ndarray, palette: list[RGB]) -> np.ndarray:
    """Map every opaque pixel to the nearest palette color (Lab distance). Alpha stays binary."""
    pal = np.array(palette, np.uint8)
    pal_lab = rgb_to_lab(pal)
    out = rgba.copy()
    mask = out[..., 3] > 0
    if mask.any():
        px = rgb_to_lab(out[mask][:, :3])
        idx = np.argmin(np.linalg.norm(px[:, None] - pal_lab[None], axis=-1), axis=1)
        out[mask, :3] = pal[idx]
    out[~mask] = 0
    return out


def anchor_point(rgba: np.ndarray, mode: str, feet_rows: int = 2) -> tuple[int, int] | None:
    """(x, y) anchor of the sprite: y = lowest opaque row; x = centre of the feet or of the bbox."""
    if mode not in ANCHORS:
        raise EmoteError(f"Unknown anchor '{mode}' (use one of {ANCHORS})")
    if mode == "none":
        return None
    ys, xs = np.nonzero(rgba[..., 3])
    if xs.size == 0:
        raise EmoteError("A frame is empty after background removal; check bg.mode / tolerance")
    bottom = int(ys.max())
    if mode == "feet":
        sel = ys > bottom - feet_rows
        x = (int(xs[sel].min()) + int(xs[sel].max()) + 1) // 2
    else:
        x = (int(xs.min()) + int(xs.max()) + 1) // 2
    return x, bottom


def place_on_canvas(frame: np.ndarray, canvas_wh: tuple[int, int], offset: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Paste `frame` at integer `offset`; returns (canvas, number of opaque pixels clipped)."""
    cw, ch = canvas_wh
    canvas = np.zeros((ch, cw, 4), np.uint8)
    ox, oy = offset
    fh, fw = frame.shape[:2]
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + fw, cw), min(oy + fh, ch)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = frame[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    clipped = int(np.count_nonzero(frame[..., 3])) - int(np.count_nonzero(canvas[..., 3]))
    return canvas, clipped


def denoise(rgba: np.ndarray, max_de: float = 25.0, min_neighbors: int = 6) -> np.ndarray:
    """Flatten dithering / speckled shading: a pixel takes its neighbours' colour when at least `min_neighbors`
    of its 8 neighbours share one other colour that is perceptually close (Lab distance <= `max_de`).

    The distance limit protects deliberate small details such as an eye highlight (white inside navy is far
    away in Lab), while tan-vs-grey noise inside a flat area is close and gets flattened. Output colours are
    always colours already present in the frame.
    """
    from collections import Counter

    h, w = rgba.shape[:2]
    opaque = rgba[..., 3] > 0
    lab = rgb_to_lab(rgba[..., :3])
    out = rgba.copy()
    for y in range(h):
        for x in range(w):
            if not opaque[y, x]:
                continue
            near = [tuple(int(v) for v in rgba[y + dy, x + dx, :3]) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w and opaque[y + dy, x + dx]]
            if len(near) < min_neighbors:
                continue
            color, n = Counter(near).most_common(1)[0]
            own = tuple(int(v) for v in rgba[y, x, :3])
            if color != own and n >= min_neighbors:
                ref = next((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                           if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w and opaque[y + dy, x + dx]
                           and tuple(int(v) for v in rgba[y + dy, x + dx, :3]) == color)
                if float(np.linalg.norm(lab[y, x] - lab[ref])) <= max_de:
                    out[y, x, :3] = color
    return out


def thin_outline(rgba: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Shave fat outlines down to 1 px, keeping the silhouette.

    `dark` marks outline-coloured pixels. Depth = 4-neighbour distance from the outside, travelling only through
    dark pixels (outermost outline layer = 1; dark shapes not connected to the outside stay unreached). A dark
    pixel is redundant when it sits in a 2x2 block of dark pixels with at least two pixels of smaller depth, which
    is the signature of a band 2+ px thick: a true 1 px outline, a staircase diagonal or an inner 1 px detail
    line never forms such a block, so they are left alone. Removed pixels take the most common nearby non-dark
    colour, so no colour outside the palette can appear.
    """
    from collections import Counter, deque

    h, w = dark.shape
    opaque = rgba[..., 3] > 0
    INF = 10**6
    depth = np.full((h, w), INF, np.int32)
    q: deque[tuple[int, int]] = deque()
    for y in range(h):
        for x in range(w):
            if dark[y, x] and any(not (0 <= y + dy < h and 0 <= x + dx < w) or not opaque[y + dy, x + dx]
                                  for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                depth[y, x] = 1
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and dark[ny, nx] and depth[ny, nx] == INF:
                depth[ny, nx] = depth[y, x] + 1
                q.append((ny, nx))
    redundant = []
    for y in range(h):
        for x in range(w):
            if depth[y, x] < 2 or depth[y, x] == INF:
                continue
            for oy in (-1, 0):
                for ox in (-1, 0):
                    ys, xs = (y + oy, y + oy + 1), (x + ox, x + ox + 1)
                    if ys[0] < 0 or xs[0] < 0 or ys[1] >= h or xs[1] >= w:
                        continue
                    block = [(a, b) for a in ys for b in xs]
                    if all(depth[a, b] < INF for a, b in block) and \
                            sum(depth[a, b] < depth[y, x] for a, b in block) >= 2:
                        redundant.append((y, x))
                        break
                else:
                    continue
                break
    out = rgba.copy()
    pending = set(redundant)
    filled: set[tuple[int, int]] = set()
    while pending:  # recolour from the body colour inward, so bands thicker than 2 px are handled too
        batch = {}
        for y, x in pending:
            near = [tuple(int(v) for v in out[y + dy, x + dx, :3]) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w and opaque[y + dy, x + dx]
                    and (y + dy, x + dx) not in pending
                    and (not dark[y + dy, x + dx] or (y + dy, x + dx) in filled)]
            if near:
                batch[(y, x)] = Counter(near).most_common(1)[0][0]
        if not batch:  # no body colour reachable: not a band over body colour, keep these pixels as they are
            break
        for (y, x), color in batch.items():
            out[y, x, :3] = color
            filled.add((y, x))
        pending -= set(batch)
    return out


def remove_specks(rgba: np.ndarray, min_px: int) -> np.ndarray:
    """Delete 8-connected opaque components smaller than `min_px` pixels (AI noise specks around wings)."""
    if min_px <= 1:
        return rgba
    h, w = rgba.shape[:2]
    mask = rgba[..., 3] > 0
    seen = np.zeros((h, w), bool)
    out = rgba.copy()
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            comp, stack = [], [(sy, sx)]
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(comp) < min_px:
                for y, x in comp:
                    out[y, x] = 0
    return out


def crop_union(frames: list[np.ndarray], pad: int = 0) -> list[np.ndarray]:
    """Crop every frame to the union bounding box of all frames (+pad), so frames stay registered."""
    alpha = np.any([f[..., 3] > 0 for f in frames], axis=0)
    ys, xs = np.nonzero(alpha)
    if xs.size == 0:
        raise EmoteError("All frames are empty; nothing to crop")
    out = []
    for f in frames:
        h, w = f.shape[:2]
        y0, y1, x0, x1 = ys.min() - pad, ys.max() + 1 + pad, xs.min() - pad, xs.max() + 1 + pad
        canvas = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        sy0, sx0 = max(y0, 0), max(x0, 0)
        sy1, sx1 = min(y1, h), min(x1, w)
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = f[sy0:sy1, sx0:sx1]
        out.append(canvas)
    return out


def snap_frames(frames: list[np.ndarray], factor: int, palette: list[RGB], native_wh: tuple[int, int],
                anchor: str = "feet", feet_rows: int = 2, canvas_pad: int | str = 0,
                ref_anchor: tuple[int, int] | None = None, crop: bool = False,
                crop_pad: int = 0, downscale: str = "center", despeckle: int = 0,
                restore: tuple | None = None, notes: list[str] | None = None, thin: bool = False,
                flatten: bool = False) -> tuple[list[np.ndarray], list[str]]:
    """Downscale -> quantize -> align. Returns (frames on a native+pad canvas, warnings).

    With an anchor, every frame is shifted so its anchor lands on `ref_anchor` (default: the
    first frame's anchor with that frame centred on the canvas). anchor="none" just centres each
    frame. `ref_anchor` is in canvas coordinates.
    """
    warnings: list[str] = []
    if downscale == "center":      # frames are exact integer upscales: sample block centres
        snapped = [quantize(downscale_nn(f, factor), palette) for f in frames]
    elif downscale == "vote":      # frames are only roughly on a grid: block majority vote
        snapped = [downscale_vote(f, factor, palette, align=True) for f in frames]
    else:
        raise EmoteError(f"snap.downscale must be 'center' or 'vote', got {downscale!r}")
    if despeckle:
        snapped = [remove_specks(f, despeckle) for f in snapped]
    if flatten:   # speckled / dithered shading -> flat colours (before thinning, so noise can't fake a band)
        snapped = [denoise(f) for f in snapped]
    if thin:      # AI outlines come out ~2 px thick; shave them to the 1 px of pixel art
        from .metrics import dark_mask

        snapped = [thin_outline(f, dark_mask(f)) for f in snapped]
    if restore:  # (base sprite, {name: box}, min_score, tolerance): exact original pixels where they match
        from .restore import restore_frames

        base_sprite, parts, min_score, tol = restore
        snapped, matches = restore_frames(snapped, base_sprite, parts, min_score, tol)
        if notes is not None:
            for m in matches:
                notes.append(f"restore '{m.part}' frame {m.frame + 1}: "
                             + (f"pasted original pixels (match {m.score:.2f})" if m.restored
                                else f"kept the AI pixels (match {m.score:.2f} < {min_score})"))
    if canvas_pad == "auto":  # canvas grows to the (downscaled) frame size, so nothing is ever clipped
        cw, ch = snapped[0].shape[1], snapped[0].shape[0]
    elif isinstance(canvas_pad, int):
        cw, ch = native_wh[0] + 2 * canvas_pad, native_wh[1] + 2 * canvas_pad
    else:
        raise EmoteError(f"canvas_pad must be an integer or 'auto', got {canvas_pad!r}")
    out = []
    ref = ref_anchor
    for i, f in enumerate(snapped):
        centred = ((cw - f.shape[1]) // 2, (ch - f.shape[0]) // 2)
        pt = anchor_point(f, anchor, feet_rows)
        if pt is None:
            offset = centred
        else:
            if ref is None:  # first frame defines the anchor
                ref = (pt[0] + centred[0], pt[1] + centred[1])
            offset = (ref[0] - pt[0], ref[1] - pt[1])
        placed, clipped = place_on_canvas(f, (cw, ch), offset)
        if clipped:
            warnings.append(f"frame {i}: {clipped} opaque pixels clipped; raise process.snap.canvas_pad")
        out.append(placed)
    if crop:
        out = crop_union(out, crop_pad)
    return out, warnings
