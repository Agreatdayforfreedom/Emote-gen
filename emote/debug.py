"""--debug-regions: draw the region boxes (and a coordinate ruler) over the upscaled sprite."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .regions import Box
from .sprite import upscale_nn

COLORS = {"head": (255, 60, 60), "body": (60, 200, 90), "legs": (70, 130, 255)}


def draw_regions(base: np.ndarray, regions: dict[str, Box], path: str | Path, scale: int = 16) -> Path:
    """Boxes are drawn on the pixel grid edges (x1/y1 exclusive), with native-pixel coordinates on the rulers."""
    h, w = base.shape[:2]
    gutter = 14
    bg = np.empty((h * scale, w * scale, 4), np.uint8)
    yy, xx = np.indices((h, w))
    bg[...] = np.repeat(np.repeat(np.where(((xx + yy) % 2 == 0)[..., None], [58, 58, 64, 255], [48, 48, 54, 255]),
                                  scale, 0), scale, 1).astype(np.uint8)
    big = upscale_nn(base, scale)
    solid = big[..., 3] > 0
    bg[solid] = big[solid]

    im = Image.new("RGB", (w * scale + gutter, h * scale + gutter), (24, 24, 27))
    im.paste(Image.fromarray(bg[..., :3]), (gutter, gutter))
    d = ImageDraw.Draw(im)
    for x in range(0, w, 2):
        d.text((gutter + x * scale + 2, 1), str(x), fill=(170, 170, 180))
    for y in range(0, h, 2):
        d.text((1, gutter + y * scale + 2), str(y), fill=(170, 170, 180))
    for name, (x0, y0, x1, y1) in regions.items():
        c = COLORS.get(name, (255, 255, 0))
        d.rectangle([gutter + x0 * scale, gutter + y0 * scale, gutter + x1 * scale - 1, gutter + y1 * scale - 1],
                    outline=c, width=2)
        d.text((gutter + x0 * scale + 4, gutter + y0 * scale + 3), f"{name} [{x0},{y0},{x1},{y1}]", fill=c)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out
