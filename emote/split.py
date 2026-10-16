"""Stage 3: split sprite sheets into frames with fixed grid math; load frame folders."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .sprite import EmoteError, load_rgba


def split_sheet(sheet: np.ndarray, rows: int, cols: int) -> list[np.ndarray]:
    """Row-major list of equally sized cells. Sheet size must divide evenly."""
    if rows < 1 or cols < 1:
        raise EmoteError(f"rows/cols must be >= 1 (got {rows}x{cols})")
    h, w = sheet.shape[:2]
    if h % rows or w % cols:
        raise EmoteError(f"Sheet {w}x{h} does not divide evenly into {cols} cols x {rows} rows")
    ch, cw = h // rows, w // cols
    return [sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].copy() for r in range(rows) for c in range(cols)]


def _natural_key(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def load_frames(folder: str | Path) -> list[np.ndarray]:
    d = Path(folder)
    if not d.is_dir():
        raise EmoteError(f"Frames folder not found: {d}")
    files = sorted((p for p in d.iterdir() if p.suffix.lower() == ".png"), key=_natural_key)
    if not files:
        raise EmoteError(f"No .png frames in {d}")
    frames = [load_rgba(p) for p in files]
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise EmoteError(f"Frames in {d} differ in size: {sorted(s[:2] for s in shapes)}")
    return frames
