"""Region config (head / body / legs bounding boxes) stored as JSON.

Format (pixel coordinates in the native sprite, x1/y1 exclusive):
{"head": [x0, y0, x1, y1], "body": [...], "legs": [...]}
"""
from __future__ import annotations

import json
from pathlib import Path

from .sprite import EmoteError

Box = tuple[int, int, int, int]
REGION_NAMES = ("head", "body", "legs", "eye")   # eye is only needed by the sleep animation


def require_regions(path: str | Path, size: tuple[int, int],
                    need: tuple[str, ...] = ("head", "legs")) -> dict[str, Box]:
    """Like load_regions, but the file must exist and define the `need`ed regions (head and legs by default)."""
    regions = load_regions(path, size)
    if regions is None:
        raise EmoteError(f"{path} not found. Create it with head/body/legs boxes "
                         '(see README: {"head": [x0,y0,x1,y1], ...}) and check with `animate --debug-regions`')
    missing = [n for n in need if n not in regions]
    if missing:
        raise EmoteError(f"{path} must define: {', '.join(missing)}")
    return regions


def load_regions(path: str | Path, size: tuple[int, int] | None = None) -> dict[str, Box] | None:
    """Return regions, or None if the file doesn't exist yet."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EmoteError(f"Invalid JSON in {p}: {exc}") from exc
    regions: dict[str, Box] = {}
    for name, box in raw.items():
        if name not in REGION_NAMES:
            raise EmoteError(f"{p}: unknown region '{name}' (expected one of {REGION_NAMES})")
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box)):
            raise EmoteError(f"{p}: region '{name}' must be [x0, y0, x1, y1] integers")
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            raise EmoteError(f"{p}: region '{name}' is empty or inverted: {box}")
        if size and (x0 < 0 or y0 < 0 or x1 > size[0] or y1 > size[1]):
            raise EmoteError(f"{p}: region '{name}' {box} falls outside the {size[0]}x{size[1]} sprite")
        regions[name] = (x0, y0, x1, y1)
    return regions
