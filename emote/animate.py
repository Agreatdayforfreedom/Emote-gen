"""Stage 2a: procedural walk cycle from pixel shifts only (no interpolation, no new colors).

The sprite is split into layers using the region boxes:
  head  = opaque pixels inside the head box
  legs  = opaque pixels inside the legs box (not in head); split into two feet at the empty gap
  body  = every other opaque pixel
Draw order is body -> head -> legs, so a dipping body slides behind the feet.

Phase of frame i is i / N for a full two-step cycle, so frame N would equal frame 0 (seamless loop).
All offsets are in native sprite pixels; "forward" is -x when facing left, +x when facing right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .regions import Box
from .sprite import EmoteError


@dataclass
class WalkParams:
    frames: int = 4
    facing: str = "left"      # direction the sprite faces
    head_amp: int = 1         # px the head thrusts forward at foot contact (signature pigeon bob)
    body_dip: int = 1         # px the body+head dip down on contact frames
    leg_swing: int = 1        # px each foot travels either side of neutral
    leg_lift: int = 1         # px a swinging foot is raised
    margin: int = 2           # transparent border added around the sprite for the offsets


@dataclass
class Offsets:
    head: tuple[int, int]
    body: tuple[int, int]
    front_foot: tuple[int, int]
    back_foot: tuple[int, int]


def frame_offsets(i: int, p: WalkParams) -> Offsets:
    """Per-layer (dx, dy) for frame i. Periodic in i with period p.frames."""
    fwd = -1 if p.facing == "left" else 1
    phi = (i % p.frames) / p.frames
    step = (2 * phi) % 1.0                           # position inside the current step
    contact = 0.5 + 0.5 * math.cos(2 * math.pi * step)   # 1 at foot contact, 0 mid-step
    dip = int(round(p.body_dip * contact))
    head_dx = fwd * int(round(p.head_amp * contact))

    def foot(phase: float) -> tuple[int, int]:
        th = 2 * math.pi * (phi + phase)
        dx = fwd * int(round(p.leg_swing * math.cos(th)))      # +1 cos = fully forward
        lifted = math.sin(th) < -1e-6                          # moving forward = swing phase = airborne
        return dx, (-p.leg_lift if lifted else 0)

    return Offsets(head=(head_dx, dip), body=(0, dip), front_foot=foot(0.0), back_foot=foot(0.5))


def split_layers(base: np.ndarray, head: Box, legs: Box) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (body, head, legs) full-size RGBA layers; every opaque pixel belongs to exactly one."""
    opaque = base[..., 3] > 0
    in_head = np.zeros_like(opaque)
    in_head[head[1]:head[3], head[0]:head[2]] = True
    in_legs = np.zeros_like(opaque)
    in_legs[legs[1]:legs[3], legs[0]:legs[2]] = True
    in_legs &= ~in_head

    def layer(mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(base)
        out[mask & opaque] = base[mask & opaque]
        return out

    return layer(~in_head & ~in_legs), layer(in_head), layer(in_legs)


def split_feet(legs_layer: np.ndarray, facing: str) -> tuple[np.ndarray, np.ndarray]:
    """Split the legs layer into (front, back) foot layers at empty column gaps."""
    cols = np.nonzero(legs_layer[..., 3].any(axis=0))[0]
    if cols.size == 0:
        raise EmoteError("Legs region contains no opaque pixels; fix the 'legs' box (try --debug-regions)")
    runs = np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1)
    if len(runs) != 2:
        raise EmoteError(f"Legs region splits into {len(runs)} feet, expected 2. Adjust the 'legs' box "
                         "so it contains exactly two separate feet (try --debug-regions)")
    feet = []
    for r in runs:
        f = np.zeros_like(legs_layer)
        f[:, r[0]:r[-1] + 1] = legs_layer[:, r[0]:r[-1] + 1]
        feet.append(f)
    left, right = feet
    return (left, right) if facing == "left" else (right, left)  # front foot first


def _paste(canvas: np.ndarray, layer: np.ndarray, dx: int, dy: int, m: int) -> None:
    ys, xs = np.nonzero(layer[..., 3])
    ty, tx = ys + dy + m, xs + dx + m
    ok = (ty >= 0) & (ty < canvas.shape[0]) & (tx >= 0) & (tx < canvas.shape[1])
    if not ok.all():
        raise EmoteError("Animation offsets push pixels off the canvas; increase animate.margin")
    canvas[ty, tx] = layer[ys, xs]


def walk_cycle(base: np.ndarray, head: Box, legs: Box, p: WalkParams) -> list[np.ndarray]:
    if p.frames < 2:
        raise EmoteError("A walk cycle needs at least 2 frames")
    if p.facing not in ("left", "right"):
        raise EmoteError(f"facing must be 'left' or 'right', got {p.facing!r}")
    body, head_l, legs_l = split_layers(base, head, legs)
    front, back = split_feet(legs_l, p.facing)
    m = p.margin
    h, w = base.shape[:2]
    frames = []
    for i in range(p.frames):
        o = frame_offsets(i, p)
        canvas = np.zeros((h + 2 * m, w + 2 * m, 4), np.uint8)
        _paste(canvas, body, *o.body, m)
        _paste(canvas, head_l, *o.head, m)
        _paste(canvas, back, *o.back_foot, m)
        _paste(canvas, front, *o.front_foot, m)
        frames.append(canvas)
    return frames
