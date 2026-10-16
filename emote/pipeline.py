"""Stages 3-6 glue: frames -> split -> bg removal -> pixel-snap -> export + preview + spec check."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from .animate import WalkParams, walk_cycle
from .bgremove import remove_background
from .export import contact_sheet, fit_square, gif_duration_ms, save_apng, save_gif, save_spritesheet
from .regions import require_regions
from .sleep import SleepParams, sleep_cycle
from .snap import snap_frames
from .specs import check_gif, load_specs
from .split import load_frames, split_sheet
from .sprite import EmoteError, extract_palette, load_rgba, save_png

STALE_SPECS_DAYS = 90   # warn when twitch_specs.json is older than this
MAX_BASE_COLORS = 64    # a clean pixel sprite has far fewer; noisy upscales have thousands


def process(cfg: dict, frames_dir: str | Path | None = None, sheet: str | Path | None = None,
            out_dir: str | Path | None = None, name: str | None = None) -> dict:
    """Run stages 3-6. Returns a summary dict (files, sizes, frame count, warnings)."""
    pc, ec = cfg["process"], cfg["export"]
    out = Path(out_dir or cfg["paths"]["out_dir"])
    name = name or ec["name"]
    base = load_rgba(cfg["paths"]["base"])
    palette = extract_palette(base, cfg["prep"]["alpha_cutoff"])
    native_wh = (base.shape[1], base.shape[0])

    # Stage 3: split
    sheet = sheet or pc["sheet"]
    if sheet:
        frames = split_sheet(load_rgba(sheet), pc["rows"], pc["cols"])
    else:
        frames_dir = Path(frames_dir or cfg["paths"]["frames_dir"])
        frames = load_frames(frames_dir)
        meta = frames_dir / "meta.json"  # written by the procedural animator: frames are native-res, transparent
        if meta.is_file():
            m = json.loads(meta.read_text(encoding="utf-8"))
            pc = {**pc, "source_scale": m["source_scale"], "bg": {**pc["bg"], "mode": m["bg_mode"]},
                  "snap": {**pc["snap"], "anchor": m["anchor"], "canvas_pad": m["canvas_pad"],
                           "downscale": m.get("downscale", "center"),
                           "despeckle": m.get("despeckle", pc["snap"]["despeckle"]),
                           "thin_outline": m.get("thin_outline", pc["snap"]["thin_outline"]),
                           "flatten": m.get("flatten", pc["snap"]["flatten"])}}

    # Stage 4: background removal
    bg = pc["bg"]
    frames = remove_background(frames, bg["mode"], tuple(bg["key"] or cfg["prep"]["chroma_color"]),
                               bg["tolerance"], bg["alpha_cutoff"], bg["ai_model"])

    # Stage 5: pixel-snap (+ anchor alignment)
    sn = pc["snap"]
    notes: list[str] = []
    rc = cfg["restore"]
    restore = None
    if rc["regions"] and sn.get("downscale") == "vote":  # AI frames only; procedural frames already keep exact pixels
        h, w = base.shape[:2]
        for region, box in rc["regions"].items():
            if not (len(box) == 4 and 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise EmoteError(f"restore.regions.{region}={box} must be [x0, y0, x1, y1] inside the {w}x{h} sprite")
        restore = (base, {k: tuple(v) for k, v in rc["regions"].items()}, rc["min_score"], rc["tolerance"])
    frames, warnings = snap_frames(frames, pc["source_scale"], palette, native_wh, sn["anchor"],
                                   sn["feet_rows"], sn["canvas_pad"], crop=sn["crop"], crop_pad=sn["crop_pad"],
                                   downscale=sn.get("downscale", "center"), despeckle=sn.get("despeckle", 0),
                                   restore=restore, notes=notes, thin=sn.get("thin_outline", False),
                                   flatten=sn.get("flatten", False))
    stray = {tuple(int(v) for v in p[:3]) for f in frames for p in f.reshape(-1, 4) if p[3]} - set(palette)
    if stray:  # unreachable by construction; guards against regressions
        raise EmoteError(f"Internal error: colors outside the base palette: {sorted(stray)[:5]}")

    # Stage 6: export
    specs = load_specs(cfg["paths"]["specs"])
    fps = ec["fps"]
    files: list[Path] = []
    frames_out = out / "frames"
    for i, f in enumerate(frames):
        save_png(f, frames_out / f"{name}_{i:02d}.png")
    for size in [*ec["sizes"], "native"]:
        sized = frames if size == "native" else [fit_square(f, size, ec["integer_upscale"]) for f in frames]
        tag = "native" if size == "native" else str(size)
        gif = save_gif(sized, out / f"{name}_{tag}.gif", fps)
        apng = save_apng(sized, out / f"{name}_{tag}.png", fps)
        files += [gif, apng]
        if size != "native":
            warnings += check_gif(gif, size, specs)
    sheet_path = contact_sheet(frames, out / f"{name}_preview.png", small=min(ec["sizes"]))
    files.append(sheet_path)
    sp = ec["spritesheet"]
    if sp["enabled"]:
        files.append(save_spritesheet(frames, out / f"{name}_sheet.png", fps, sp["cols"], sp["scale"]))
        files.append(out / f"{name}_sheet.json")

    eff = 1000 / gif_duration_ms(fps)
    if abs(eff - fps) / fps > 0.10:
        warnings.append(f"GIF timing is whole centiseconds (min 20 ms): {fps:g} fps becomes {eff:.2f} fps")
    with Image.open(out / f"{name}_native.gif") as im:
        if im.n_frames < len(frames):
            warnings.append(f"GIF has {im.n_frames} frames for {len(frames)} inputs: identical consecutive "
                            "frames are merged (timing is preserved)")
    age = (date.today() - date.fromisoformat(specs["date_checked"])).days
    if age > STALE_SPECS_DAYS:
        warnings.append(f"twitch_specs.json was checked {age} days ago ({specs['date_checked']}); re-verify at "
                        f"{specs['source_url']} and update the file")
    dims = []
    for p in files:
        if p.suffix == ".json":
            dims.append("atlas")
            continue
        with Image.open(p) as im:
            dims.append("%dx%d" % im.size)
    return {
        "frames": len(frames),
        "files": [(p, p.stat().st_size, d) for p, d in zip(files, dims)],
        "fps_requested": fps,
        "fps_gif_effective": eff,
        "warnings": [w for w in warnings if w],
        "notes": notes,
        "specs_source": f"{specs['source_url']} (checked {specs['date_checked']})",
        "strict": bool(cfg.get("strict")),
    }


def animate_to_folder(cfg: dict, out_dir: str | Path | None = None) -> tuple[Path, list[np.ndarray], Path]:
    """Stage 2a: write procedural frames for `animate.motion` (walk | sleep) + meta.json for `process`.
    Returns (folder, frames, preview)."""
    base = load_rgba(cfg["paths"]["base"])
    ac = cfg["animate"]
    motion = ac.get("motion", "walk")
    size = (base.shape[1], base.shape[0])
    if motion == "walk":
        regions = require_regions(cfg["paths"]["regions"], size)
        params = WalkParams(**{k: ac[k] for k in ("frames", "facing", "head_amp", "body_dip", "leg_swing",
                                                  "leg_lift", "margin")})
        frames = walk_cycle(base, regions["head"], regions["legs"], params)
    elif motion == "sleep":
        regions = require_regions(cfg["paths"]["regions"], size, need=("head", "legs", "eye"))
        sc = {k: v for k, v in ac["sleep"].items() if k != "fps"}
        sc["zzz_color"] = tuple(sc["zzz_color"])
        frames = sleep_cycle(base, regions["head"], regions["legs"], regions["eye"], SleepParams(**sc))
    else:
        raise EmoteError(f"Unknown animate.motion '{motion}' (use 'walk' or 'sleep')")
    out = Path(out_dir or ac["out"])
    for old in out.glob("frame_*.png") if out.is_dir() else []:
        old.unlink()
    for i, f in enumerate(frames):
        save_png(f, out / f"frame_{i:02d}.png")
    (out / "meta.json").write_text(json.dumps(
        {"source_scale": 1, "bg_mode": "none", "anchor": "none", "canvas_pad": "auto"}, indent=2), encoding="utf-8")
    preview = contact_sheet(frames, Path(cfg["paths"]["out_dir"]) / f"{motion}_preview.png")
    return out, frames, preview


def check_base(cfg: dict) -> None:
    """Fail early, with a fix, if base.png isn't a clean pixel sprite."""
    base = load_rgba(cfg["paths"]["base"])
    n = len(extract_palette(base, cfg["prep"]["alpha_cutoff"]))
    if n > MAX_BASE_COLORS:
        raise EmoteError(f"{cfg['paths']['base']} has {n} colors, so it is not a clean pixel sprite "
                         f"(noise / anti-aliasing / upscaling). Run `python -m emote clean` first.")
    if (base[..., 3] > 0).all():
        raise EmoteError(f"{cfg['paths']['base']} has no transparent pixels; the animator needs a sprite on "
                         "a transparent background. Run `python -m emote clean` first.")


def run(cfg: dict, out_dir: str | Path | None = None) -> dict:
    """Full pipeline: validate base -> frames (backend) -> stages 3-6."""
    check_base(cfg)
    backend = cfg["run"]["backend"]
    name = None
    if backend == "procedural":
        folder, _, _ = animate_to_folder(cfg)
        motion = cfg["animate"].get("motion", "walk")
        if motion != "walk":  # other motions get their own file names and their natural speed
            name = f"{cfg['export']['name']}_{motion}"
            cfg = {**cfg, "export": {**cfg["export"], "fps": cfg["animate"][motion]["fps"]}}
    elif backend == "ai":
        from .aigen import generate
        folder = generate(cfg)
        name = cfg["ai"]["name"]
    else:
        raise EmoteError(f"Unknown run.backend '{backend}' (use 'procedural' or 'ai')")
    return process(cfg, folder, out_dir=out_dir, name=name)


def make_test_frames(cfg: dict, out_dir: str | Path, noise: int = 3, seed: int | None = None) -> list[Path]:
    """Hand-made test frames: the base sprite shifted by whole pixels, upscaled onto chroma green with
    per-pixel colour noise. Frames 1-3 are deliberately misaligned to exercise anchoring."""
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    base = load_rgba(cfg["paths"]["base"])
    k = cfg["prep"]["upscale"]
    key = np.array(cfg["prep"]["chroma_color"], np.uint8)
    m = 3  # transparent margin (native px) so shifted frames are never clipped
    base = np.pad(base, ((m, m), (m, m), (0, 0)))
    h, w = base.shape[:2]
    shifts = [(0, 0), (2, -1), (-2, 1), (1, 0)]  # whole-sprite (dx, dy) in native pixels
    head_dx = [0, -1, 0, 1]                      # extra shift of the top 40% (pigeon head bob)
    paths = []
    for i, (dx, dy) in enumerate(shifts):
        canvas = np.zeros_like(base)
        ys, xs = np.nonzero(base[..., 3])
        head = ys < m + int(0.4 * (h - 2 * m))
        ny, nx = ys + dy, xs + dx + np.where(head, head_dx[i], 0)
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        canvas[ny[ok], nx[ok]] = base[ys[ok], xs[ok]]
        big = np.repeat(np.repeat(canvas, k, 0), k, 1)
        rgb = np.where(big[..., 3:4] > 0, big[..., :3], key).astype(np.int16)
        rgb = np.clip(rgb + rng.integers(-noise, noise + 1, rgb.shape), 0, 255).astype(np.uint8)
        paths.append(save_png(np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]),
                              Path(out_dir) / f"frame_{i:02d}.png"))
    return paths
