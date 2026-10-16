"""Command-line entrypoint: python -m emote <command>."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .gpu import gpu_report
from .regions import load_regions, require_regions
from .sprite import (
    EmoteError, bounding_box, extract_palette, load_rgba, on_background,
    palette_counts, save_png, semi_transparent_pixels, upscale_nn,
)


def _hex(c: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % c


def cmd_inspect(args: argparse.Namespace, cfg: dict) -> int:
    base = Path(args.image or cfg["paths"]["base"])
    cutoff = cfg["prep"]["alpha_cutoff"]
    rgba = load_rgba(base)
    h, w = rgba.shape[:2]
    print(f"File:        {base}")
    print(f"Native size: {w}x{h}")

    corners = {tuple(int(v) for v in rgba[y, x]) for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))}
    transparent = int((rgba[..., 3] == 0).sum())
    print(f"Transparent pixels: {transparent} ({transparent / (w * h):.0%})")
    if transparent == 0:
        print(f"  note: no transparency; corner colors = {sorted(_hex(c[:3]) for c in corners)} "
              "(flat background? it would be counted as a palette color)")
    semi = semi_transparent_pixels(rgba)
    if semi:
        print(f"  WARNING: {semi} semi-transparent pixels (anti-aliasing?); they are ignored below {cutoff} alpha")

    box = bounding_box(rgba, cutoff)
    if box:
        print(f"Opaque bounding box: {box}  ({box[2] - box[0]}x{box[3] - box[1]})")

    pal = palette_counts(rgba, cutoff)
    print(f"Palette: {len(pal)} colors")
    for color, n in pal:
        print(f"  {_hex(color)}  rgb{color}  {n} px")

    regions_path = Path(cfg["paths"]["regions"])
    regions = load_regions(regions_path, (w, h))
    if regions is None:
        print(f"Regions: not defined yet ({regions_path} missing). Create it as "
              '{"head": [x0,y0,x1,y1], "body": [...], "legs": [...]}')
    else:
        print(f"Regions ({regions_path}):")
        for name, b in regions.items():
            print(f"  {name:5s} {b}")

    if args.debug_regions:
        _debug_regions(cfg)
    if args.upscale_preview:
        factor = cfg["prep"]["upscale"]
        bg = tuple(cfg["prep"]["chroma_color"])
        out = save_png(on_background(upscale_nn(rgba, factor), bg), Path(cfg["paths"]["out_dir"]) / f"base_x{factor}_chroma.png")
        print(f"Saved {factor}x chroma-key preview: {out}")
    return 0


def cmd_clean(args: argparse.Namespace, cfg: dict) -> int:
    import shutil

    import numpy as np

    from .clean import clean_sprite

    src = Path(args.image or cfg["paths"]["base"])
    c = cfg["clean"]
    rgba = load_rgba(src)
    sprite, grid = clean_sprite(
        rgba[..., :3], colors=c["colors"], merge_de=c["merge_delta_e"], bg_tolerance=c["bg_tolerance"],
        remove_shadow=c["remove_shadow"], pad=c["pad"], block=c["block"],
        offset=tuple(c["offset"]) if c["offset"] else None,
    )
    print(f"Grid: period {grid.period_x:.2f}x{grid.period_y:.2f}px, offset ({grid.off_x:.2f}, {grid.off_y:.2f})")
    print(f"Native sprite: {sprite.shape[1]}x{sprite.shape[0]}, palette {len(extract_palette(sprite))} colors")
    dst = Path(args.output or cfg["paths"]["base"])
    if dst.resolve() == src.resolve():
        backup = src.with_name("original" + src.suffix)
        if not backup.exists():
            shutil.copyfile(src, backup)
            print(f"Original kept at {backup}")
    save_png(sprite, dst)
    print(f"Saved {dst}")
    save_png(upscale_nn(sprite, 8), Path(cfg["paths"]["out_dir"]) / "clean_preview_x8.png")
    print(f"Preview: {Path(cfg['paths']['out_dir']) / 'clean_preview_x8.png'}")
    return 0


def _debug_regions(cfg: dict) -> int:
    from .debug import draw_regions

    base = load_rgba(cfg["paths"]["base"])
    regions = require_regions(cfg["paths"]["regions"], (base.shape[1], base.shape[0]))
    out = draw_regions(base, regions, Path(cfg["paths"]["out_dir"]) / "debug_regions.png")
    print(f"Wrote {out}  (edit {cfg['paths']['regions']} and re-run to tune)")
    return 0


def cmd_animate(args: argparse.Namespace, cfg: dict) -> int:
    from .pipeline import animate_to_folder

    if args.debug_regions:
        return _debug_regions(cfg)
    out, frames, prev = animate_to_folder(cfg, args.out)
    print(f"Wrote {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]}) to {out}")
    print(f"Preview: {prev}\nNext: python -m emote process {out}")
    return 0


def _print_summary(res: dict) -> int:
    print(f"Frames: {res['frames']}   fps requested {res['fps_requested']:g} "
          f"(GIF effective {res['fps_gif_effective']:.2f}, whole centiseconds)")
    for p, n, dims in res["files"]:
        print(f"  {str(p):40s} {dims:9s} {n / 1024:8.1f} KB")
    for n in res.get("notes", []):
        print(f"  note: {n}")
    print(f"Twitch specs: {res['specs_source']}")
    for w in res["warnings"]:
        print(f"  WARNING: {w}")
    if not res["warnings"]:
        print("  all GIFs within Twitch limits")
    return 1 if res["warnings"] and res["strict"] else 0


def cmd_generate(args: argparse.Namespace, cfg: dict) -> int:
    from .aigen import generate

    out = generate(cfg, args.out)
    print(f"Wrote AI frames to {out} (raw_XX.png = unprocessed model output)\nNext: python -m emote process {out}")
    return 0


def cmd_ui(args: argparse.Namespace, cfg: dict) -> int:
    from .ui import serve

    serve(args.config, args.port, args.work, not args.no_open)
    return 0


def cmd_process(args: argparse.Namespace, cfg: dict) -> int:
    from .pipeline import process

    cfg["strict"] = args.strict
    return _print_summary(process(cfg, args.frames, args.sheet, args.out, args.name))


def cmd_run(args: argparse.Namespace, cfg: dict) -> int:
    from .pipeline import run

    cfg["strict"] = args.strict
    print(f"[1/2] frames via '{cfg['run']['backend']}' backend")
    res = run(cfg, args.out)
    print("[2/2] process")
    return _print_summary(res)
    return 0


def cmd_make_test_frames(args: argparse.Namespace, cfg: dict) -> int:
    from .pipeline import make_test_frames

    for p in make_test_frames(cfg, args.out):
        print(p)
    return 0


def cmd_gpu(args: argparse.Namespace, cfg: dict) -> int:
    for k, v in gpu_report().items():
        print(f"{k:28s} {v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="emote", description="Local pixel-art emote animation pipeline")
    ap.add_argument("-c", "--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("-s", "--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. -s animate.head_amp=2 (repeatable)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="sprite size, palette, region boxes")
    p.add_argument("image", nargs="?", help="sprite path (default: paths.base from config)")
    p.add_argument("--upscale-preview", action="store_true", help="also write the upscaled chroma-key image to out/")
    p.add_argument("--debug-regions", action="store_true", help="draw the region boxes to out/debug_regions.png")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("clean", help="recover a clean native pixel sprite from a noisy/upscaled image")
    p.add_argument("image", nargs="?", help="source image (default: paths.base)")
    p.add_argument("-o", "--output", help="output path (default: overwrite paths.base, keeping assets/original.png)")
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("animate", help="stage 2a: procedural walk cycle -> frames/walk")
    p.add_argument("--out", help="output folder (default: animate.out)")
    p.add_argument("--debug-regions", action="store_true",
                   help="draw the region boxes over the sprite to out/debug_regions.png and exit")
    p.set_defaults(func=cmd_animate)

    p = sub.add_parser("generate", help="stage 2b: AI pose frames via local ComfyUI (FLUX.2 klein) -> frames/ai")
    p.add_argument("--out", help="output folder (default: ai.out)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("ui", help="local web app: upload a sprite, describe it, generate the sprite sheet")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--work", default="ui_jobs", help="folder where each job's files are kept")
    p.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("process", help="stages 3-6 on a folder of frames (or a sheet)")
    p.add_argument("frames", nargs="?", help="folder of frame PNGs (default: paths.frames_dir)")
    p.add_argument("--sheet", help="sprite sheet to split (uses process.rows/cols)")
    p.add_argument("--out", help="output folder (default: paths.out_dir)")
    p.add_argument("--name", help="output base name (default: export.name)")
    p.add_argument("--strict", action="store_true", help="exit with status 1 if any Twitch-spec warning is raised")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("run", help="full pipeline end to end from config.yaml (frames -> export)")
    p.add_argument("--out", help="output folder (default: paths.out_dir)")
    p.add_argument("--strict", action="store_true", help="exit with status 1 if any Twitch-spec warning is raised")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("make-test-frames", help="write misaligned, noisy hand-made test frames")
    p.add_argument("--out", default="frames/test")
    p.set_defaults(func=cmd_make_test_frames)

    p = sub.add_parser("gpu", help="report CUDA / onnxruntime status")
    p.set_defaults(func=cmd_gpu)
    return ap


def _hoist_globals(argv: list[str]) -> list[str]:
    """Move -c/--config and -s/--set (with their values) before the subcommand so they work anywhere."""
    glob, rest, it = [], [], iter(argv)
    for tok in it:
        if tok in ("-c", "--config", "-s", "--set"):
            glob += [tok, next(it, "")]
        elif tok.startswith(("--config=", "--set=")):
            glob.append(tok)
        else:
            rest.append(tok)
    return glob + rest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(_hoist_globals(sys.argv[1:] if argv is None else argv))
    try:
        return args.func(args, load_config(args.config, args.set))
    except EmoteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
