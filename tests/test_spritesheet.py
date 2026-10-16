import json

import numpy as np

from emote.config import load_config
from emote.export import save_spritesheet
from emote.pipeline import run
from emote.sprite import extract_palette, load_rgba


def _frames(n=5):
    out = []
    for i in range(n):
        f = np.zeros((6 + i % 2, 8, 4), np.uint8)  # slightly different heights
        f[1:4, 1:5] = (10 * i + 5, 20, 30, 255)
        out.append(f)
    return out


def test_sheet_layout_atlas_and_pixel_exactness(tmp_path):
    frames = _frames()
    p = save_spritesheet(frames, tmp_path / "s.png", 8, cols=3)
    atlas = json.loads(p.with_suffix(".json").read_text())
    sheet = load_rgba(p)
    assert (atlas["columns"], atlas["rows"], atlas["frame_count"], atlas["fps"]) == (3, 2, 5, 8)
    assert sheet.shape[:2] == (atlas["rows"] * atlas["frame_height"], atlas["columns"] * atlas["frame_width"])
    for fr, f in zip(atlas["frames"], frames):  # every frame is recoverable from its atlas rect
        cell = sheet[fr["y"]:fr["y"] + fr["h"], fr["x"]:fr["x"] + fr["w"]]
        ys, xs = np.nonzero(cell[..., 3])
        assert (cell[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == f[1:4, 1:5]).all()
    assert {tuple(int(v) for v in x[:3]) for x in sheet.reshape(-1, 4) if x[3]} == \
           {tuple(int(v) for v in x[:3]) for f in frames for x in f.reshape(-1, 4) if x[3]}


def test_single_row_default_and_integer_scale(tmp_path):
    p = save_spritesheet(_frames(4), tmp_path / "s.png", 8, cols=0, scale=3)
    atlas = json.loads(p.with_suffix(".json").read_text())
    assert atlas["rows"] == 1 and atlas["columns"] == 4 and atlas["frame_width"] == 24
    assert load_rgba(p).shape[1] == 4 * 24


def test_run_exports_spritesheet_with_palette_colors(tmp_path):
    cfg = load_config("config.yaml", [f"paths.out_dir={tmp_path / 'out'}", f"animate.out={tmp_path / 'w'}"])
    run(cfg)
    sheet = load_rgba(tmp_path / "out" / "emote_sheet.png")
    assert {tuple(int(v) for v in x[:3]) for x in sheet.reshape(-1, 4) if x[3]} <= set(
        extract_palette(load_rgba("assets/base.png")))
    assert json.loads((tmp_path / "out" / "emote_sheet.json").read_text())["frame_count"] == 4
