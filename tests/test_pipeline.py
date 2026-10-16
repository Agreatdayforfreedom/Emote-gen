import hashlib

import numpy as np
import pytest
from PIL import Image

from emote.config import load_config
from emote.export import fit_square
from emote.pipeline import make_test_frames, process
from emote.sprite import extract_palette, load_rgba
from emote.split import split_sheet
from emote.sprite import EmoteError

CFG = load_config("config.yaml")
CFG["process"]["snap"]["crop"] = False  # these tests compare against the uncropped base sprite
PALETTE = set(extract_palette(load_rgba(CFG["paths"]["base"])))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("t")
    make_test_frames(CFG, d / "frames")
    res = process(CFG, d / "frames", out_dir=d / "out")
    return d, res


def _png_frames(d):
    return [load_rgba(p) for p in sorted((d / "out" / "frames").glob("*.png"))]


def test_processed_frames_use_only_base_palette(run):
    d, _ = run
    for f in _png_frames(d):
        assert {tuple(int(v) for v in p[:3]) for p in f.reshape(-1, 4) if p[3]} <= PALETTE
        assert set(np.unique(f[..., 3])) <= {0, 255}  # binary alpha, no halos


def test_exports_only_contain_palette_colors_and_binary_alpha(run):
    d, _ = run
    for p in (d / "out").glob("emote_*.png"):
        if "preview" in p.name:
            continue
        with Image.open(p) as im:
            for i in range(im.n_frames):
                im.seek(i)
                a = np.array(im.convert("RGBA"))
                assert {tuple(int(v) for v in x[:3]) for x in a.reshape(-1, 4) if x[3]} <= PALETTE, p
                assert set(np.unique(a[..., 3])) <= {0, 255}, p


def test_gifs_only_contain_palette_colors(run):
    d, _ = run
    for p in (d / "out").glob("emote_*.gif"):
        with Image.open(p) as im:
            for i in range(im.n_frames):
                im.seek(i)
                a = np.array(im.convert("RGBA"))
                assert {tuple(int(v) for v in x[:3]) for x in a.reshape(-1, 4) if x[3]} <= PALETTE, p


def test_anchor_alignment_removes_jitter(run):
    d, _ = run
    frames = _png_frames(d)
    base = load_rgba(CFG["paths"]["base"])
    lower = slice(base.shape[0] // 2, None)
    for f in frames:  # test frames were shifted by whole pixels; the feet anchor undoes it
        assert f.shape == base.shape
        assert (f[lower] == base[lower]).all()  # body/feet back on the original sprite, pixel for pixel
    assert not (frames[1] == frames[2]).all()  # head bob survives alignment


def test_export_sizes_and_frame_count(run):
    d, res = run
    assert res["frames"] == 4
    for s in (112, 56, 28):
        with Image.open(d / "out" / f"emote_{s}.gif") as im:
            assert im.size == (s, s) and im.n_frames == 4
    assert not res["warnings"]


def test_fit_square_centres_without_stretching():
    f = np.zeros((10, 20, 4), np.uint8)
    f[..., 3] = 255
    out = fit_square(f, 112)
    ys, xs = np.nonzero(out[..., 3])
    assert (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) == (100, 50)  # integer x5, aspect kept
    assert abs(xs.min() - (112 - xs.max() - 1)) <= 1 and abs(ys.min() - (112 - ys.max() - 1)) <= 1


def test_deterministic(tmp_path):
    hashes = []
    for i in range(2):
        make_test_frames(CFG, tmp_path / f"f{i}")
        process(CFG, tmp_path / f"f{i}", out_dir=tmp_path / f"o{i}")
        hashes.append(hashlib.sha256((tmp_path / f"o{i}" / "emote_112.gif").read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


def test_split_sheet_grid_math_and_errors():
    sheet = np.arange(4 * 6 * 4, dtype=np.uint8).reshape(4, 6, 4)
    cells = split_sheet(sheet, 2, 3)
    assert len(cells) == 6 and cells[0].shape == (2, 2, 4)
    assert (cells[4] == sheet[2:4, 2:4]).all()
    with pytest.raises(EmoteError):
        split_sheet(sheet, 3, 3)
