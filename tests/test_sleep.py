import numpy as np
import pytest
from PIL import Image

from emote.config import load_config
from emote.pipeline import run
from emote.regions import load_regions, require_regions
from emote.sleep import SleepParams, breath_offsets, close_eyes, sleep_cycle, zzz_glyphs
from emote.sprite import EmoteError, extract_palette, load_rgba

CFG = load_config("config.yaml")
BASE = load_rgba(CFG["paths"]["base"])
REG = load_regions(CFG["paths"]["regions"], (BASE.shape[1], BASE.shape[0]))
PAL = set(extract_palette(BASE))
CORAL = (220, 110, 98)


def _cycle(**kw):
    return sleep_cycle(BASE, REG["head"], REG["legs"], REG["eye"], SleepParams(**kw))


def test_closed_eyes_remove_highlight_keep_silhouette_and_palette():
    out = close_eyes(BASE, REG["eye"])
    x0, y0, x1, y1 = REG["eye"]
    assert (out[..., 3] == BASE[..., 3]).all()                               # silhouette unchanged
    box = out[y0:y1, x0:x1]
    navy = (box[..., :3] == (53, 49, 73)).all(-1) & (box[..., 3] > 0)
    assert 3 <= navy.sum() <= 6                                              # a short arc, not a blob
    assert (box[..., :3][~navy & (box[..., 3] > 0)] == (168, 153, 147)).all()  # everything else = head colour
    assert not (box[..., :3] == (242, 233, 231)).all(-1).any()               # the white highlight is gone
    assert {tuple(int(v) for v in p[:3]) for p in out.reshape(-1, 4) if p[3]} <= PAL
    ys, xs = np.nonzero(navy)                                                # arc shape: ends one row above the middle
    assert ys.min() < ys.max() and xs.max() - xs.min() == 3


def test_close_eyes_clear_errors():
    with pytest.raises(EmoteError, match="no dark eye pixels"):
        close_eyes(BASE, (20, 0, 25, 5))     # empty corner of the canvas, no eye
    with pytest.raises(EmoteError, match="No head colour"):
        close_eyes(_dark_only(), (1, 1, 4, 4))


def _dark_only():
    s = np.zeros((6, 6, 4), np.uint8)
    s[1:4, 1:4] = (30, 30, 40, 255)   # dark sprite, no non-dark colour around the eye box
    return s


def test_breath_is_periodic_slow_and_head_nods_only_at_the_deepest_point():
    p = SleepParams()
    body = [breath_offsets(i, p)[0] for i in range(p.frames)]
    head = [breath_offsets(i, p)[1] for i in range(p.frames)]
    assert body == [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    assert [h - b for h, b in zip(head, body)] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    for n in (4, 6, 12):
        q = SleepParams(frames=n)
        assert all(breath_offsets(i, q) == breath_offsets(i + n, q) for i in range(n))


def test_zzz_glyphs_grow_travel_and_loop():
    p = SleepParams()
    head = REG["head"]
    seen = set()
    for i in range(p.frames):
        g = zzz_glyphs(i, p, head)
        assert 2 <= len(g) <= 3
        seen |= {s for s, _, _ in g}
        assert all(3 <= s <= 5 for s, _, _ in g)
        assert zzz_glyphs(i, p, head) == zzz_glyphs(i + p.frames, p, head)
    assert seen == {3, 4, 5}
    # one glyph's path: x increases and y decreases as it ages (up and to the right)
    track = [min(zzz_glyphs(i, p, head), key=lambda g: g[0])[1:] for i in range(3)]
    assert track[-1][0] >= track[0][0] and track[-1][1] <= track[0][1]


def test_sleep_frames_are_distinct_loop_and_palette_only_with_zzz_color():
    frames = _cycle()
    assert len(frames) == 12
    assert all(not np.array_equal(frames[i], frames[(i + 1) % 12]) for i in range(12))  # incl. the loop seam
    for f in frames:
        cols = {tuple(int(v) for v in p[:3]) for p in f.reshape(-1, 4) if p[3]}
        assert cols <= PAL and set(np.unique(f[..., 3])) <= {0, 255}
    assert any(((f[..., :3] == CORAL).all(-1) & (f[..., 3] > 0)).sum() > 8 for f in frames)  # beak + feet + glyphs


def test_zzz_only_ever_draws_on_transparent_pixels():
    with_z, without = _cycle(), _cycle(zzz=False)
    for a, b in zip(with_z, without):
        changed = (a != b).any(-1)
        assert (b[changed][..., 3] == 0).all()        # every pixel Zzz touched was empty before
    assert all(not np.array_equal(a, b) for a, b in zip(with_z[:6], without[:6]))


def test_zzz_color_must_be_in_the_palette_and_few_frames_rejected():
    with pytest.raises(EmoteError, match="not a color of the sprite"):
        _cycle(zzz_color=(1, 2, 3))
    with pytest.raises(EmoteError, match="at least 4"):
        _cycle(frames=3)


def test_sleep_requires_eye_region(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"head": [2, 1, 17, 11], "legs": [7, 24, 19, 26]}')
    with pytest.raises(EmoteError, match="must define: eye"):
        require_regions(p, (29, 27), need=("head", "legs", "eye"))


def test_run_sleep_end_to_end(tmp_path):
    cfg = load_config("config.yaml", [f"paths.out_dir={tmp_path / 'out'}", f"animate.out={tmp_path / 'f'}",
                                       "animate.motion=sleep"])
    res = run(cfg)
    assert res["frames"] == 12 and not res["warnings"]
    assert abs(res["fps_gif_effective"] - 5.88) < 0.05                          # sleep's own 6 fps, not export.fps
    out = tmp_path / "out"
    with Image.open(out / "emote_sleep_112.gif") as im:
        assert im.size == (112, 112) and im.n_frames == 12
    assert (out / "emote_sleep_sheet.png").is_file() and not (out / "emote.png").exists()
    for f in sorted((out / "frames").glob("emote_sleep_*.png")):
        a = load_rgba(f)
        assert {tuple(int(v) for v in p[:3]) for p in a.reshape(-1, 4) if p[3]} <= PAL


def test_unknown_motion_is_a_clear_error(tmp_path):
    cfg = load_config("config.yaml", [f"paths.out_dir={tmp_path}", f"animate.out={tmp_path / 'f'}",
                                       "animate.motion=dance"])
    with pytest.raises(EmoteError, match="Unknown animate.motion"):
        run(cfg)
