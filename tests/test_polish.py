import numpy as np

from emote.metrics import dark_mask, outline_runs, outline_summary
from emote.snap import denoise, snap_frames, thin_outline

NAVY, GREY, CREAM, TAN = (53, 49, 73), (168, 153, 147), (242, 233, 231), (173, 140, 120)


def _canvas(h=16, w=16):
    return np.zeros((h, w, 4), np.uint8)


def _box(frame, y0, y1, x0, x1, color):
    frame[y0:y1, x0:x1] = (*color, 255)


def _fat_square(t):
    """Body (grey) with an outline `t` px thick around it, on a 16x16 canvas."""
    f = _canvas()
    _box(f, 3, 13, 3, 13, NAVY)
    _box(f, 3 + t, 13 - t, 3 + t, 13 - t, GREY)
    return f


def test_thin_outline_shaves_fat_bands_to_one_pixel_keeping_silhouette_and_colors():
    for t in (2, 3):
        f = _fat_square(t)
        out = thin_outline(f, dark_mask(f))
        assert (out[..., 3] == f[..., 3]).all()                       # silhouette unchanged
        assert {tuple(int(v) for v in p[:3]) for p in out.reshape(-1, 4) if p[3]} <= {NAVY, GREY}
        # outline is now exactly the outermost 1 px ring; the removed layers took the body colour
        assert tuple(out[3, 8, :3]) == NAVY and tuple(out[4, 8, :3]) == GREY and tuple(out[5, 8, :3]) == GREY
        assert outline_summary([out])["fat"] == 0.0 and outline_summary([f])["fat"] > 0.0


def test_thin_outline_leaves_correct_one_pixel_outlines_and_details_alone():
    f = _fat_square(1)
    _box(f, 6, 9, 6, 9, NAVY)                    # enclosed dark blob (an eye): not connected to the outline
    f[8, 8] = (*CREAM, 255)                      # with a highlight
    _box(f, 4, 8, 10, 11, NAVY)                  # 1 px inner detail line touching the outline
    out = thin_outline(f, dark_mask(f))
    assert (out == f).all()


def test_thin_outline_keeps_diagonal_staircase_outline():
    f = _canvas()
    for i in range(8):                           # 1 px diagonal outline with 2 px steps, grey body below
        f[3 + i, 3 + i:5 + i] = (*NAVY, 255)
        f[3 + i, 5 + i:13] = (*GREY, 255)
    out = thin_outline(f, dark_mask(f))
    assert (out == f).all()


def test_denoise_flattens_close_colour_noise_but_keeps_distant_details():
    f = _canvas()
    _box(f, 2, 14, 2, 14, GREY)
    f[6, 6] = (*TAN, 255)                        # tan speck in a grey area: close in Lab -> flattened
    _box(f, 8, 12, 8, 12, NAVY)
    f[10, 10] = (*CREAM, 255)                    # eye-style highlight in navy: far in Lab -> kept
    out = denoise(f)
    assert tuple(out[6, 6, :3]) == GREY
    assert tuple(out[10, 10, :3]) == CREAM
    assert (out[..., 3] == f[..., 3]).all()
    assert {tuple(int(v) for v in p[:3]) for p in out.reshape(-1, 4) if p[3]} <= {NAVY, GREY, TAN, CREAM}


def test_denoise_does_not_erode_edges_or_thin_features():
    f = _canvas()
    _box(f, 4, 12, 4, 12, GREY)
    _box(f, 7, 9, 4, 12, TAN)                    # a 2 px stripe: each pixel has only 5 same-colour neighbours
    out = denoise(f)
    assert (out == f).all()


def test_metrics_outline_runs():
    one = _fat_square(1)
    assert outline_summary([one])["one_px"] == 1.0
    assert set(outline_runs(_fat_square(2)).tolist()) == {2}


def test_snap_frames_polish_flags_apply_and_stay_palette_only():
    f = _fat_square(2)
    f[8, 8] = (*TAN, 255)
    big = np.repeat(np.repeat(f, 8, 0), 8, 1)
    pal = [NAVY, GREY, TAN, CREAM]
    plain, _ = snap_frames([big], 8, pal, (16, 16), anchor="none", canvas_pad="auto", downscale="vote")
    polished, _ = snap_frames([big], 8, pal, (16, 16), anchor="none", canvas_pad="auto", downscale="vote",
                              thin=True, flatten=True)
    assert outline_summary(plain)["fat"] > 0 and outline_summary(polished)["fat"] == 0.0
    assert tuple(polished[0][8, 8, :3]) == GREY
    assert {tuple(int(v) for v in p[:3]) for p in polished[0].reshape(-1, 4) if p[3]} <= set(pal)
