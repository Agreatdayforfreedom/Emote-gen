import numpy as np
import pytest

from emote.restore import best_match, restore_frames
from emote.snap import snap_frames
from emote.sprite import EmoteError

NAVY, GREY, CREAM, TAN = (53, 49, 73), (168, 153, 147), (242, 233, 231), (173, 140, 120)
PAL = [NAVY, GREY, CREAM, TAN]


def _sprite():
    """12x10 sprite: grey head with a navy eye that has a cream highlight, tan body."""
    s = np.zeros((10, 12, 4), np.uint8)
    s[1:6, 2:9] = (*GREY, 255)
    s[2:5, 4:7] = (*NAVY, 255)
    s[2, 4] = (*CREAM, 255)           # eye highlight
    s[6:10, 1:11] = (*TAN, 255)
    return s


def _ai_version(base, dx=3, dy=2, hl=False):
    """The 'AI redraw': same shapes shifted, but the eye highlight lost (all navy) -> what we want to fix."""
    f = np.zeros((24, 24, 4), np.uint8)
    f[dy:dy + 10, dx:dx + 12] = base
    if not hl:
        f[dy + 2, dx + 4] = (*NAVY, 255)
    return f


EYE = (4, 2, 7, 5)


def test_restore_pastes_original_eye_pixels_at_the_matching_offset():
    base = _sprite()
    ai = _ai_version(base)
    assert tuple(ai[4, 7, :3]) == NAVY  # highlight missing in the AI frame
    out, rep = restore_frames([ai], base, {"eye": EYE}, min_score=0.7)
    assert rep[0].restored and rep[0].offset == (3 + 4, 2 + 2)
    assert tuple(out[0][4, 7, :3]) == CREAM         # highlight is back, exactly the original colour
    assert (out[0][..., 3] == ai[..., 3]).all()     # only colours changed, never the silhouette


def test_restore_leaves_frame_untouched_when_match_is_poor():
    base = _sprite()
    other = np.zeros((24, 24, 4), np.uint8)
    other[5:15, 5:15] = (*TAN, 255)                 # nothing like the eye region
    out, rep = restore_frames([other], base, {"eye": EYE}, min_score=0.7)
    assert not rep[0].restored and rep[0].offset is None and (out[0] == other).all()


def test_restore_does_not_modify_input_frames():
    base = _sprite(); ai = _ai_version(base); before = ai.copy()
    restore_frames([ai], base, {"eye": EYE})
    assert (ai == before).all()


def test_empty_region_is_a_clear_error():
    with pytest.raises(EmoteError, match="no opaque pixels"):
        restore_frames([_ai_version(_sprite())], _sprite(), {"x": (10, 0, 12, 1)})  # row 0 is empty


def test_best_match_prefers_exact_location_over_partial():
    base = _sprite(); ai = _ai_version(base, hl=True)
    tmpl, mask = base[2:5, 4:7], base[2:5, 4:7, 3] > 0
    score, off = best_match(ai, tmpl, mask, 12.0)
    assert score == 1.0 and off == (7, 4)


def test_snap_frames_applies_restore_and_reports_notes():
    base = _sprite()
    ai = _ai_version(base)
    big = np.repeat(np.repeat(ai, 8, 0), 8, 1)
    notes = []
    out, _ = snap_frames([big], 8, PAL, (12, 10), anchor="none", canvas_pad="auto", downscale="vote",
                         restore=(base, {"eye": EYE}, 0.7, 12.0), notes=notes)
    assert tuple(out[0][4, 7, :3]) == CREAM
    assert notes and "pasted original pixels" in notes[0]
    plain, _ = snap_frames([big], 8, PAL, (12, 10), anchor="none", canvas_pad="auto", downscale="vote")
    assert tuple(plain[0][4, 7, :3]) == NAVY        # without restore the highlight stays lost

