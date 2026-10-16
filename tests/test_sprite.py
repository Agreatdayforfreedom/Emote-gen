import numpy as np

from emote.sprite import bounding_box, extract_palette, on_background, upscale_nn


def _sprite():
    a = np.zeros((4, 4, 4), np.uint8)
    a[1:3, 1:3] = (10, 20, 30, 255)
    a[1, 1] = (200, 100, 50, 255)
    return a


def test_palette_unique_opaque_only():
    assert set(extract_palette(_sprite())) == {(10, 20, 30), (200, 100, 50)}


def test_upscale_is_pure_repetition():
    s = _sprite()
    up = upscale_nn(s, 8)
    assert up.shape == (32, 32, 4)
    assert (up[::8, ::8] == s).all()
    assert {tuple(p) for p in up.reshape(-1, 4)} == {tuple(p) for p in s.reshape(-1, 4)}


def test_background_adds_no_new_colors_besides_key():
    out = on_background(_sprite(), (0, 255, 0))
    assert {tuple(p[:3]) for p in out.reshape(-1, 4)} == {(10, 20, 30), (200, 100, 50), (0, 255, 0)}


def test_bbox():
    assert bounding_box(_sprite()) == (1, 1, 3, 3)
