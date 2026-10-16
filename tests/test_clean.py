import numpy as np

from emote.clean import clean_sprite, detect_grid
from emote.sprite import upscale_nn


def _synthetic():
    rng = np.random.default_rng(0)
    pal = np.array([(40, 40, 60), (170, 140, 120), (240, 230, 230)], np.uint8)
    sprite = pal[rng.integers(0, 3, (12, 12))]
    img = np.full((20, 20, 3), 255, np.uint8)
    img[4:16, 4:16] = sprite
    return img, sprite


def test_grid_detected_and_sprite_recovered():
    img, sprite = _synthetic()
    big = upscale_nn(np.dstack([img, np.full(img.shape[:2], 255, np.uint8)]), 5)[..., :3]
    grid = detect_grid(big)
    assert abs(grid.period_x - 5) < 0.2 and abs(grid.period_y - 5) < 0.2
    out, _ = clean_sprite(big, colors=3, pad=0, remove_shadow=False)
    assert out.shape[:2] == (12, 12)
    assert (out[..., :3] == sprite).all()
