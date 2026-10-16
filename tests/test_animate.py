import numpy as np
import pytest

from emote.animate import WalkParams, frame_offsets, split_feet, split_layers, walk_cycle
from emote.config import load_config
from emote.debug import draw_regions
from emote.regions import load_regions
from emote.sprite import EmoteError, extract_palette, load_rgba

CFG = load_config("config.yaml")
BASE = load_rgba(CFG["paths"]["base"])
REG = load_regions(CFG["paths"]["regions"], (BASE.shape[1], BASE.shape[0]))
PAL = set(extract_palette(BASE))


def _cycle(**kw):
    return walk_cycle(BASE, REG["head"], REG["legs"], WalkParams(**kw))


def test_loop_is_seamless_offsets_are_periodic():
    for n in (2, 3, 4, 6, 8):
        p = WalkParams(frames=n)
        for i in range(n):
            assert frame_offsets(i, p) == frame_offsets(i + n, p)


def test_neighbouring_frames_differ_and_pattern_alternates():
    f = _cycle(frames=4)
    assert all(not np.array_equal(f[i], f[(i + 1) % 4]) for i in range(4))  # incl. last -> first (the loop seam)
    p = WalkParams(frames=4)
    assert frame_offsets(0, p).body == frame_offsets(2, p).body == (0, 1)   # contact: body dips 1 px
    assert frame_offsets(1, p).body == frame_offsets(3, p).body == (0, 0)
    assert frame_offsets(0, p).head[0] == -1 and frame_offsets(1, p).head[0] == 0  # head thrusts forward (left)
    # feet alternate: front foot lifted on the frame the back foot is planted, and vice versa
    assert frame_offsets(3, p).front_foot[1] < 0 and frame_offsets(1, p).back_foot[1] < 0
    assert frame_offsets(0, p).front_foot[0] == -1 and frame_offsets(0, p).back_foot[0] == 1


def test_mirrored_when_facing_right():
    l, r = WalkParams(facing="left"), WalkParams(facing="right")
    for i in range(4):
        a, b = frame_offsets(i, l), frame_offsets(i, r)
        assert a.head == (-b.head[0], b.head[1]) and a.front_foot[0] == -b.front_foot[0]


def test_only_base_palette_colors_and_binary_alpha():
    for f in _cycle(frames=6, head_amp=2, leg_swing=2):
        assert {tuple(int(v) for v in p[:3]) for p in f.reshape(-1, 4) if p[3]} <= PAL
        assert set(np.unique(f[..., 3])) <= {0, 255}


def test_layers_partition_the_sprite_exactly():
    body, head, legs = split_layers(BASE, REG["head"], REG["legs"])
    a = [(l[..., 3] > 0) for l in (body, head, legs)]
    assert not (a[0] & a[1]).any() and not (a[0] & a[2]).any() and not (a[1] & a[2]).any()
    assert (a[0] | a[1] | a[2] == (BASE[..., 3] > 0)).all()


def test_zero_offset_frame_reproduces_base():
    f = walk_cycle(BASE, REG["head"], REG["legs"], WalkParams(frames=4, head_amp=0, body_dip=0, leg_swing=0, leg_lift=0))
    m = 2
    assert all((x[m:-m, m:-m] == BASE).all() for x in f)


def test_deterministic():
    assert all((a == b).all() for a, b in zip(_cycle(), _cycle()))


def test_bad_legs_region_gives_clear_error():
    with pytest.raises(EmoteError, match="two|expected 2"):
        split_feet(split_layers(BASE, REG["head"], (7, 24, 12, 26))[2], "left")  # only one foot inside
    with pytest.raises(EmoteError, match="no opaque"):
        split_feet(np.zeros_like(BASE), "left")


def test_margin_too_small_is_reported():
    with pytest.raises(EmoteError, match="margin"):
        _cycle(margin=0, leg_swing=2, head_amp=3)


def test_regions_validation(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"head": [5, 5, 3, 9]}')
    with pytest.raises(EmoteError, match="inverted"):
        load_regions(p)
    p.write_text('{"tail": [0, 0, 1, 1]}')
    with pytest.raises(EmoteError, match="unknown region"):
        load_regions(p)
    p.write_text('{"head": [0, 0, 99, 9]}')
    with pytest.raises(EmoteError, match="outside"):
        load_regions(p, (29, 27))


def test_debug_regions_writes_image(tmp_path):
    out = draw_regions(BASE, REG, tmp_path / "d.png")
    assert out.is_file() and out.stat().st_size > 0


def test_set_override_and_errors():
    assert load_config("config.yaml", ["animate.head_amp=2"])["animate"]["head_amp"] == 2
    assert load_config("config.yaml")["animate"]["head_amp"] == 1  # overrides must not leak into later configs
    assert load_config("nonexistent.yaml")["animate"]["head_amp"] == 1
    with pytest.raises(EmoteError):
        load_config("config.yaml", ["animate.nope=1"])
    with pytest.raises(EmoteError):
        load_config("config.yaml", ["garbage"])
