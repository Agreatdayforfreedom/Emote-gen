import numpy as np

from emote.snap import downscale_vote, snap_frames

PAL = [(10, 20, 30), (200, 100, 50)]


def _blocky(noise_px=0):
    """4x4 native sprite upscaled x8 (so 32x32), with an optional misaligned stripe of the wrong colour."""
    nat = np.zeros((4, 4, 4), np.uint8)
    nat[1:3, 1:3] = (*PAL[0], 255)
    nat[1, 2] = (*PAL[1], 255)
    big = np.repeat(np.repeat(nat, 8, 0), 8, 1)
    if noise_px:  # smear one block edge by a few pixels, as imperfect AI grid alignment would
        big[8:16, 16:16 + noise_px] = (*PAL[0], 255)
    return nat, big


def test_vote_recovers_native_grid_despite_misaligned_edges():
    nat, big = _blocky(noise_px=3)
    out = downscale_vote(big, 8, PAL)
    assert (out == nat).all()


def test_vote_snaps_off_palette_colors_and_drops_sparse_blocks():
    nat, big = _blocky()
    big[..., :3] = np.clip(big[..., :3].astype(int) + 7, 0, 255).astype(np.uint8)  # colour drift
    big[0:8, 0:3] = (255, 0, 0, 255)  # <50% of a transparent block is opaque -> stays transparent
    out = downscale_vote(big, 8, PAL)
    assert out[0, 0, 3] == 0
    assert {tuple(int(v) for v in p[:3]) for p in out.reshape(-1, 4) if p[3]} <= set(PAL)


def test_vote_align_recovers_grid_offset_of_several_pixels():
    nat = np.zeros((10, 10, 4), np.uint8)
    nat[2:8, 2:8] = (*PAL[0], 255)
    nat[4:6, 4:6] = (*PAL[1], 255)
    nat[2, 7] = nat[7, 2] = (*PAL[1], 255)
    big = np.repeat(np.repeat(nat, 16, 0), 16, 1)
    shifted = np.roll(np.roll(big, 6, axis=0), 7, axis=1)  # grid now starts 6 px down, 7 px right
    out = downscale_vote(shifted, 16, PAL, align=True)
    ys, xs = np.nonzero(out[..., 3])
    cut = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    assert (cut == nat[2:8, 2:8]).all()


def test_despeckle_removes_small_islands_only():
    from emote.snap import remove_specks

    f = np.zeros((10, 10, 4), np.uint8)
    f[2:6, 2:6] = (*PAL[0], 255)          # 16 px body
    f[8, 8] = (*PAL[1], 255)              # 1 px speck
    f[8, 1:3] = (*PAL[1], 255)            # 2 px speck
    f[6, 6] = (*PAL[1], 255)              # diagonal neighbour of the body: stays (8-connected)
    out = remove_specks(f, 4)
    assert out[8, 8, 3] == 0 and out[8, 1, 3] == 0 and out[8, 2, 3] == 0
    assert out[6, 6, 3] == 255 and (out[2:6, 2:6] == f[2:6, 2:6]).all()
    assert (remove_specks(f, 0) == f).all()


def test_snap_frames_vote_mode_matches_center_mode_on_clean_input():
    nat, big = _blocky()
    a, _ = snap_frames([big], 8, PAL, (4, 4), anchor="none", canvas_pad="auto", downscale="center")
    b, _ = snap_frames([big], 8, PAL, (4, 4), anchor="none", canvas_pad="auto", downscale="vote")
    assert (a[0] == b[0]).all() and (a[0] == nat).all()
