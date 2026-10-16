import contextlib
import json

import numpy as np
import pytest

from emote import aigen
from emote.config import load_config
from emote.pipeline import process
from emote.sprite import EmoteError, extract_palette, load_rgba


def _cfg(tmp_path, *ov):
    return load_config("config.yaml", [f"ai.out={tmp_path / 'ai'}", f"paths.out_dir={tmp_path / 'out'}", *ov])


class FakeClient:
    """Stands in for ComfyUI: returns a 1024x1024 'sheet' whose four 512px cells hold blocky shapes on green."""

    def __init__(self):
        self.calls = []

    def upload_image(self, path, name=None):
        return name

    def run(self, workflow, node_id=None):
        self.calls.append(workflow)
        sheet = np.zeros((1024, 1024, 4), np.uint8)
        sheet[..., :3] = (0, 255, 0)
        sheet[..., 3] = 255
        for i, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            y0, x0 = r * 512 + 160, c * 512 + 160 + i * 16
            sheet[y0:y0 + 160, x0:x0 + 192, :3] = (173, 140, 120)   # a palette colour
            sheet[y0:y0 + 32, x0:x0 + 64, :3] = (242, 233, 231)
        return [sheet]


@pytest.fixture
def fake_server(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(aigen, "server", lambda cfg: contextlib.nullcontext(fake))
    monkeypatch.setattr(aigen, "check_models", lambda cfg: None)
    return fake


def test_reference_scales_by_whole_number_and_uses_chroma():
    cfg = load_config("config.yaml")
    ref, k = aigen.reference_image(cfg)
    assert ref.shape[:2] == (1024, 1024) and k == 8  # canvas 128: each of the 4 sheet cells has room for the sprite
    assert tuple(ref[0, 0, :3]) == (0, 255, 0)
    allowed = set(extract_palette(load_rgba("assets/base.png"))) | {(0, 255, 0)}
    assert {tuple(int(v) for v in p[:3]) for p in ref.reshape(-1, 4)} <= allowed


def test_reference_rejects_canvas_that_does_not_divide_working_size():
    with pytest.raises(EmoteError, match="whole number"):
        aigen.reference_image(load_config("config.yaml", ["ai.canvas=48"]))
    with pytest.raises(EmoteError, match="smaller than the sprite"):
        aigen.reference_image(load_config("config.yaml", ["ai.canvas=16"]))


def test_sheet_prompt_lists_four_frames_and_character():
    p = aigen.sheet_prompt(load_config("config.yaml"))
    assert "2x2 sprite sheet" in p and "Frame 4 (bottom-right)" in p and "#a89993" in p and "#00FF00" in p
    with pytest.raises(EmoteError, match="exactly 4 poses"):
        aigen.sheet_prompt(load_config("config.yaml", ['ai.poses=["a","b"]']))


def test_generate_sheet_mode_splits_to_four_frames_and_fixed_seed(tmp_path, fake_server):
    out = aigen.generate(_cfg(tmp_path))
    assert len(fake_server.calls) == 1  # one pass for the whole sheet
    assert fake_server.calls[0]["17"]["inputs"]["noise_seed"] == 1234  # fixed seed from config
    frames = sorted(out.glob("frame_*.png"))
    assert len(frames) == 4 and load_rgba(frames[0]).shape[:2] == (512, 512)
    assert (out / "raw" / "sheet.png").is_file() and (out / "raw" / "reference.png").is_file()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["source_scale"] == 8 and meta["downscale"] == "vote" and meta["bg_mode"] == "chroma"
    assert "15" not in fake_server.calls[0]  # LoRA node bypassed when ai.lora is null


def test_generate_frames_mode_reuses_identical_poses(tmp_path, fake_server):
    aigen.generate(_cfg(tmp_path, "ai.mode=frames"))
    assert len(fake_server.calls) == 3  # poses 2 and 4 are identical -> generated once


def test_ai_frames_flow_through_process_palette_only(tmp_path, fake_server):
    cfg = _cfg(tmp_path)
    out = aigen.generate(cfg)
    res = process(cfg, out, out_dir=tmp_path / "out", name="fly")
    pal = set(extract_palette(load_rgba("assets/base.png")))
    assert res["frames"] == 4
    for f in sorted((tmp_path / "out" / "frames").glob("fly_*.png")):
        a = load_rgba(f)
        assert {tuple(int(v) for v in p[:3]) for p in a.reshape(-1, 4) if p[3]} <= pal
        assert set(np.unique(a[..., 3])) <= {0, 255}


def test_pipeline_restore_regions_are_validated_and_only_apply_to_ai_frames(tmp_path, fake_server):
    cfg = _cfg(tmp_path, 'restore.regions={"eye": [0, 0, 99, 99]}')
    out = aigen.generate(cfg)
    with pytest.raises(EmoteError, match="inside the 29x27 sprite"):
        process(cfg, out, out_dir=tmp_path / "o")
    ok = _cfg(tmp_path, 'restore.regions={"eye": [9, 5, 14, 10]}')
    res = process(ok, out, out_dir=tmp_path / "o2", name="fly")
    assert (tmp_path / "o2" / "fly_112.gif").is_file()  # the restore loop must not clobber the output name
    assert len(res["notes"]) == 4 and all(n.startswith("restore 'eye' frame") for n in res["notes"])
    # procedural frames never use restore, even with regions configured
    from emote.pipeline import animate_to_folder
    walk, _, _ = animate_to_folder(_cfg(tmp_path, f"animate.out={tmp_path / 'walk'}"))
    assert process(ok, walk, out_dir=tmp_path / "o3")["notes"] == []


def test_unknown_mode_is_a_clear_error(tmp_path, fake_server):
    with pytest.raises(EmoteError, match="ai.mode"):
        aigen.generate(_cfg(tmp_path, "ai.mode=nope"))
