import json
from datetime import date, timedelta

import pytest
from PIL import Image

from emote.cli import main
from emote.config import load_config
from emote.pipeline import run
from emote.sprite import EmoteError


def _cfg(tmp_path, *overrides):
    return load_config("config.yaml", [f"paths.out_dir={tmp_path / 'out'}", f"animate.out={tmp_path / 'walk'}",
                                       *overrides])


def test_run_end_to_end_meets_twitch_limits(tmp_path):
    res = run(_cfg(tmp_path))
    assert res["frames"] == 4 and not res["warnings"]
    for size in (112, 56, 28):
        with Image.open(tmp_path / "out" / f"emote_{size}.gif") as im:
            assert im.size == (size, size) and im.n_frames == 4
    assert (tmp_path / "out" / "emote_preview.png").is_file()
    assert all(n <= 512 * 1024 for p, n, _ in res["files"] if p.suffix == ".gif")


def test_run_rejects_noisy_base_with_fix_hint(tmp_path):
    with pytest.raises(EmoteError, match="emote clean"):
        run(_cfg(tmp_path, "paths.base=assets/original.png"))


def test_run_ai_backend_reports_missing_models(tmp_path):
    with pytest.raises(EmoteError, match="Missing ComfyUI model files"):
        run(_cfg(tmp_path, "run.backend=ai", f"ai.comfy_dir={tmp_path}"))


def test_frame_cap_warning_uses_real_gif_frames_and_strict_exit(tmp_path):
    specs = json.loads(open("twitch_specs.json", encoding="utf-8").read())
    specs["animated"]["max_frames"] = 3
    p = tmp_path / "specs.json"
    p.write_text(json.dumps(specs))
    res = run(_cfg(tmp_path, f"paths.specs={p}"))
    assert any("exceeds max 3" in w for w in res["warnings"])


def test_size_limit_warning(tmp_path):
    specs = json.loads(open("twitch_specs.json", encoding="utf-8").read())
    specs["animated"]["max_file_bytes_manual_each"] = 500
    p = tmp_path / "specs.json"
    p.write_text(json.dumps(specs))
    res = run(_cfg(tmp_path, f"paths.specs={p}"))
    assert any("per-size limit" in w for w in res["warnings"])


def test_stale_specs_warning(tmp_path):
    specs = json.loads(open("twitch_specs.json", encoding="utf-8").read())
    specs["date_checked"] = (date.today() - timedelta(days=200)).isoformat()
    p = tmp_path / "specs.json"
    p.write_text(json.dumps(specs))
    assert any("re-verify" in w for w in run(_cfg(tmp_path, f"paths.specs={p}"))["warnings"])


def test_cli_run_and_global_flags_anywhere(tmp_path):
    out = tmp_path / "o"
    code = main(["run", "--out", str(out), "-s", f"animate.out={tmp_path / 'w'}", "--strict"])
    assert code == 0 and (out / "emote_28.gif").is_file()
    assert main(["run", "-s", "run.backend=nope"]) == 2

