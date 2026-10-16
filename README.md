# pipeline-sprite

Local, scriptable pipeline that turns one pixel-art base emote (`assets/base.png`) into a looping
animated Twitch emote (GIF + APNG). No paid APIs. **Pixel-exact:** nearest-neighbor only (never bicubic /
bilinear), every output pixel is a color from the base sprite's palette, alpha is strictly 0 or 255.

```
base.png -> [1] prep -> [2] frames -> [3] split -> [4] bg removal -> [5] pixel-snap -> [6] assemble/export
```

## Setup (Windows 11, PowerShell, Python 3.13)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m emote gpu        # GPU check
```

Use `.\.venv\Scripts\python.exe -m emote ...` or activate with `.\.venv\Scripts\Activate.ps1` and use `python -m emote ...`.

### PyTorch / CUDA 12.8 note (RTX 50-series)

The RTX 5070 is Blackwell (compute capability `sm_120`). It **needs a PyTorch build compiled for CUDA 12.8 or
newer**; the default PyPI wheel will either run on CPU only or fail with "no kernel image". `requirements.txt` pins
`torch==2.9.1+cu128` and points pip at `https://download.pytorch.org/whl/cu128` (about 2.7 GB, BSD-3-Clause).
`python -m emote gpu` prints `torch.cuda.is_available()`, the device and compute capability, and runs a real
kernel to prove the GPU works. Torch is only needed for the optional AI backend / GPU checks; the procedural
pipeline does not use it.

The optional AI background-removal mode (`bg.mode: ai`) additionally needs `rembg` and `onnxruntime-gpu`
plus a BiRefNet model download (hundreds of MB). They are intentionally not in `requirements.txt`; the pipeline
errors with instructions if you select `ai` without them and refuses to run it on CPU by accident.

## Quick start

```powershell
python -m emote clean            # only if base.png is a noisy / upscaled image (keeps assets/original.png)
python -m emote inspect          # size, palette, regions
python -m emote animate --debug-regions   # -> out/debug_regions.png, tune assets/regions.json
python -m emote run              # full pipeline -> out/
```

## Commands

All commands read `config.yaml`. Override any value for one run with `-s key=value` (repeatable, works before
or after the command): `python -m emote run -s animate.head_amp=2 -s export.fps=10`. `-c other.yaml` picks another config.

| Command | What it does |
| --- | --- |
| `inspect [image] [--debug-regions] [--upscale-preview]` | Native size, unique-color palette with counts, bounding box, region boxes. Warns about semi-transparent pixels or a noisy palette. |
| `clean [image] [-o out]` | Recovers a clean native sprite from a blurred / re-compressed upscale: detects the pixel-block grid, samples one color per block, clusters to a small palette (Lab), flood-fills the flat background, drops a light-blue ground shadow and stray specks. Original is kept as `assets/original.png`. Tune in `clean:` (`block`, `offset`, `colors`, `merge_delta_e`, `bg_tolerance`, `remove_shadow`). |
| `animate [--out dir] [--debug-regions]` | Stage 2a: procedural walk cycle -> `frames/walk/frame_XX.png` (+ `meta.json`, `out/walk_preview.png`). |
| `process [frames_dir] [--sheet file] [--out dir] [--name n] [--strict]` | Stages 3-6 on a folder of frames, or on a sprite sheet split with `process.rows` x `process.cols`. |
| `generate [--out dir]` | Stage 2b: AI pose frames through local ComfyUI -> `frames/ai`. |
| `run [--out dir] [--strict]` | Everything end to end: validates `base.png`, makes frames with `run.backend` (`procedural` or `ai`), then stages 3-6. |
| `make-test-frames` | Writes noisy, deliberately misaligned frames to `frames/test` (chroma-green background, 8x) for testing stages 3-6. |
| `ui [--port 8765] [--no-open]` | Local web app (see below): upload a sprite, describe it, pick four poses, generate. |
| `gpu` | torch / CUDA / onnxruntime status. |

`--strict` makes the exit status 1 when any Twitch-spec warning is raised (useful in scripts).

## Outputs (`out/`)

- `emote_112.gif`, `emote_56.gif`, `emote_28.gif`, `emote_native.gif` - transparent GIFs.
- `emote_112.png`, ... - transparent APNGs (not accepted by Twitch for animated emotes; handy elsewhere).
- `emote_preview.png` - contact sheet: the 28 px version on `#18181b` and `#efeff1` (plus 4x zoom of each) and the full-size frames.
- `frames/emote_XX.png` - the final native-size frames.

Sizes are fit into a square without stretching, centered on transparent padding. Upscales use the largest
whole-number factor (crisp pixels, `export.integer_upscale`); a downscale such as 27 px -> 28 px box necessarily uses nearest sampling.

Frame time: GIF delays are whole centiseconds with a browser minimum of 20 ms, so 8 fps becomes 120 ms
(8.33 fps). The summary prints the effective rate; a warning appears if it is more than 10% off.

## Twitch limits

`twitch_specs.json` stores the limits with source URL and date checked
(<https://help.twitch.tv/s/article/emote-guidelines>, checked 2026-10-14):

- animated emotes: **GIF** only, square, 28x28 / 56x56 / 112x112 (manual upload) or one 112-4096 px GIF (auto-resize);
- each manual-upload file <= **512 KB** (<= 1 MB for the auto-resize file);
- <= **60 frames**; no more than **3 flashes per second** (not checked automatically; walk cycles are fine, avoid strobing);
- no frame-rate or duration limit is published (stored as `null`).

Every run checks the exported GIFs against this file (dimensions, size, GIF frame count) and prints warnings.
It also warns when the file is older than 90 days: re-read the page and update the JSON.

## AI backend: new poses from one sprite (ComfyUI + FLUX.2 klein 4B)

The procedural animator can only move existing pixels. For poses that need new drawing (e.g. a flying pigeon) the
`ai` backend asks a local image-edit model to draw them, then the normal stages clean the result up.

```powershell
python -m emote generate           # stage 2b only -> frames/ai (raw model output in frames/ai/raw)
python -m emote run -s run.backend=ai   # generate + process -> out/emote_fly_*.gif / _sheet.png
```

How it works (`emote/aigen.py`, `emote/comfy.py`, `workflows/klein_edit.json`):

1. The base sprite is centred on a 64x64 canvas, upscaled by a whole number (x16) onto chroma green = 1024x1024.
2. A fixed prompt (character description with the exact palette hex colors, plus the poses) and a **fixed seed** go to a
   local ComfyUI server through its HTTP API. `ai.mode: sheet` (default) makes **one 2x2 sheet** with all four poses in
   a single pass, which keeps the character far more consistent than `ai.mode: frames` (one edit per pose, which drifts:
   different head color, proportions, even species).
3. The sheet is split 2x2, chroma-keyed, the model's pixel grid is re-found per frame (`snap.downscale: vote`:
   block majority vote), every pixel is snapped to the base palette (Lab) and frames are aligned. Output is
   pixel-exact and palette-only, exactly like the procedural path (the same tests apply).

Setup (one time, ~13.5 GB, Windows): ComfyUI portable (NVIDIA build) unpacked at `ai.comfy_dir`
(default `../comfyui/app/ComfyUI_windows_portable`) with these files in its `ComfyUI/models` folders:

| Folder | File | Size | License |
| --- | --- | --- | --- |
| `diffusion_models` | `flux-2-klein-4b-fp8.safetensors` ([black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8)) | 3.79 GB | Apache 2.0 |
| `text_encoders` | `qwen_3_4b.safetensors` ([Comfy-Org/flux2-klein-4B](https://huggingface.co/Comfy-Org/flux2-klein-4B)) | 7.49 GB | Apache 2.0 |
| `vae` | `flux2-vae.safetensors` ([Comfy-Org/flux2-dev](https://huggingface.co/Comfy-Org/flux2-dev)) | 0.31 GB | Apache 2.0 |

The pipeline starts ComfyUI itself if it is not running (`ai.autostart`, about 25 s) and stops it afterwards; start it
yourself (`run_nvidia_gpu.bat`) to keep it warm between experiments (about 12 s per generation on an RTX 5070).

Tuning: `ai.poses` (four pose phrases), `ai.character`, `ai.seed`, `ai.despeckle`, `ai.anchor`, `ai.canvas`.

What the tuning grid showed (3 pose sets x 2 character prompts x 4 seeds, all with the same pipeline):

- **Chest color:** naming the palette hex colors is not enough; the model still paints an orange chest, which the
  palette snap turns coral. Adding "only the beak and the feet are coral-orange; the chest and body have NO orange or
  red" to `ai.character` fixed it in every run.
- **Wing variety:** "wings swept down below its body" never produced a down pose (it gave two alternating poses).
  Phrasing the poses as a flap cycle (`fully raised straight up` / `halfway up, angled 45 degrees` / `folded down
  along the sides` / `halfway up`) did produce a real folded-wings frame. Very different poses (flat T-shape,
  V-shape) vary more but the bird's proportions drift more, so they are not the default. The folded frame reads a bit
  like a perched bird.
- **Seeds:** results vary a lot by seed. Seeds 5 and 42 failed outright (the model did not draw the green background,
  so the chroma key removed nothing); 1234 and 99 worked. If the output looks wrong, try another `ai.seed` first.
- `ai.despeckle` (default 4) drops stray AI pixel islands smaller than that many pixels (`snap.despeckle` for `process`).

Other limits: the character can still drift slightly between cells, and there is no guarantee a seed that works for
one sprite works for another. `fal/flux-2-klein-4b-spritesheet-lora` (Apache 2.0) was tried in
mind but it makes multi-view sheets (isometric / side / top-down on red), not animation frames, and targets the
non-distilled base model, so it is not used (`ai.lora` stays null).

## Fidelity: keeping the AI frames faithful to your sprite

Two things make the AI frames match the original (measured on the pigeon with 6 motions x 2 seeds):

1. **Give each pose enough room (`ai.canvas`).** In sheet mode every pose gets one quarter of the 1024 px image. With the
   sprite on a 64 px canvas (16 px blocks) the model shrank the pigeon to about 75% of its size, so it was drawn with
   far fewer pixels and details like the eye highlight were lost (black eyes). With a 128 px canvas (8 px blocks, each
   cell about 1.6x the sprite) the bird comes back at your sprite's size with its eye highlight and true colors.
   Rule of thumb: `canvas` >= 3.2 x the longest sprite side, from 64/128/256 (the web app picks it; max sprite 80 px).
2. **Restore exact pixels (`restore:` in `config.yaml`).** For regions you name (default: the pigeon's eye) the original
   pixels are searched for in each frame (translation only) and pasted back wherever at least `min_score` (0.7) of the
   region already matches. Where the AI drew the head tilted or very differently the frame is left untouched, and a
   `note:` line reports it. It applies to AI frames only (procedural frames already keep your exact pixels).
   In the web app, drag a box over the detail on the sprite preview instead.

Measured: with the finer canvas the eye region matched at 0.68-0.88 (so the paste applies); at the old canvas the
whole-head match scored only 0.31-0.48, which is why pasting the whole head failed (double heads) and small features
are matched instead. Over 12 runs the eye was restored in 30 of 48 frames; it is skipped where the head is bent
(peck) or the AI drew a different eye shape. Still seed-dependent: some seeds paint an orange chest (seed 1234 on
"Wave"), others keep the character on-palette (seed 99). Jump and Hop drift the most between frames.

### Cleaning up the AI's pixel art: thin outlines and flat shading

The model draws its outlines about 2 px thick at this grid and speckles shading with noisy patches. Two post-processing
steps (on for AI frames: `ai.thin_outline`, `ai.flatten`; `process.snap.thin_outline` / `.flatten` elsewhere) fix that
without ever adding a color:

- **Outline thinning** (`snap.thin_outline`): the outline is recognised as dark palette colors. Where it forms 2x2 blocks
  of dark pixels (the signature of a band 2+ px thick; a true 1 px outline, a diagonal staircase and a 1 px inner detail
  line never do) the inner layers are replaced by the neighbouring body color, so only the outermost 1 px ring stays.
  The silhouette is never changed.
- **Flatten** (`snap.flatten`): a pixel adopts its neighbours' color when at least 6 of its 8 neighbours share one other
  color that is close in Lab (<= 25). Distant colors (an eye highlight inside navy) are never touched.

Measured with `emote.metrics.outline_summary` (mean outline thickness in px; share of boundary that is exactly 1 px;
share with no dark outline at all) over 12 AI runs against the original sprite:

| | mean | 1 px | missing |
| --- | --- | --- | --- |
| original sprite | 1.47 | 72% | 0% |
| AI, no cleanup | 1.56 | 55% | 3% |
| AI, with thinning + flatten | 1.46 | 63% | 3% |

**Prompt A/B (6 runs per variant, all polished):** asking for "a complete, unbroken one-pixel outline" did not change the
thickness (1.45-1.48 for all wordings, so the post-processing does that job), but adding "flat shading, no dithering,
no extra dark lines" made the model drop the outline in 14% of edge positions (and produced tan fringes), so that
wording was removed. Lesson: do not tell the model to omit dark lines; clean up afterwards.

## Web app: upload a sprite, describe it, generate the sheet

```powershell
python -m emote ui          # or double-click start_ui.bat; opens http://127.0.0.1:8765
```

1. **Sprite:** drop any image. A noisy / upscaled image is cleaned to its true pixel grid and palette automatically,
   and a flat background is removed (sprites up to 80 px). The preview shows the cleaned sprite; optionally drag a box
   over a detail (the eye) to keep it pixel-exact (see Fidelity above).
2. **Character:** a starting description with your palette hex colors is filled in; edit it to say what the character
   is, its parts, and where each color goes and must NOT go (this is what keeps colors on-palette).
3. **Motion:** pick a preset (Flap wings, Walk, Jump, Idle bounce, Wave) or write four frame descriptions yourself,
   choose the facing, a seed (empty = random) and the fps.
4. **Generate:** the page shows progress, then the animated GIFs (112 px, and 28 px on dark and light chat), the sprite
   sheet, the raw AI sheet next to it, and downloads (zip with everything, GIFs, sheet PNG, atlas JSON).

The app starts ComfyUI on first use (about 25 s) and keeps it warm; the **Open ComfyUI** button opens ComfyUI's own
node UI at http://127.0.0.1:8188. Every job is kept in `ui_jobs/<id>/` (upload, cleaned sprite, frames, `out/`).
Jobs run one at a time because they share the GPU. For a sprite it has not seen, try a few seeds: some fail (see the
tuning notes above).

Security: the server binds to 127.0.0.1 only, rejects other Host headers and non-JSON POSTs (so other websites cannot
drive it), limits uploads to 8 MB, and serves only files inside a job's folder.

## Spritesheet export

Every run also writes `<name>_sheet.png` (all final frames packed row-major, transparent, native pixels) and
`<name>_sheet.json` (frame size, columns/rows, per-frame rects, fps, loop) for game engines. `export.spritesheet.cols`
(0 = one row) and `.scale` (integer nearest-neighbor) tune it.

## Tuning the regions (`assets/regions.json`)

```json
{ "head": [2, 1, 17, 11], "body": [1, 11, 28, 24], "legs": [7, 24, 19, 26] }
```

Boxes are `[x0, y0, x1, y1]` in native sprite pixels, **x1/y1 exclusive**. Run
`python -m emote animate --debug-regions` and open `out/debug_regions.png`: boxes are drawn over the 16x
sprite with pixel coordinates on the rulers. Edit the JSON, re-run, repeat.

How the animator uses them:

- `head`: opaque pixels in this box move as the head (thrust forward `head_amp` px on foot contact, and dip with the body).
- `legs`: pixels in this box (not already in `head`) are the feet. The box must contain **exactly two feet separated
  by at least one empty column**, otherwise you get a clear error. The foot toward the facing direction is the "front" foot.
- `body`: informational / for the overlay; every pixel not in `head` or `legs` is body.
- Draw order is body, head, legs, so a dipping body slides behind the feet.

Tips: keep the head box's bottom edge on the neck where colors blend (a 1 px seam can appear there on contact
frames; move the edge up or down a row to hide it); make the legs box start below the belly outline so the outline stays with the body.

## Sleep animation (procedural, pixel edits only)

```powershell
python -m emote run -s animate.motion=sleep      # -> out/emote_sleep_*.gif / _sheet.png (12 frames at 6 fps)
```

The same pixel-exact approach as the walk, with three parts (`emote/sleep.py`):

1. **Closed eyes:** the open eye inside the `eye` box of `assets/regions.json` (`[x0, y0, x1, y1]`, here `[9, 5, 14, 10]`) is
   replaced by the head color plus a small dark arc using the eye's own color.
2. **Breathing:** body and head settle down by `breath_amp` px for part of the loop and come back (`breaths` per loop,
   feet stay planted so the body slides behind them); at the deepest point the head nods `head_nod` px further.
3. **Zzz:** three z glyphs (3, 4 and 5 px) leave the head a third of a loop apart, growing as they drift up and to the right.
   They are drawn only on transparent pixels, in `zzz_color`, which must be a color of the sprite: coral by default, because
   grey vanished on a light chat background at 28 px.

All parameters live under `animate.sleep` in `config.yaml`. Sleep uses its own `fps` (6, so a breath takes 2 s) instead of
`export.fps`, and its files are named `emote_sleep_*` so they do not overwrite the walk. `animate.motion` is `walk` or `sleep`.

## Animator parameters (`animate:` in `config.yaml`)

`frames` (loop length, phase `i/N` so the loop is seamless), `facing`, `head_amp`, `body_dip`, `leg_swing`,
`leg_lift`, `margin` (transparent border so shifts never clip). All shifts are whole pixels.

## Background removal and snapping

- `bg.mode: chroma` keys out `prep.chroma_color` (or `bg.key`) with a Lab-distance `tolerance`; `ai` uses rembg
  (`bg.ai_model`, default `birefnet-general`) on the GPU; `none` just binarizes alpha. In every mode alpha is thresholded
  at `bg.alpha_cutoff` (default 128): no semi-transparent edges.
- Stage 5 samples the center pixel of each `source_scale` x `source_scale` block, snaps each color to the nearest
  base-palette color (Lab), then aligns frames to a common anchor (`snap.anchor`: `feet` = lowest row, centered on the
  feet; `bbox-bottom`; `none`) so the sprite does not jitter, and finally crops all frames to their union bounding
  box (`snap.crop`). Frames from the procedural animator already share a registration and use `anchor: none`
  (set automatically through `frames/walk/meta.json`).

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Covers palette exactness (processed frames, GIFs and APNGs contain only base-palette colors, alpha is binary),
upscale / downscale / split math, grid detection and sprite recovery, anchor alignment, loop periodicity and facing
mirror, region validation, determinism (same config + seed gives byte-identical output), and the spec warnings.

## Project layout

```
emote/        package (cli, config, sprite, clean, color, regions, animate, debug, split, bgremove, snap, export, specs, pipeline, gpu)
assets/       base.png (clean sprite), original.png (source), regions.json
frames/       intermediate frames
out/          exports and previews
config.yaml   all parameters
twitch_specs.json
tests/
```

## Status

All five phases are done: setup, stages 3-6, procedural animator, `run`, and the optional ComfyUI AI backend.
