"""Stage 4: background removal (chroma / ai / none) with binary alpha thresholding."""
from __future__ import annotations

import numpy as np

from .color import delta_e, rgb_to_lab
from .sprite import EmoteError

RGB = tuple[int, int, int]


def binarize_alpha(rgba: np.ndarray, cutoff: int = 128) -> np.ndarray:
    """Threshold alpha to 0/255 and zero the RGB of transparent pixels (no halos)."""
    out = rgba.copy()
    solid = out[..., 3] >= cutoff
    out[..., 3] = np.where(solid, 255, 0)
    out[~solid, :3] = 0
    return out


def chroma_key(rgba: np.ndarray, key: RGB, tolerance: float = 12.0, cutoff: int = 128) -> np.ndarray:
    """Key out `key` using Lab distance.

    Soft alpha ramps from 0 at `tolerance` dE to 255 at 2 * tolerance, then is thresholded,
    so the effective cut sits at tolerance * (1 + cutoff / 255).
    """
    if tolerance <= 0:
        raise EmoteError("chroma tolerance must be > 0")
    d = delta_e(rgb_to_lab(rgba[..., :3]), rgb_to_lab(np.array(key)))
    soft = np.clip((d - tolerance) / tolerance, 0.0, 1.0) * 255.0
    out = rgba.copy()
    out[..., 3] = np.minimum(out[..., 3], soft.astype(np.uint8))
    return binarize_alpha(out, cutoff)


def ai_remove(frames: list[np.ndarray], model: str = "birefnet-general", cutoff: int = 128) -> list[np.ndarray]:
    """rembg segmentation on GPU through onnxruntime-gpu; fails loudly if the GPU isn't used."""
    try:
        import onnxruntime as ort
        from PIL import Image
        from rembg import new_session, remove
    except ImportError as exc:
        raise EmoteError("ai background removal needs rembg + onnxruntime-gpu (see README); "
                         f"missing: {exc.name}") from exc
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise EmoteError("onnxruntime has no CUDAExecutionProvider; install onnxruntime-gpu "
                         "(and CUDA 12 / cuDNN 9 runtime libs) or use bg.mode: chroma")
    session = new_session(model, providers=["CUDAExecutionProvider"])
    active = session.inner_session.get_providers()
    if not active or active[0] != "CUDAExecutionProvider":
        raise EmoteError(f"rembg session is not running on the GPU (providers: {active})")
    out = []
    for f in frames:
        res = remove(Image.fromarray(f[..., :3], "RGB"), session=session)
        out.append(binarize_alpha(np.array(res.convert("RGBA")), cutoff))
    return out


def remove_background(frames: list[np.ndarray], mode: str, key: RGB, tolerance: float, cutoff: int,
                      ai_model: str) -> list[np.ndarray]:
    if mode == "chroma":
        return [chroma_key(f, key, tolerance, cutoff) for f in frames]
    if mode == "ai":
        return ai_remove(frames, ai_model, cutoff)
    if mode == "none":
        return [binarize_alpha(f, cutoff) for f in frames]
    raise EmoteError(f"Unknown background mode '{mode}' (use chroma, ai or none)")
