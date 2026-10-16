"""GPU diagnostics. torch / onnxruntime are optional and imported lazily."""
from __future__ import annotations


def gpu_report() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch.version.cuda"] = str(torch.version.cuda)
        avail = torch.cuda.is_available()
        info["torch.cuda.is_available()"] = str(avail)
        if avail:
            info["device"] = torch.cuda.get_device_name(0)
            info["compute capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
            # Blackwell (sm_120) needs a CUDA 12.8+ build; run a real kernel to prove it.
            x = torch.ones(8, device="cuda") * 2
            info["cuda kernel test"] = "ok" if float(x.sum()) == 16.0 else "FAILED"
    except ImportError:
        info["torch"] = "not installed"
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["ort providers"] = ", ".join(ort.get_available_providers())
    except ImportError:
        info["onnxruntime"] = "not installed"
    return info
