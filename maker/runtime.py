"""Inference using the unmodified NVIDIA generator and reference operators."""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import platform
import shutil

from .models import MakerError


def environment() -> dict:
    import numpy
    import scipy
    import torch
    from PIL import __version__ as pillow_version
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "numpy": numpy.__version__,
        "scipy": scipy.__version__, "pillow": pillow_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvcc": shutil.which("nvcc"),
    }


def select_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise MakerError("CUDA is not available. Check the NVIDIA driver, or use --device cpu.")
    return requested


@contextmanager
def operators(impl: str):
    # Reference implementations are supplied by NVIDIA. Force all three together:
    # filtered_lrelu's reference implementation also calls bias_act and upfirdn2d.
    from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d
    originals = []
    try:
        if impl == "ref":
            for module, name in ((bias_act, "bias_act"), (filtered_lrelu, "filtered_lrelu"), (upfirdn2d, "upfirdn2d")):
                original = getattr(module, name)
                @wraps(original)
                def reference(*args, _original=original, **kwargs):
                    kwargs["impl"] = "ref"
                    return _original(*args, **kwargs)
                originals.append((module, name, original))
                setattr(module, name, reference)
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def load_generator(path, spec, device):
    import torch
    from safetensors.torch import load_file
    from training.networks_stylegan3 import Generator
    # Initialization is overwritten in full by strict state loading.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        generator = Generator(**spec["generator_kwargs"])
    generator.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    if (generator.c_dim, generator.img_resolution, generator.img_channels) != (6, 512, 3):
        raise MakerError("Expected a six-class, 512×512 RGB StyleGAN3 generator.")
    return generator.eval().requires_grad_(False).to(device)


def synthesize(generator, grade, seed, truncation, device):
    import numpy as np
    import torch
    # Match gen_images.py's independent RandomState for each seed.
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, generator.z_dim)).to(device)
    label = torch.zeros([1, generator.c_dim], device=device)
    label[:, grade] = 1
    with torch.no_grad():
        tensor = generator(z, label, truncation_psi=truncation, noise_mode="const", force_fp32=True)
    if not torch.isfinite(tensor).all():
        raise MakerError("Generator produced non-finite pixels; no image was saved.")
    return (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
