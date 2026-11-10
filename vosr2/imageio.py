"""Image I/O helpers: files, numpy arrays and torch tensors in BHWC float [0,1].

Deliberately dependency-light: Pillow for decoding/encoding, numpy as the
interchange format. Nothing here quantises to uint8 except the final file write.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: str | Path) -> torch.Tensor:
    """Load one image file as a ``(1, H, W, 3)`` float32 tensor in [0, 1].

    EXIF orientation is applied and alpha is composited over white, so a phone
    photo comes in the right way up and a transparent PNG does not come in with
    a black background.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def load_images(paths) -> list[torch.Tensor]:
    return [load_image(p) for p in paths]


def save_image(image_bhwc01: torch.Tensor, path: str | Path, quality: int = 95) -> Path:
    """Write one image (``(H, W, 3)`` or ``(1, H, W, 3)``) in [0, 1] to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = image_bhwc01
    if x.dim() == 4:
        if x.shape[0] != 1:
            raise ValueError(f"save_image expects a single image, got batch of {x.shape[0]}.")
        x = x[0]
    arr = (x.detach().float().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    im = Image.fromarray(arr, mode="RGB")
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        im.save(path, quality=quality, subsampling=0)
    elif suffix == ".webp":
        im.save(path, quality=quality)
    else:
        im.save(path)
    return path


def to_pil(image_hwc01: torch.Tensor) -> Image.Image:
    arr = (image_hwc01.detach().float().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def from_pil(im: Image.Image) -> torch.Tensor:
    """PIL image -> ``(1, H, W, 3)`` float32 [0, 1]."""
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def from_numpy(arr: np.ndarray) -> torch.Tensor:
    """numpy HWC/CHW uint8 or float -> ``(1, H, W, 3)`` float32 [0, 1]."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype != np.float32:
        arr = arr.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def collect_inputs(paths, recursive: bool = False) -> list[Path]:
    """Expand files/directories into a sorted list of image files."""
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out.extend(sorted(q for q in it if q.suffix.lower() in SUPPORTED_SUFFIXES))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    seen, unique = set(), []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique
