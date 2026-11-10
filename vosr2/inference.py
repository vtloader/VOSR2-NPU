"""VOSR 2.0 inference: bicubic pre-scale, pad, encode, one-step denoise, decode,
color-align, crop -- with optional DiT/VAE tiling.

``forward_flexible`` (models/lightningdit.py) asserts a square input -- true of
every upstream tiled call (tiles are always cropped ``lt_size x lt_size``), but
not of a whole non-square image's latent. To support arbitrary aspect ratios
without tiling, the untiled path additionally pads to a square before the DiT
call and crops back afterward; this is invisible in the output (still cropped
to the exact requested size) and only engages when ``tile_size == 0``.

Reflect padding: ``ReflectionPad2d`` (and ``F.pad(mode="reflect")``) requires
each pad amount to be smaller than the corresponding input dimension -- see
CANN's ``aclnn_reflection_pad2d.h`` and PyTorch's own check. A 4x-upscaled
panorama easily violates that (512 px tall padded to 4096 px square needs a
3584 px pad), so :func:`_reflect_pad2d` falls back to an index-gather that
implements the same periodic reflection for arbitrarily large pads.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .color import apply_color_alignment
from .tiled_vae import _gaussian_weights, _make_tile_grid

log = logging.getLogger("vosr2.inference")

AE_FACTOR = 8
DIT_PATCH_SIZE = 2
PAD_MULTIPLE = AE_FACTOR * DIT_PATCH_SIZE  # 16


def _reflect_indices(n: int, pad_before: int, pad_after: int, device) -> torch.Tensor:
    """Indices implementing reflect padding of ``n`` samples by the given amounts.

    Reflection here is "reflect without repeating the edge", period ``2*(n-1)``,
    matching ``F.pad(mode="reflect")`` but valid for pads of any size.
    """
    period = 2 * (n - 1)
    i = torch.arange(-pad_before, n + pad_after, device=device)
    j = torch.remainder(i, period)
    j = torch.where(j >= n, period - j, j)
    return j


def _reflect_pad2d(x: torch.Tensor, left: int, right: int, top: int, bottom: int) -> torch.Tensor:
    """Reflect-pad a BCHW tensor; works when a pad exceeds the input dimension."""
    if left == right == top == bottom == 0:
        return x
    _, _, h, w = x.shape
    # Fast path: the native op when it is within its documented limits.
    if left < w and right < w and top < h and bottom < h:
        return F.pad(x, (left, right, top, bottom), mode="reflect")
    if left == right == top == bottom == 0:
        return x
    if w > 1 and (left or right):
        idx = _reflect_indices(w, left, right, x.device)
        x = x.index_select(3, idx)
    if h > 1 and (top or bottom):
        idx = _reflect_indices(h, top, bottom, x.device)
        x = x.index_select(2, idx)
    return x


def _pad_to_multiple(x: torch.Tensor, multiple: int):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return _reflect_pad2d(x, 0, pad_w, 0, pad_h)


def _pad_to_square(x: torch.Tensor):
    _, _, h, w = x.shape
    side = max(h, w)
    pad_h, pad_w = side - h, side - w
    if pad_h == 0 and pad_w == 0:
        return x
    return _reflect_pad2d(x, 0, pad_w, 0, pad_h)


def _generate_noise(shape, seed: int, device, dtype) -> torch.Tensor:
    """One item's worth of noise per call; a local CPU Generator keeps global RNG state untouched."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    return noise.to(device=device, dtype=dtype)


def _generate_noise_batch(shape_per_item, seed: int, batch_size: int, device, dtype) -> torch.Tensor:
    return torch.stack([_generate_noise(shape_per_item, seed + i, device, dtype) for i in range(batch_size)], dim=0)


def _tile_params(tile_size: int, tile_overlap: int, lh: int, lw: int):
    """Pixel tile size/overlap -> latent-space, aligned to the DiT patch size."""
    lt_size = max((tile_size // AE_FACTOR // DIT_PATCH_SIZE) * DIT_PATCH_SIZE, DIT_PATCH_SIZE)
    lt_overlap = max(tile_overlap // AE_FACTOR, lt_size // 8)
    lt_size = min(lt_size, min(lh, lw))
    lt_overlap = min(lt_overlap, lt_size - 1)
    return lt_size, lt_overlap


def _resize_to_target(images_bhwc01: torch.Tensor, upscale: int, device) -> torch.Tensor:
    # .contiguous() first: torch_npu warns when movedim() hands it a non-contiguous
    # view ("Cannot create tensor with interal format ...") for every single call.
    x = images_bhwc01.movedim(-1, 1).contiguous().to(device)  # BCHW [0, 1]
    _, _, h, w = x.shape
    target_h, target_w = h * upscale, w * upscale
    return F.interpolate(x, size=(target_h, target_w), mode="bicubic").clamp(0.0, 1.0)


def _run_untiled_batch(model, resized01: torch.Tensor, seed: int, vae_tile_size: int, vae_tile_overlap: int,
                       progress=None, same_seed: bool = False) -> torch.Tensor:
    b, _, h, w = resized01.shape
    padded01 = _pad_to_multiple(resized01, PAD_MULTIPLE)
    padded01 = _pad_to_square(padded01)
    padded_pm1 = padded01 * 2.0 - 1.0

    if progress:
        progress("encoding", 0.10)
    lq_latent, latents_mean, latents_std = model.encode_tiled(padded_pm1, vae_tile_size, vae_tile_overlap)
    if progress:
        progress("extracting vision features", 0.35)
    venc_fea = model.vision_features(padded01)

    if progress:
        progress("denoising (one step)", 0.55)
    if same_seed:
        noise = _generate_noise(lq_latent.shape[1:], seed, lq_latent.device, lq_latent.dtype)
        noise = noise.unsqueeze(0).expand(b, *noise.shape).contiguous()
    else:
        noise = _generate_noise_batch(lq_latent.shape[1:], seed, b, lq_latent.device, lq_latent.dtype)
    sr_latent = model.denoise_one_step(lq_latent, noise, venc_fea)

    if progress:
        progress("decoding", 0.80)
    decoded_pm1 = model.decode_tiled(sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)

    return decoded_pm1[:, :, :h, :w]


def _count_dit_tiles(h: int, w: int, tile_size: int, tile_overlap: int) -> int:
    """Tiles a run will submit to the DiT for one already-upscaled HxW image."""
    padded_h = h + (-h) % PAD_MULTIPLE
    padded_w = w + (-w) % PAD_MULTIPLE
    lh, lw = padded_h // AE_FACTOR, padded_w // AE_FACTOR
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)
    if lh <= lt_size and lw <= lt_size:
        return 1
    return len(_make_tile_grid(lh, lt_size, lt_overlap)) * len(_make_tile_grid(lw, lt_size, lt_overlap))


def _run_tiled_single(model, resized01: torch.Tensor, seed: int, tile_size: int, tile_overlap: int,
                      vae_tile_size: int, vae_tile_overlap: int, progress=None, base: float = 0.0,
                      span: float = 1.0) -> torch.Tensor:
    """Latent-space tiled DiT inference for one image (B=1). Ported from VOSR's tiled_latent_inference."""
    _, _, h, w = resized01.shape
    padded01 = _pad_to_multiple(resized01, PAD_MULTIPLE)
    padded_pm1 = padded01 * 2.0 - 1.0

    if progress:
        progress("encoding", base)
    lq_latent, latents_mean, latents_std = model.encode_tiled(padded_pm1, vae_tile_size, vae_tile_overlap)
    _, lc, lh, lw = lq_latent.shape
    lt_size, lt_overlap = _tile_params(tile_size, tile_overlap, lh, lw)

    total = 0
    if lh <= lt_size and lw <= lt_size:
        if progress:
            progress("denoising (one step)", base + 0.35 * span)
        lq_sq = _pad_to_square(lq_latent)
        venc_fea = model.vision_features(padded01)
        noise = _generate_noise(lq_sq.shape[1:], seed, lq_sq.device, lq_sq.dtype).unsqueeze(0)
        sr_latent = model.denoise_one_step(lq_sq, noise, venc_fea)[:, :, :lh, :lw]
    else:
        h_pos = _make_tile_grid(lh, lt_size, lt_overlap)
        w_pos = _make_tile_grid(lw, lt_size, lt_overlap)
        total = len(h_pos) * len(w_pos)
        g_weight = _gaussian_weights(lt_size, lt_size, lc, lq_latent.device)

        if progress:
            progress(f"vision features ({total} tiles)", base + 0.15 * span)
        tile_venc = {}
        for n, hi in enumerate(h_pos):
            for wi in w_pos:
                ph_s, pw_s = hi * AE_FACTOR, wi * AE_FACTOR
                ph_e = min((hi + lt_size) * AE_FACTOR, padded01.shape[2])
                pw_e = min((wi + lt_size) * AE_FACTOR, padded01.shape[3])
                tile_venc[(hi, wi)] = model.vision_features(padded01[:, :, ph_s:ph_e, pw_s:pw_e])

        noise = _generate_noise(lq_latent.shape[1:], seed, lq_latent.device, lq_latent.dtype).unsqueeze(0)
        z = noise

        # Explicit zeros, not zeros_like: on NPU, zeros_like of a bf16 tensor logs
        # "Cannot create tensor with interal format" on every call.
        u_acc = torch.zeros(lq_latent.shape, device=lq_latent.device, dtype=lq_latent.dtype)
        w_acc = torch.zeros(lq_latent.shape, device=lq_latent.device, dtype=lq_latent.dtype)
        done = 0
        for hi in h_pos:
            for wi in w_pos:
                he, we = hi + lt_size, wi + lt_size
                inp = torch.cat([lq_latent[:, :, hi:he, wi:we], z[:, :, hi:he, wi:we]], dim=1)
                u_tile = model.dit_velocity(inp, 1.0, 0.0, tile_venc[(hi, wi)])
                u_acc[:, :, hi:he, wi:we] += u_tile * g_weight
                w_acc[:, :, hi:he, wi:we] += g_weight
                done += 1
                if progress:
                    progress(f"denoising tile {done}/{total}", base + 0.35 * span + 0.45 * span * done / total)

        sr_latent = z - u_acc / w_acc

    if progress:
        progress("decoding", base + 0.85 * span)
    decoded_pm1 = model.decode_tiled(sr_latent, latents_mean, latents_std, vae_tile_size, vae_tile_overlap)
    return decoded_pm1[:, :, :h, :w]


def run_vosr2(
    model,
    images_bhwc01: torch.Tensor,
    upscale: int = 4,
    seed: int = 42,
    color_alignment: str = "wavelet",
    tile_size: int = 0,
    tile_overlap: int = 32,
    vae_tile_size: int = 0,
    vae_tile_overlap: int = 32,
    progress=None,
    same_seed: bool = False,
) -> torch.Tensor:
    """Upscale a BHWC RGB float batch in [0, 1]; returns BHWC float in [0, 1].

    ``progress`` is an optional ``callable(message: str, fraction: float)``.

    By default batch item *i* uses ``seed + i``, which makes a batch equivalent
    to running each image on its own. ``same_seed`` gives every item the same
    noise instead; video uses it so consecutive frames do not each draw
    different latent noise (which would flicker, since the noise is part of this
    one-step model's output).
    """
    if images_bhwc01.shape[-1] != 3:
        raise ValueError(f"VOSR 2.0 expects a 3-channel RGB image, got {images_bhwc01.shape[-1]} channels.")
    if upscale < 1:
        raise ValueError(f"upscale must be >= 1, got {upscale}.")
    if tile_size > 0 and tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap ({tile_overlap}) must be smaller than tile_size ({tile_size}).")
    if vae_tile_size > 0 and vae_tile_overlap >= vae_tile_size:
        raise ValueError(f"vae_tile_overlap ({vae_tile_overlap}) must be smaller than vae_tile_size ({vae_tile_size}).")

    device = model.device
    b, in_h, in_w = images_bhwc01.shape[0], images_bhwc01.shape[1], images_bhwc01.shape[2]
    h, w = in_h * upscale, in_w * upscale

    if tile_size == 0 and (h > 512 or w > 512):
        log.warning(
            "target size %dx%d exceeds VOSR 2.0's native 512 px training resolution with "
            "tile_size=0 (DiT tiling disabled); quality will likely degrade. Set "
            "tile_size=512 for large outputs.", w, h,
        )
        if max(h, w) > 2 * min(h, w):
            log.warning(
                "aspect ratio %d:%d is wider than 2:1; the untiled path pads the image to a "
                "square, so most of the DiT's work is spent on padding. Use tiling instead.",
                w, h,
            )
    if vae_tile_size == 0 and (h > 1024 or w > 1024):
        log.warning(
            "target size %dx%d with vae_tile_size=0 decodes the full image in one pass, a "
            "common source of out-of-memory errors. Set vae_tile_size=1024 for large outputs.",
            w, h,
        )

    if tile_size > 0:
        # Resize one item at a time: a video batch can be 8 frames of 4K output,
        # and materialising the whole upscaled batch on the device before the
        # first tile is computed is exactly the peak-memory blowup tiling exists
        # to avoid.
        aligned_items = []
        for i in range(b):
            item_progress = None
            if progress is not None:
                def item_progress(msg, frac, _i=i):
                    progress(f"[{_i + 1}/{b}] {msg}", (_i + frac) / b)
            item_src = images_bhwc01[i:i + 1]
            item_resized = _resize_to_target(item_src, upscale, device)
            item_pm1 = _run_tiled_single(
                model, item_resized, seed if same_seed else seed + i,
                tile_size, tile_overlap,
                vae_tile_size, vae_tile_overlap, progress=item_progress,
            ).cpu()
            item01 = (item_pm1.clamp(-1.0, 1.0) + 1.0) / 2.0
            aligned_items.append(apply_color_alignment(item01, item_resized.cpu(), color_alignment))
            del item_resized
        aligned01 = torch.cat(aligned_items, dim=0)
    else:
        resized01 = _resize_to_target(images_bhwc01, upscale, device)
        outputs_pm1 = _run_untiled_batch(
            model, resized01, seed, vae_tile_size, vae_tile_overlap,
            progress=progress, same_seed=same_seed,
        )
        decoded01 = (outputs_pm1.clamp(-1.0, 1.0) + 1.0) / 2.0
        if progress:
            progress("color alignment", 0.95)
        aligned01 = apply_color_alignment(decoded01, resized01, color_alignment)

    if progress:
        progress("done", 1.0)
    return aligned01.movedim(1, -1)
