"""Discovery, download and strict loading of a VOSR 2.0 bundle.

A bundle is a self-contained folder (by default ``models/VOSR2``):

    <bundle>/
        args.json
        checkpoints/ema_model.safetensors       (or clean_weights/, or bundle root)
        Qwen-Image-vae-2d/{config.json, diffusion_pytorch_model.safetensors}
        dinov2_vitl14.safetensors

VOSR 2.0 is a fixed DiT/VAE/DINOv2 triple -- the DiT runs entirely in *this*
Qwen 2D VAE's latent space (its channel count, scale factor and per-channel
latents_mean/std) and is conditioned on *this* DINOv2-L feature extractor, so
none of the three is independently swappable.

Missing parts are fetched from the pinned ``CSWRY/VOSR`` repo on first use;
once present nothing touches the network. Unlike the ComfyUI build, the three
components are *not* wrapped in ModelPatcher: there is no queue of other models
to make room for, so each module simply lives on the compute device (or in host
RAM when ``--cpu-offload`` is requested).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import device as dev
from .models.dinov2 import build_dinov2_vitl14
from .models.lightningdit import LightningDiT
from .models.qwenimage_vae2d import AutoencoderKLQwenImage2D

log = logging.getLogger("vosr2.loader")

# Repo root is one level above this package; weights live in <root>/models by
# default, matching the layout this project shipped with.
PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent
DEFAULT_MODEL_DIR = Path(os.environ.get("VOSR2_MODEL_DIR", PROJECT_ROOT / "models"))

HF_REPO_ID = "CSWRY/VOSR"
KNOWN_MODEL = "VOSR2"                    # the one confirmed bundle name
_VAE_SUBDIR = "Qwen-Image-vae-2d"
_VAE_FILES = ("config.json", "diffusion_pytorch_model.safetensors")
_VISION_FILENAME = "dinov2_vitl14.safetensors"

_DIT_HF_FILES = ("VOSR2/args.json", "VOSR2/checkpoints/ema_model.safetensors")
_VAE_HF_FILES = tuple(f"{_VAE_SUBDIR}/{name}" for name in _VAE_FILES)
_DINOV2_HF_FILE = "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth"

REQUIRED_ARGS = {
    "ae_type": "qwen",
    "dim": 1536,
    "depth": 36,
    "num_heads": 24,
    "patch_size": 2,
    "enc_type": "dinov2l",
    "enc_dim": 1024,
    "layer_dinov2b_list": [17],
    "auxiliary_time_cond": False,
    "distill_type": "onestep",
}

DIT_ARG_KEYS = ("mlp_ratio", "use_qknorm", "use_swiglu", "use_rope", "use_rmsnorm", "encdim_ratio", "resolution")

_TRAINING_ONLY_KEY_PATTERNS = (
    re.compile(r"^n_averaged$"),
    re.compile(r"^step_count$"),
    re.compile(r"^decay$"),
)
_STRIPPABLE_PREFIXES = ("module.", "_orig_mod.", "ema_model.")


class VOSR2LoadError(RuntimeError):
    """Raised for any missing file, incompatible config, or malformed state dict."""


def model_dir() -> Path:
    """Root directory holding bundle folders."""
    return Path(os.environ.get("VOSR2_MODEL_DIR", DEFAULT_MODEL_DIR))


def bundle_dir(model_name: str = KNOWN_MODEL) -> Path:
    """Resolve a bundle name against the model root, rejecting path escapes."""
    root = model_dir()
    if not model_name or "/" in model_name or "\\" in model_name or model_name in (".", ".."):
        raise VOSR2LoadError(f"Invalid VOSR2 selection: {model_name!r}")
    candidate = (root / model_name).resolve()
    if root.resolve() not in candidate.parents and candidate != root.resolve():
        raise VOSR2LoadError(f"Invalid VOSR2 selection: {model_name!r}")
    return candidate


def list_model_bundles() -> list[str]:
    root = model_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "args.json").is_file())


def model_options() -> list[str]:
    """Bundle options for the UI: what is on disk, always offering the known name."""
    found = list_model_bundles()
    return found if KNOWN_MODEL in found else [KNOWN_MODEL, *found]


def bundle_complete(model_name: str = KNOWN_MODEL) -> bool:
    """True when every file of the bundle is present (no network needed)."""
    b = bundle_dir(model_name)
    vae = b / _VAE_SUBDIR
    return (
        (b / "args.json").is_file()
        and _find_dit_weight(b) is not None
        and all((vae / n).is_file() for n in _VAE_FILES)
        and (b / _VISION_FILENAME).is_file()
    )


def _convert_dinov2_pth_to_safetensors(src_pth: Path, dest: Path) -> None:
    state = torch.load(str(src_pth), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise VOSR2LoadError(
            f"Unexpected DINOv2 checkpoint format at {src_pth}: expected a flat state dict."
        )
    tensors = {k: v.contiguous() for k, v in state.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise VOSR2LoadError(f"DINOv2 checkpoint at {src_pth} contained no tensors.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    save_file(tensors, str(tmp))
    tmp.replace(dest)


def ensure_vosr2_files(model_name: str = KNOWN_MODEL, allow_download: bool = True) -> None:
    """Fetch missing parts of the confirmed bundle from ``CSWRY/VOSR``.

    A no-op once the files are present -- it never contacts the network then.
    Only ``KNOWN_MODEL`` can be auto-fetched; a custom bundle name that is absent
    is left for the normal "not found" error in :func:`load_vosr2`.
    """
    if model_name != KNOWN_MODEL:
        return

    root = model_dir()
    bundle = root / KNOWN_MODEL
    vae_dir = bundle / _VAE_SUBDIR

    dit_missing = _find_dit_weight(bundle) is None or not (bundle / "args.json").is_file()
    vae_missing = not all((vae_dir / name).is_file() for name in _VAE_FILES)
    vision_missing = not (bundle / _VISION_FILENAME).is_file()

    if not (dit_missing or vae_missing or vision_missing):
        return
    if not allow_download:
        raise VOSR2LoadError(
            f"VOSR2 bundle at {bundle} is incomplete and downloads are disabled "
            f"(VOSR2_NO_DOWNLOAD=1). Place the missing files manually -- see the "
            f"README 'Model files' section."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise VOSR2LoadError(
            "VOSR 2.0 model files are missing and huggingface_hub is not available "
            "to download them. Install huggingface_hub, or place the files manually "
            "-- see the README 'Model files' section."
        ) from exc

    if dit_missing:
        log.info("downloading DiT bundle from %s ...", HF_REPO_ID)
        for f in _DIT_HF_FILES:
            hf_hub_download(HF_REPO_ID, f, local_dir=str(root))
    if vae_missing:
        log.info("downloading Qwen-Image 2D VAE from %s ...", HF_REPO_ID)
        vae_dir.mkdir(parents=True, exist_ok=True)
        for f in _VAE_HF_FILES:
            hf_hub_download(HF_REPO_ID, f, local_dir=str(bundle))
    if vision_missing:
        log.info("downloading + converting DINOv2-L encoder from %s ...", HF_REPO_ID)
        src = hf_hub_download(HF_REPO_ID, _DINOV2_HF_FILE)
        _convert_dinov2_pth_to_safetensors(Path(src), bundle / _VISION_FILENAME)


def _load_args_json(b: Path) -> dict:
    args_path = b / "args.json"
    if not args_path.is_file():
        raise VOSR2LoadError(f"VOSR2 model bundle at {b} is missing args.json.")
    with open(args_path, "r") as f:
        args = json.load(f)

    for key, expected in REQUIRED_ARGS.items():
        actual = args.get(key)
        if actual != expected:
            raise VOSR2LoadError(
                f"VOSR2 model bundle at {b} has an incompatible config: "
                f"expected {key}={expected!r}, got {actual!r}. This build supports only "
                f"the VOSR 2.0 one-step 1.4B checkpoint."
            )
    missing = [k for k in DIT_ARG_KEYS if k not in args]
    if missing:
        raise VOSR2LoadError(
            f"VOSR2 model bundle at {b}'s args.json is missing required field(s): {missing}."
        )
    return args


def _find_dit_weight(b: Path):
    for candidate in (
        b / "clean_weights" / "ema_model.safetensors",
        b / "checkpoints" / "ema_model.safetensors",
        b / "ema_model.safetensors",
    ):
        if candidate.is_file():
            return candidate
    return None


def _resolve_dit_weight_path(b: Path) -> Path:
    weight = _find_dit_weight(b)
    if weight is None:
        raise VOSR2LoadError(
            f"No ema_model.safetensors found under {b} "
            f"(looked in clean_weights/, checkpoints/, and the bundle root)."
        )
    return weight


def _strip_key(key: str) -> str | None:
    stripped = key
    for prefix in _STRIPPABLE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if any(pattern.match(stripped) for pattern in _TRAINING_ONLY_KEY_PATTERNS):
        return None
    return stripped


def _load_state_dict_lean(module: torch.nn.Module, path: Path, dtype: torch.dtype, source: Path) -> None:
    """Stream a safetensors checkpoint into ``module`` one tensor at a time.

    Reading + casting one tensor at a time off the safetensors mmap (rather than
    an eager ``load_file()``) means the checkpoint is never resident a second
    time in its original dtype, so peak host RAM is roughly the model at its
    final dtype instead of the model plus a full extra copy.
    """
    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            stripped = _strip_key(key)
            if stripped is None:
                continue
            sd[stripped] = f.get_tensor(key).to(dtype)

    missing, unexpected = module.load_state_dict(sd, strict=False, assign=True)
    if missing or unexpected:
        raise VOSR2LoadError(
            f"VOSR2 checkpoint at {source} does not match the expected {type(module).__name__} "
            f"architecture (missing={missing}, unexpected={unexpected})."
        )
    del sd
    gc.collect()


class VOSR2Model:
    """Runtime bundle: DiT + VAE + vision encoder + config, ready for inference.

    Exposes narrow operations (``vision_features``, ``encode_tiled``,
    ``denoise_one_step``, ``decode_tiled``) rather than raw modules, so callers
    never need to probe for capabilities on the wrapped components.
    """

    def __init__(self, dit, vae, vision_encoder, args: dict, device: torch.device, dtype: torch.dtype,
                 cpu_offload: bool = False):
        self.dit = dit
        self.vae = vae
        self.vision_encoder = vision_encoder
        self.args = args
        self.device = device
        self.dtype = dtype
        self.vision_layer_index = args["layer_dinov2b_list"][0]
        self.dinov2_size = args.get("dinov2_size", 448)
        self.cpu_offload = cpu_offload
        # Per-module residency, so a phase that only needs the DiT does not drag
        # the VAE and vision encoder back to the device as well. Everything is
        # built on the host, so nothing is resident until to_device() runs.
        self._resident = {id(dit): False, id(vae): False, id(vision_encoder): False}

    # -- lifecycle ---------------------------------------------------------
    def to_cpu(self) -> None:
        """Move all weights to host RAM; the next inference phase brings them back."""
        for m in (self.dit, self.vae, self.vision_encoder):
            if self._resident[id(m)]:
                m.to("cpu")
                self._resident[id(m)] = False
        dev.empty_cache()

    def to_device(self) -> None:
        for m in (self.dit, self.vae, self.vision_encoder):
            if not self._resident[id(m)]:
                m.to(self.device)
                self._resident[id(m)] = True

    def _model(self, m):
        if not self._resident[id(m)]:
            m.to(self.device)
            self._resident[id(m)] = True
        return m

    # -- inference operations ---------------------------------------------
    def vision_features(self, lq_bchw_01: torch.Tensor) -> list:
        """lq_bchw_01: (B, 3, H, W) in [0, 1]. Returns a one-item list (fixed layer)."""
        model = self._model(self.vision_encoder)
        x = torch.nn.functional.interpolate(
            lq_bchw_01.to(self.device), size=(self.dinov2_size, self.dinov2_size), mode="bicubic"
        ).clamp(0.0, 1.0)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        feats = model.forward_intermediate_layer(x.to(model.pos_embed.dtype), self.vision_layer_index)
        return [feats]

    def encode(self, x_bchw_neg1_1: torch.Tensor):
        from . import tiled_vae
        model = self._model(self.vae)
        return tiled_vae.encode_latent(model, x_bchw_neg1_1.to(self.device))

    def encode_tiled(self, x_bchw_neg1_1: torch.Tensor, tile_size: int, tile_overlap: int):
        from . import tiled_vae
        model = self._model(self.vae)
        return tiled_vae.encode_dispatch(model, x_bchw_neg1_1.to(self.device), tile_size, tile_overlap)

    def decode(self, latent: torch.Tensor, latents_mean, latents_std) -> torch.Tensor:
        from . import tiled_vae
        model = self._model(self.vae)
        return tiled_vae.decode_latent(model, latent.to(self.device), latents_mean, latents_std)

    def decode_tiled(self, latent: torch.Tensor, latents_mean, latents_std, tile_size: int, tile_overlap: int) -> torch.Tensor:
        from . import tiled_vae
        model = self._model(self.vae)
        return tiled_vae.decode_dispatch(model, latent.to(self.device), latents_mean, latents_std, tile_size, tile_overlap)

    def dit_velocity(self, inp: torch.Tensor, t_cur: float, t_next: float, venc_fea: list) -> torch.Tensor:
        """Single DiT forward: predicted flow velocity for ``inp = cat([lq_latent, z], dim=1)``."""
        model = self._model(self.dit)
        compute_dtype = model.t_embedder.mlp[0].weight.dtype
        inp = inp.to(device=self.device, dtype=compute_dtype)
        venc_fea = [f.to(device=self.device, dtype=compute_dtype) for f in venc_fea]
        b = inp.shape[0]
        t_cur_t = torch.full((b,), t_cur, device=self.device, dtype=compute_dtype)
        t_next_t = torch.full((b,), t_next, device=self.device, dtype=compute_dtype)
        return model.forward_flexible(inp, t_cur_t, t_next_t, venc_fea)

    def denoise_one_step(self, lq_latent: torch.Tensor, noise: torch.Tensor, venc_fea: list) -> torch.Tensor:
        """One Euler flow-matching update from t=1 to t=0 (VOSR 2.0 is a one-step model)."""
        z = noise.to(self.device)
        u = self.dit_velocity(torch.cat([lq_latent.to(self.device), z], dim=1), 1.0, 0.0, venc_fea)
        return z - u


def load_vosr2(
    model_name: str = KNOWN_MODEL,
    dtype: str = "default",
    device_index: int = 0,
    cpu_offload: bool = False,
    allow_download: bool = True,
    progress=None,
) -> VOSR2Model:
    """Load a VOSR 2.0 bundle and return it ready for inference.

    ``progress`` is an optional ``callable(message: str, fraction: float | None)``
    used by the CLI and web UI to report loading stages.
    """
    def report(msg, frac=None):
        log.info("%s", msg)
        if progress is not None:
            progress(msg, frac)

    report("checking model files", 0.0)
    ensure_vosr2_files(model_name, allow_download=allow_download)

    b = bundle_dir(model_name)
    if not b.is_dir():
        raise VOSR2LoadError(f"VOSR2 model bundle not found: {model_name!r} (looked in {b})")
    args = _load_args_json(b)
    dit_weight_path = _resolve_dit_weight_path(b)

    vae_dir = b / _VAE_SUBDIR
    if not (vae_dir / "config.json").is_file():
        raise VOSR2LoadError(
            f"VOSR2 bundle {b} is missing its Qwen-Image 2D VAE ({vae_dir}/config.json). "
            f"The VAE's latent space is tied to this DiT and cannot be substituted."
        )
    vision_path = b / _VISION_FILENAME
    if not vision_path.is_file():
        raise VOSR2LoadError(f"VOSR2 bundle {b} is missing its DINOv2-L encoder ({vision_path}).")

    device = dev.set_device(device_index)
    compute_dtype = dev.resolve_dtype(dtype)
    report(f"loading on {device} as {str(compute_dtype).replace('torch.', '')}", 0.05)

    # A second replica for frame-parallel video is built while the first one
    # already sits on another device; scoping the device context keeps any
    # allocation-triggered context on the right card.
    base_channels = 16  # Qwen 2D VAE latent channels
    report("building LightningDiT (1.4B)", 0.10)
    dit = LightningDiT(
        input_size=args["resolution"] // 8,
        patch_size=args["patch_size"],
        in_channels=2 * base_channels,
        out_channels=base_channels,
        hidden_size=args["dim"],
        depth=args["depth"],
        num_heads=args["num_heads"],
        mlp_ratio=args["mlp_ratio"],
        z_dims=args["enc_dim"],
        encdim_ratio=args["encdim_ratio"],
        auxiliary_time_cond=args["auxiliary_time_cond"],
        use_qknorm=args["use_qknorm"],
        use_swiglu=args["use_swiglu"],
        use_rope=args["use_rope"],
        use_rmsnorm=args["use_rmsnorm"],
        num_fused_layers=len(args["layer_dinov2b_list"]),
    )
    report("loading DiT weights (5.2 GB)", 0.20)
    _load_state_dict_lean(dit, dit_weight_path, compute_dtype, dit_weight_path)
    dit = dit.eval()
    for p in dit.parameters():
        p.requires_grad_(False)

    report("loading Qwen-Image 2D VAE", 0.60)
    vae = AutoencoderKLQwenImage2D.from_pretrained(str(vae_dir))
    vae = vae.eval().float()  # the Qwen VAE always runs in fp32
    for p in vae.parameters():
        p.requires_grad_(False)

    report("loading DINOv2-L encoder", 0.75)
    vision_encoder = build_dinov2_vitl14()
    _load_state_dict_lean(vision_encoder, vision_path, compute_dtype, vision_path)
    vision_encoder = vision_encoder.eval()
    for p in vision_encoder.parameters():
        p.requires_grad_(False)

    model = VOSR2Model(dit, vae, vision_encoder, args, device, compute_dtype, cpu_offload=cpu_offload)

    if cpu_offload:
        report("keeping weights in host RAM (cpu offload)", 0.9)
    else:
        report("moving weights to device", 0.9)
        model.to_device()

    gc.collect()
    dev.empty_cache()
    report("model ready", 1.0)
    return model
