"""Command-line interface for standalone VOSR 2.0 super-resolution.

Examples
--------
Upscale one image 4x (tiling is mandatory above 512 px output)::

    python -m vosr2 upscale -i photo.jpg -o photo_4x.png -u 4 \
        --tile-size 512 --vae-tile-size 1024

Batch a whole folder::

    python -m vosr2 upscale -i inputs/ -o outputs/ --recursive

Upscale a video (frames streamed through the model)::

    python -m vosr2 video -i clip.mp4 -o clip_2x.mp4 -u 2 --frame-batch 6

Start the web UI::

    python -m vosr2 web --port 7860

Inspect the environment::

    python -m vosr2 info
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import device as dev
from . import imageio
from .loader import DEFAULT_MODEL_DIR, KNOWN_MODEL, VOSR2LoadError, list_model_bundles, load_vosr2
from .inference import run_vosr2

log = logging.getLogger("vosr2.cli")

_EXAMPLES = """\
examples:
  python -m vosr2 upscale -i photo.jpg -o photo_4x.png -u 4 --tile-size 512 --vae-tile-size 1024
  python -m vosr2 upscale -i inputs/ -o outputs/ -r
  python -m vosr2 video -i clip.mp4 -o clip_2x.mp4 -u 2 --frame-batch 6
  python -m vosr2 web --port 7860
  python -m vosr2 info
"""


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_common_load_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=KNOWN_MODEL, help=f"bundle folder under the model dir (default: {KNOWN_MODEL})")
    p.add_argument("--model-dir", default=None,
                   help=f"directory containing bundle folders (default: {DEFAULT_MODEL_DIR})")
    p.add_argument("--dtype", default="default", choices=["default", "fp16", "bf16", "fp32"],
                   help="compute dtype for DiT + vision encoder; the VAE always runs fp32 (default: auto)")
    p.add_argument("--device", default=None, help="device index, e.g. 0 or 1 (default: 0)")
    p.add_argument("--cpu-offload", action="store_true",
                   help="keep weights in host RAM and stream them in per phase (slow, low HBM)")
    p.add_argument("--no-download", action="store_true", help="never download weights; fail if files are missing")
    p.add_argument("-v", "--verbose", action="store_true")


def _add_inference_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-u", "--upscale", type=int, default=4, help="exact output multiplier (default: 4)")
    p.add_argument("--seed", type=int, default=42, help="latent noise seed (default: 42)")
    p.add_argument("--color-alignment", default="wavelet", choices=["wavelet", "adain", "none"],
                   help="post-process against the bicubic target (default: wavelet)")
    p.add_argument("--tile-size", type=int, default=512,
                   help="DiT pixel tile; 0 disables tiling (default: 512; required above 512 px output)")
    p.add_argument("--tile-overlap", type=int, default=64, help="DiT tile overlap in pixels (default: 64)")
    p.add_argument("--vae-tile-size", type=int, default=1024,
                   help="VAE pixel tile; 0 decodes in one pass (default: 1024)")
    p.add_argument("--vae-tile-overlap", type=int, default=128, help="VAE tile overlap in pixels (default: 128)")


def _apply_model_dir(args) -> None:
    if args.model_dir:
        import os
        os.environ["VOSR2_MODEL_DIR"] = str(Path(args.model_dir).expanduser().resolve())


def _progress_printer():
    last = [""]

    def progress(msg: str, frac: float | None = None):
        text = f"{msg}" + (f"  [{frac * 100:5.1f}%]" if frac is not None else "")
        if text != last[0]:
            print(f"\r  {text:<70}", end="", flush=True)
            last[0] = text
    return progress


def _finish_progress():
    print("\r" + " " * 72 + "\r", end="", flush=True)


def cmd_upscale(args) -> int:
    _apply_model_dir(args)
    inputs = imageio.collect_inputs(args.input, recursive=args.recursive)
    if not inputs:
        log.error("no images found in %s", args.input)
        return 1

    out = Path(args.output)
    multiple_inputs = len(inputs) > 1 or out.is_dir() or args.output.endswith(("/", "\\"))
    if multiple_inputs:
        out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    progress = _progress_printer() if not args.quiet else None
    model = load_vosr2(
        model_name=args.model,
        dtype=args.dtype,
        device_index=int(args.device or 0),
        cpu_offload=args.cpu_offload,
        allow_download=not args.no_download,
        progress=progress,
    )
    if progress:
        _finish_progress()
    log.info("loaded in %.1fs on %s", time.time() - t0, dev.describe())

    failures = 0
    for idx, src in enumerate(inputs, 1):
        dst = out / f"{src.stem}_x{args.upscale}.png" if multiple_inputs else out
        if dst.suffix.lower() not in imageio.SUPPORTED_SUFFIXES:
            dst = dst.with_suffix(".png")
        try:
            image = imageio.load_image(src)
            h, w = image.shape[1], image.shape[2]
            log.info("[%d/%d] %s (%dx%d) -> %dx%d", idx, len(inputs), src.name, w, h,
                     w * args.upscale, h * args.upscale)
            t1 = time.time()
            result = run_vosr2(
                model, image,
                upscale=args.upscale,
                seed=args.seed,
                color_alignment=args.color_alignment,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                vae_tile_size=args.vae_tile_size,
                vae_tile_overlap=args.vae_tile_overlap,
                progress=progress,
            )
            if progress:
                _finish_progress()
            imageio.save_image(result, dst, quality=args.quality)
            log.info("  wrote %s in %.1fs", dst, time.time() - t1)
        except Exception as exc:
            failures += 1
            if progress:
                _finish_progress()
            log.error("  failed on %s: %s", src, exc)
            if args.fail_fast:
                raise
    log.info("done: %d image(s), %d failure(s), %.1fs total", len(inputs), failures, time.time() - t0)
    return 1 if failures else 0


def _resolve_devices(args) -> list[int]:
    """Devices for a run: ``--devices`` wins, else ``--device``, else auto.

    ``--devices all`` expands to every visible device.
    """
    raw = getattr(args, "devices", None)
    if raw:
        spec = str(raw).strip().lower()
        if spec in ("all", "auto"):
            n = dev.device_count()
            if n <= 0:
                return [0]
            return list(range(n))
        out = []
        for part in spec.replace(",", " ").split():
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return sorted(dict.fromkeys(out)) or [0]
    return [int(args.device or 0)]


def cmd_upscale_video(args) -> int:
    _apply_model_dir(args)
    from .video import VideoError, probe
    from .parallel import upscale_video_parallel
    from .video_infer import upscale_video

    src = Path(args.input)
    if not src.is_file():
        log.error("video not found: %s", src)
        return 1
    try:
        info = probe(src)
    except VideoError as exc:
        log.error("%s", exc)
        return 1
    log.info("input  %s · %s · audio %s", src.name, info.summary(),
             info.audio_codec if info.has_audio else "none")

    out = Path(args.output)
    if out.is_dir() or args.output.endswith(("/", "\\")):
        out.mkdir(parents=True, exist_ok=True)
        out = out / f"{src.stem}_x{args.upscale}.mp4"
    elif out.suffix.lower() not in (".mp4", ".mkv", ".mov"):
        out = out.with_suffix(".mp4")

    devices = _resolve_devices(args)
    if len(devices) > 1 and args.cpu_offload:
        log.warning("--cpu-offload with multiple devices would need one host copy per card; "
                    "using a single device instead")
        devices = devices[:1]
    if len(devices) > 1:
        log.info("frame-parallel across %d devices: %s", len(devices), devices)

    t0 = time.time()
    progress = _progress_printer() if not args.quiet else None

    def make_model(device_index: int):
        return load_vosr2(
            model_name=args.model,
            dtype=args.dtype,
            device_index=device_index,
            cpu_offload=args.cpu_offload,
            allow_download=not args.no_download,
            progress=progress,
        )

    model = make_model(devices[0])
    if progress:
        _finish_progress()
    log.info("loaded in %.1fs on %s", time.time() - t0, dev.describe())

    try:
        if len(devices) == 1:
            result = upscale_video(
                model, src, out,
                upscale=args.upscale,
                seed=args.seed,
                color_alignment=args.color_alignment,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                vae_tile_size=args.vae_tile_size,
                vae_tile_overlap=args.vae_tile_overlap,
                frame_batch=args.frame_batch,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                crf=args.crf,
                preset=args.preset,
                keep_audio=not args.no_audio,
                progress=progress,
            )
        else:
            def factory(device_index: int):
                return model if device_index == devices[0] else make_model(device_index)

            result = upscale_video_parallel(
                factory, src, out, devices,
                upscale=args.upscale,
                seed=args.seed,
                color_alignment=args.color_alignment,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                vae_tile_size=args.vae_tile_size,
                vae_tile_overlap=args.vae_tile_overlap,
                frame_batch=args.frame_batch,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                crf=args.crf,
                preset=args.preset,
                keep_audio=not args.no_audio,
                progress=progress,
            )
    except VideoError as exc:
        if progress:
            _finish_progress()
        log.error("%s", exc)
        return 1
    if progress:
        _finish_progress()

    fps = result.frames / result.seconds if result.seconds > 0 else 0
    log.info("wrote %s · %d frames · %dx%d · %d device(s) · %.1fs (%.2f fps)",
             result.path, result.frames, result.width, result.height,
             getattr(result, "devices", 1), result.seconds, fps)
    return 0


def cmd_info(args) -> int:
    _apply_model_dir(args)
    print(f"torch        {__import__('torch').__version__}")
    try:
        import torch_npu
        print(f"torch_npu    {torch_npu.__version__}")
    except Exception:
        print("torch_npu    not installed")
    print(f"backend      {dev.backend()}")
    print(f"devices      {dev.describe()}")
    total = dev.memory_total()
    if total:
        print(f"memory       {total / 2**30:.1f} GiB per device")
    print(f"dtype        {dev.default_dtype()}")
    print(f"model dir    {DEFAULT_MODEL_DIR if not args.model_dir else Path(args.model_dir).expanduser().resolve()}")
    bundles = list_model_bundles()
    print(f"bundles      {', '.join(bundles) if bundles else '(none found)'}")
    return 0


def cmd_web(args) -> int:
    _apply_model_dir(args)
    from .webui import launch
    launch(
        model_dir=args.model_dir,
        model_name=args.model,
        dtype=args.dtype,
        device_index=int(args.device or 0),
        cpu_offload=args.cpu_offload,
        no_download=args.no_download,
        host=args.host,
        port=args.port,
        share=args.share,
        open_browser=not args.no_browser,
        defaults={
            "upscale": args.upscale,
            "seed": args.seed,
            "color_alignment": args.color_alignment,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "vae_tile_size": args.vae_tile_size,
            "vae_tile_overlap": args.vae_tile_overlap,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vosr2",
        description="VOSR 2.0 — one-step 1.4B image super-resolution (standalone).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXAMPLES,
    )
    sub = p.add_subparsers(dest="command", required=True)
    up = sub.add_parser("upscale", help="upscale one image, several images, or a folder")
    up.add_argument("-i", "--input", nargs="+", required=True, help="image file(s) or folder(s)")
    up.add_argument("-o", "--output", required=True, help="output file (single input) or folder")
    up.add_argument("-r", "--recursive", action="store_true", help="recurse into input folders")
    up.add_argument("--quality", type=int, default=95, help="JPEG/WebP quality 1-100 (default: 95)")
    up.add_argument("--fail-fast", action="store_true", help="stop at the first failed image")
    up.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    _add_common_load_args(up)
    _add_inference_args(up)
    up.set_defaults(func=cmd_upscale)

    vid = sub.add_parser("video", help="upscale a video (frames streamed through the model)")
    vid.add_argument("-i", "--input", required=True, help="input video file")
    vid.add_argument("-o", "--output", required=True, help="output .mp4 file or folder")
    vid.add_argument("--frame-batch", type=int, default=4,
                     help="frames processed at once per device (default: 4)")
    vid.add_argument("--devices", default=None,
                     help="devices for frame-parallel processing: 'all', '0,1', or '0-1' "
                          "(default: the single --device)")
    vid.add_argument("--start-frame", type=int, default=0, help="first frame to process (default: 0)")
    vid.add_argument("--max-frames", type=int, default=None,
                     help="process at most this many frames (default: all); audio is dropped for slices")
    vid.add_argument("--crf", type=int, default=18, help="H.264 quality, lower is better (default: 18)")
    vid.add_argument("--preset", default="medium",
                     choices=["ultrafast", "veryfast", "fast", "medium", "slow", "veryslow"],
                     help="x264 speed/size preset (default: medium)")
    vid.add_argument("--no-audio", action="store_true", help="do not carry the source audio track")
    vid.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    _add_common_load_args(vid)
    _add_inference_args(vid)
    vid.set_defaults(func=cmd_upscale_video)

    w = sub.add_parser("web", help="start the Gradio web interface")
    w.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    w.add_argument("--port", type=int, default=7860, help="port (default: 7860)")
    w.add_argument("--share", action="store_true", help="create a public Gradio link")
    w.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    _add_common_load_args(w)
    _add_inference_args(w)
    w.set_defaults(func=cmd_web)

    inf = sub.add_parser("info", help="show the detected device, dtype and model bundles")
    inf.add_argument("--model-dir", default=None)
    inf.set_defaults(func=cmd_info)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except VOSR2LoadError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
