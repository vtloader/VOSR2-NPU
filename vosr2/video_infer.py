"""Video super-resolution driver: stream frames through VOSR 2.0 and re-encode.

Design notes
------------
*Frames are streamed.* Decoding and encoding both run in ffmpeg subprocesses
piped to/from this process, so RAM is bounded by one chunk of frames rather
than the whole clip. A 4K/60fps minute is ~100 GB as a float tensor; it is
never materialised.

*The seed is shared across frames.* ``run_vosr2`` advances the seed per batch
item (``seed + i``), which is right for a set of independent stills but would
make consecutive frames draw *different* latent noise -- and in this one-step
model the noise is part of the output, so that reads as flicker. Video therefore
passes the same seed for every frame, which keeps the temporal signal stable.
The chunk size (``frame_batch``) is a pure throughput/memory knob and does not
change the result.

*Audio is streamed through untouched* when the whole clip is processed: the
source audio stream is remuxed (not re-encoded when the codec is mp4-friendly).
When only a slice is processed the audio is dropped, because a partial video
with full-length audio would be worse than no audio at all.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .inference import run_vosr2
from .video import VideoError, VideoInfo, VideoWriter, iter_frames, probe

log = logging.getLogger("vosr2.video_infer")


@dataclass
class VideoResult:
    path: Path
    frames: int
    width: int
    height: int
    seconds: float
    info: VideoInfo


def upscale_video(
    model,
    input_path: str | Path,
    output_path: str | Path,
    upscale: int = 2,
    seed: int = 42,
    color_alignment: str = "wavelet",
    tile_size: int = 512,
    tile_overlap: int = 64,
    vae_tile_size: int = 1024,
    vae_tile_overlap: int = 128,
    frame_batch: int = 4,
    start_frame: int = 0,
    max_frames: int | None = None,
    crf: int = 18,
    preset: str = "medium",
    keep_audio: bool = True,
    progress=None,
) -> VideoResult:
    """Upscale a video file, writing an H.264 mp4.

    ``frame_batch`` is how many frames are handed to the model at once -- the
    "同时处理帧数" knob in the UI. Larger batches are faster in the untiled path
    and cost proportionally more device memory; with DiT tiling enabled the
    model loops over items internally, so the batch only bounds memory.

    ``progress`` is an optional ``callable(message: str, fraction: float)``.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    frame_batch = max(1, int(frame_batch))
    if upscale < 1:
        raise ValueError(f"upscale must be >= 1, got {upscale}")

    info = probe(input_path)
    if info.width <= 0 or info.height <= 0:
        raise VideoError(f"Could not determine the frame size of {input_path}")

    total = info.frame_count_estimate
    start_frame = max(0, int(start_frame))
    if max_frames is not None and max_frames <= 0:
        max_frames = None
    if max_frames is not None and total:
        max_frames = min(max_frames, max(0, total - start_frame))
    expected = max_frames if max_frames is not None else max(0, total - start_frame)

    out_w, out_h = info.width * upscale, info.height * upscale
    full_clip = start_frame == 0 and (max_frames is None or (total and max_frames >= total))
    mux_audio = bool(keep_audio and info.has_audio and full_clip)
    if keep_audio and info.has_audio and not full_clip:
        log.info("processing a slice of the video: the audio track is not carried over")

    log.info(
        "video %s -> %dx%d, %s frames, batch %d",
        info.summary(), out_w, out_h, expected or "?", frame_batch,
    )

    t0 = time.time()
    done = 0
    writer = VideoWriter(
        output_path, out_w, out_h, info.fps,
        audio_source=input_path if mux_audio else None,
        audio_codec=info.audio_codec,
        crf=crf, preset=preset,
    )
    try:
        for chunk, index in iter_frames(input_path, info, start_frame, max_frames, frame_batch):
            def sub_progress(msg, frac, _done=done):
                if progress is not None and expected:
                    overall = min(1.0, (_done + frac * chunk.shape[0]) / expected)
                    progress(f"{msg} · 帧 {_done + 1}-{_done + chunk.shape[0]}/{expected}", overall)
                elif progress is not None:
                    progress(msg, None)

            # Same seed for every frame -- see the module docstring.
            out = run_vosr2(
                model, chunk,
                upscale=upscale,
                seed=seed,
                color_alignment=color_alignment,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                vae_tile_size=vae_tile_size,
                vae_tile_overlap=vae_tile_overlap,
                progress=sub_progress,
                same_seed=True,
            )
            writer.write(out)
            done += out.shape[0]
            if progress is not None and expected:
                progress(f"编码中 · 已处理 {done}/{expected} 帧", min(1.0, done / expected))
    except Exception:
        writer.__exit__(Exception, None, None)  # kill the encoder, leave no half file
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    else:
        writer.close()

    seconds = time.time() - t0
    log.info("wrote %s (%d frames) in %.1fs", output_path, done, seconds)
    if progress is not None:
        progress(f"完成 · {done} 帧", 1.0)
    return VideoResult(output_path, done, out_w, out_h, seconds, info)
