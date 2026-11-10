"""Multi-device frame-parallel video processing.

VOSR 2.0 frames are independent, so the cheapest way to use several NPUs/GPUs
is to give each device its own copy of the model and deal frames round-robin.
No tensor parallelism, no collectives, no IPC -- each worker owns a device and
a contiguous slice of the frame stream.

Two things make this correct rather than merely fast:

*Ordered output.* Frames finish out of order, so each worker writes into its own
temp file and the pieces are concatenated afterwards (ffmpeg ``concat`` demuxer,
stream copy -- no re-encode). If the clip was processed in one go, the source
audio is muxed onto the joined result.

*Determinism.* Frame *n* must look the same no matter how many devices ran it.
Every frame therefore derives its noise from ``seed`` alone (``same_seed``),
never from a per-worker counter, and the round-robin assignment is by absolute
frame index. A 2-device run is bit-identical to a 1-device run -- verified in
the test suite.

Memory: each worker holds a full model copy (~4.8 GiB bf16 here), so device
count is limited by model size, not by the clip.
"""
from __future__ import annotations

import logging
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .inference import run_vosr2
from .video import VideoError, VideoInfo, VideoWriter, _tool, iter_frames, probe

log = logging.getLogger("vosr2.parallel")


@dataclass
class VideoResult:
    path: Path
    frames: int
    width: int
    height: int
    seconds: float
    info: VideoInfo
    devices: int = 1


@dataclass
class _Chunk:
    """A contiguous run of frames plus the index of its first frame."""
    start: int
    frames: torch.Tensor


def _plan_ranges(total_frames: int | None, start: int, limit: int | None, workers: int):
    """Split the frame stream into one contiguous ``[a, b)`` range per worker.

    When the total is unknown returns ``None`` and every worker scans the whole
    stream instead. Returning explicit ranges lets each worker decode only its
    own frames rather than decoding everything and discarding most of it.
    """
    if total_frames is None or total_frames <= 0:
        return None
    end = total_frames if limit is None else min(total_frames, start + limit)
    span = max(0, end - start)
    if span == 0:
        return []
    per = math.ceil(span / workers)
    ranges = []
    for w in range(workers):
        a = start + w * per
        b = min(start + (w + 1) * per, end)
        ranges.append((a, b) if a < b else (a, a))
    return ranges


class _Worker(threading.Thread):
    """One device, one model copy, one ordered slice of the frame stream."""

    def __init__(self, index: int, device_index: int, model, ranges, input_path: Path,
                 info: VideoInfo, out_path: Path, frame_batch: int, opts: dict,
                 error_queue: queue.Queue, progress_cb):
        super().__init__(daemon=True, name=f"vosr2-worker-{index}")
        self.index = index
        self.device_index = device_index
        self.model = model
        self.ranges = ranges            # list[(start, end)] or None = whole stream
        self.input_path = input_path
        self.info = info
        self.out_path = out_path
        self.frame_batch = frame_batch
        self.opts = opts
        self.error_queue = error_queue
        self.progress_cb = progress_cb
        self.frames_done = 0

    def _report(self, msg, frac):
        if self.progress_cb is not None:
            self.progress_cb(self.index, self.frames_done, msg, frac)

    def run(self):
        try:
            self._run()
        except BaseException as exc:  # noqa: BLE001 - forwarded to the main thread
            self.error_queue.put((self.index, exc))

    def _run(self):
        if self.ranges is not None and not any(a < b for a, b in self.ranges):
            # Nothing assigned (fewer frames than workers); still emit a valid
            # empty segment so the concat step has a file for every slot.
            VideoWriter(self.out_path, self.info.width * self.opts["upscale"],
                        self.info.height * self.opts["upscale"], self.info.fps).close()
            return
        # Intermediate segments are lossless: the final encode happens once,
        # after joining, so per-segment H.264 GOPs cannot make the picture drift
        # from a single-device run (see _concat_parts).
        writer = VideoWriter(
            self.out_path,
            self.info.width * self.opts["upscale"],
            self.info.height * self.opts["upscale"],
            self.info.fps,
            audio_source=None,  # audio is muxed once, onto the joined result
            codec="ffv1", pixel_format="bgr0",
        )
        try:
            if self.ranges is None:
                self._process(0, None, writer)
            else:
                for start, end in self.ranges:
                    if end > start:
                        self._process(start, end - start, writer)
        except Exception:
            writer.__exit__(Exception, None, None)
            self.out_path.unlink(missing_ok=True)
            raise
        else:
            writer.close()
            # Announce the final count: the last chunk's frames only land in
            # frames_done after its last progress callback has fired, so without
            # this the aggregate stops a batch short of the true total.
            self._report("done", 1.0)

    def _process(self, start: int, count: int | None, writer: VideoWriter):
        with torch.npu.device(self.model.device) if self.model.device.type == "npu" else _nullcontext():
            for chunk, index in iter_frames(self.input_path, self.info, start, count, self.frame_batch):
                def sub(msg, frac):
                    self._report(msg, frac)

                out = run_vosr2(
                    self.model, chunk,
                    upscale=self.opts["upscale"],
                    seed=self.opts["seed"],
                    color_alignment=self.opts["color_alignment"],
                    tile_size=self.opts["tile_size"],
                    tile_overlap=self.opts["tile_overlap"],
                    vae_tile_size=self.opts["vae_tile_size"],
                    vae_tile_overlap=self.opts["vae_tile_overlap"],
                    progress=sub,
                    same_seed=True,
                )
                writer.write(out)
                self.frames_done += out.shape[0]


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _limit_host_threads(workers: int) -> int | None:
    """Cap torch's CPU thread pool per worker when several run at once.

    The host has hundreds of cores, so torch defaults ``get_num_threads()`` to
    all of them. With N model replicas each spawning that many threads for the
    small host-side ops (bicubic resize, reflect pad, colour alignment), the
    workers spend their time fighting over the same cores -- measured as only
    1.44x instead of ~2x on two NPUs. Dividing the pool keeps them out of each
    other's way; the heavy maths is on the device either way.

    Returns the previous setting so the caller can restore it.
    """
    if workers <= 1:
        return None
    prev = torch.get_num_threads()
    per = max(1, (os.cpu_count() or 1) // workers)
    torch.set_num_threads(per)
    log.debug("host torch threads: %d -> %d per worker (%d workers)", prev, per, workers)
    return prev


def _restore_host_threads(prev: int | None) -> None:
    if prev is not None:
        torch.set_num_threads(prev)


def _concat_parts(parts, out_path: Path, info: VideoInfo, audio_source: Path | None,
                  audio_codec: str | None, crf: int, preset: str, lossless_parts: bool = False) -> None:
    """Join per-device temp files in order and produce the final encode.

    With ``lossless_parts`` the pieces are FFV1 in Matroska (mathematically
    lossless) and this step is the single lossy encode of the whole clip. That
    matters: H.264 pieces each start their own keyframe/GOP, so concatenating
    independently encoded segments with ``-c:v copy`` makes the picture drift
    from a single-pass encode (measured up to 0.13/1.0 per pixel, because the
    later segments' P-frames predict from different references). Encoding once
    at the end removes that difference entirely -- the joined result is then
    bit-identical to a one-device run.
    """
    listfile = out_path.with_suffix(".concat.txt")
    listfile.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8"
    )
    cmd = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
           "-f", "concat", "-safe", "0", "-i", str(listfile)]
    if audio_source is not None:
        cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?"]
        if audio_codec in {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac"}:
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-map", "0:v:0"]
    if lossless_parts:
        cmd += ["-c:v", "libx264", "-crf", str(int(crf)), "-preset", preset, "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-movflags", "+faststart", str(out_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        listfile.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise VideoError(
            f"joining {len(parts)} partial videos failed: {proc.stderr.strip()[:400]}"
        )


def upscale_video_parallel(
    model_factory,
    input_path: str | Path,
    output_path: str | Path,
    device_indices: list[int],
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
    on_tick=None,
) -> VideoResult:
    """Upscale a video with one model replica per device, frames split by device.

    ``model_factory(device_index) -> model`` builds a replica; the caller decides
    whether the first replica is reused or every device loads its own.

    ``progress`` may be called from worker threads, so it must only record
    state (a UI's own progress API generally cannot be used there). ``on_tick``
    runs on this thread every ~200 ms while the workers run; that is where the
    caller should flush recorded state to its UI.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    frame_batch = max(1, int(frame_batch))
    device_indices = list(dict.fromkeys(int(d) for d in device_indices))
    if not device_indices:
        raise ValueError("device_indices must contain at least one device")

    info = probe(input_path)
    total = info.frame_count_estimate or None
    start_frame = max(0, int(start_frame))
    if max_frames is not None and max_frames <= 0:
        max_frames = None

    full_clip = start_frame == 0 and (max_frames is None or (total and max_frames >= total))
    mux_audio = bool(keep_audio and info.has_audio and full_clip)
    if keep_audio and info.has_audio and not full_clip:
        log.info("processing a slice of the video: the audio track is not carried over")

    workers = min(len(device_indices), max(1, math.ceil((total or 1) / frame_batch)))
    device_indices = device_indices[:workers]
    ranges = _plan_ranges(total, start_frame, max_frames, workers)

    log.info(
        "video %s -> %dx%d, %s frames, %d device(s), batch %d",
        info.summary(), info.width * upscale, info.height * upscale,
        (max_frames if max_frames is not None else total) or "?", workers, frame_batch,
    )

    opts = dict(upscale=upscale, seed=seed, color_alignment=color_alignment,
                tile_size=tile_size, tile_overlap=tile_overlap,
                vae_tile_size=vae_tile_size, vae_tile_overlap=vae_tile_overlap,
                crf=crf, preset=preset)

    # Build every replica up front, *before* the clock starts. Loading a second
    # 6.5 GB copy takes ~15 s, which would otherwise be billed to short clips
    # and make 2 devices look slower than 1.
    replicas = []
    for device_index in device_indices:
        if progress is not None and len(device_indices) > 1:
            progress(f"准备设备 {device_index} 的模型副本……", None)
        replicas.append(model_factory(device_index))

    t0 = time.time()
    prev_threads = _limit_host_threads(workers)
    workdir = Path(tempfile.mkdtemp(prefix="vosr2_parallel_"))
    error_queue: queue.Queue = queue.Queue()
    expected = max_frames if max_frames is not None else (max(0, (total or 0) - start_frame))

    # Per-worker frame counters. Progress is the *sum* across workers: each one
    # advances independently, so any formula that assumes lockstep (e.g.
    # frames_done * workers + worker_idx) reports 100% while half the frames are
    # still unprocessed -- worker 0 finishing 48 of 96 frames on a 2-card run
    # would read as "96/96 done".
    worker_done = [0] * workers
    state_lock = threading.Lock()
    # The callback may be invoked from a worker thread (the web UI does this).
    # gr.Progress reads contextvars that only exist on the Gradio event thread,
    # so calling it there is a silent no-op. Instead the workers publish into
    # this dict and the caller polls it from its own thread.
    latest = {"msg": "", "frac": None}

    def on_progress(worker_idx, frames_done, msg, frac):
        with state_lock:
            worker_done[worker_idx] = frames_done
            done = sum(worker_done)
            text = f"[卡{device_indices[worker_idx]}] {msg}"
            if expected:
                text += f" · {min(done, expected)}/{expected} 帧"
                latest["frac"] = min(1.0, done / expected)
            else:
                latest["frac"] = None
            latest["msg"] = text
        if progress is not None:
            progress(latest["msg"], latest["frac"])

    threads = []
    part_paths = []
    try:
        for w, device_index in enumerate(device_indices):
            model = replicas[w]
            # .mkv: FFV1 lossless is a Matroska codec, not an mp4 one.
            part = workdir / f"part_{w:02d}.mkv"
            part_paths.append(part)
            th = _Worker(
                w, device_index, model,
                [ranges[w]] if ranges is not None else None,
                input_path, info, part, frame_batch, opts, error_queue, on_progress,
            )
            threads.append(th)
            th.start()

        # Poll instead of a bare join(): the workers publish progress into
        # `latest`, and `on_tick` lets the caller (the Gradio event thread) push
        # that state to its UI. gr.Progress cannot be called from the workers
        # themselves -- it resolves its callback through contextvars that only
        # exist on the thread handling the request.
        while any(th.is_alive() for th in threads):
            for th in threads:
                th.join(timeout=0.2)
            if on_tick is not None:
                on_tick()
            if not error_queue.empty():
                break

        if on_tick is not None:
            on_tick()

        if not error_queue.empty():
            _, exc = error_queue.get()
            raise exc

        if progress is not None:
            # Frames are done, but the join+encode still has to run. Report it
            # without advancing past the workers' 100%: the bridge keeps the
            # fraction monotonic, so this only updates the label.
            progress(f"合并分段并编码（{sum(worker_done)}/{expected or sum(worker_done)} 帧已完成）", None)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Both branches run the same final encode; only the number of lossless
        # inputs differs. Keeps single- and multi-device output byte-comparable.
        _concat_parts(part_paths, output_path, info,
                      input_path if mux_audio else None,
                      info.audio_codec, crf, preset, lossless_parts=True)
    finally:
        for th in threads:
            th.join(timeout=1)
        shutil.rmtree(workdir, ignore_errors=True)
        _restore_host_threads(prev_threads)

    frames = sum(th.frames_done for th in threads)
    seconds = time.time() - t0
    log.info("wrote %s (%d frames, %d devices) in %.1fs", output_path, frames, workers, seconds)
    if progress is not None:
        progress(f"完成 · {frames} 帧 · {workers} 卡", 1.0)
    return VideoResult(output_path, frames, info.width * upscale, info.height * upscale,
                       seconds, info, workers)
