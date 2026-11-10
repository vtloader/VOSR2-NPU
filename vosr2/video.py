"""Streaming video I/O for VOSR 2.0, built on the system ``ffmpeg``/``ffprobe``.

Deliberately no PyAV/OpenCV dependency: ffmpeg is already required to have any
video support at all, and piping raw RGB frames keeps the whole path to a
single subprocess per direction with no intermediate files on disk.

Frames are decoded lazily and written as they come out, so a two-hour clip uses
the same RAM as a two-second one -- only the current chunk of frames is ever
resident.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("vosr2.video")

# Audio codecs that can be stream-copied into an mp4 container without re-encoding.
_MP4_AUDIO_COPYABLE = {"aac", "mp3", "ac3", "eac3", "alac", "opus", "flac"}


class VideoError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or a video cannot be read/written."""


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise VideoError(
            f"`{name}` was not found on PATH. Install ffmpeg (e.g. `apt-get install -y ffmpeg`) "
            f"to use video input/output."
        )
    return exe


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    nb_frames: int          # 0 when the container does not report it
    has_audio: bool
    audio_codec: str | None
    rotation: int
    codec: str

    @property
    def frame_count_estimate(self) -> int:
        if self.nb_frames > 0:
            return self.nb_frames
        if self.duration > 0 and self.fps > 0:
            return int(round(self.duration * self.fps))
        return 0

    def summary(self) -> str:
        n = self.nb_frames if self.nb_frames else f"~{self.frame_count_estimate}"
        return f"{self.width}×{self.height} @ {self.fps:.3g} fps · {n} 帧 · {self.duration:.1f}s"


def probe(path: str | Path) -> VideoInfo:
    """Read stream metadata with ffprobe."""
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"Video not found: {path}")
    cmd = [
        _tool("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"ffprobe failed on {path}: {proc.stderr.strip()[:400]}")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoError(f"No video stream found in {path}")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # Rotation: phone clips are stored landscape with a 90° display matrix, and
    # ffmpeg autorotates on decode by default -- so the frames we receive are the
    # *display* orientation, and width/height must be swapped for 90/270.
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                rotation = int(float(side["rotation"]))
            except (TypeError, ValueError):
                pass
    if rotation == 0:
        try:
            rotation = int(float(video.get("tags", {}).get("rotate", 0)))
        except (TypeError, ValueError):
            rotation = 0

    w, h = int(video.get("width", 0)), int(video.get("height", 0))
    if abs(rotation) % 180 == 90:
        w, h = h, w

    fps = 0.0
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = video.get(key)
        if raw and raw not in ("0/0", "0"):
            try:
                fps = float(Fraction(raw))
                break
            except (ValueError, ZeroDivisionError):
                continue
    if fps <= 0:
        fps = 25.0

    duration = 0.0
    for raw in (video.get("duration"), data.get("format", {}).get("duration")):
        try:
            duration = float(raw)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    nb = 0
    for raw in (video.get("nb_frames"),):
        try:
            nb = int(raw)
        except (TypeError, ValueError):
            nb = 0
    if nb <= 0 and duration > 0:
        nb = int(round(duration * fps))

    return VideoInfo(
        width=w, height=h, fps=fps, duration=duration, nb_frames=nb,
        has_audio=audio is not None,
        audio_codec=(audio or {}).get("codec_name"),
        rotation=rotation,
        codec=video.get("codec_name", "?"),
    )


def iter_frames(path: str | Path, info: VideoInfo, start_frame: int = 0,
                max_frames: int | None = None, frame_batch: int = 1):
    """Yield ``(chunk, global_index)`` where chunk is ``(B, H, W, 3)`` float32 in [0, 1].

    Decoding is a single ffmpeg pipe; frames are grouped into chunks of
    ``frame_batch`` so the caller can hand a real batch to the model.
    """
    w, h = info.width, info.height
    frame_bytes = w * h * 3
    if frame_bytes <= 0:
        raise VideoError(f"Invalid video dimensions for {path}: {w}x{h}")

    cmd = [_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start_frame > 0 and info.fps > 0:
        cmd += ["-ss", f"{start_frame / info.fps:.6f}"]
    cmd += ["-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 2)
    buf = bytearray()
    sent = 0
    index = start_frame
    closed_early = False
    try:
        while True:
            if max_frames is not None and sent >= max_frames:
                closed_early = True  # consumer asked for a slice; ffmpeg still has frames to write
                break
            need = frame_bytes - len(buf)
            data = proc.stdout.read(need)
            if not data:
                break
            buf.extend(data)
            if len(buf) < frame_bytes:
                continue
            arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
            buf.clear()
            chunk = [torch.from_numpy(arr)]
            index_start = index
            index += 1
            sent += 1

            # Greedily top up this chunk from the pipe without blocking on a
            # partial frame: read whatever is already buffered.
            while len(chunk) < frame_batch and (max_frames is None or sent < max_frames):
                data = proc.stdout.read(frame_bytes)
                if not data or len(data) < frame_bytes:
                    if data:
                        buf.extend(data)
                    break
                arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
                chunk.append(torch.from_numpy(arr))
                index += 1
                sent += 1

            yield torch.stack(chunk, dim=0), index_start
    except GeneratorExit:
        closed_early = True
        raise
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        proc.stderr.close()
        # A "Broken pipe" here just means we stopped reading early (a slice, or
        # the caller broke out of the loop) -- not a decode failure.
        if not closed_early and proc.returncode not in (0, None, -15) and err:
            raise VideoError(f"ffmpeg decoding failed: {err[:500]}")


class VideoWriter:
    """Encode frames to H.264 in a subprocess, optionally muxing the source audio."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float,
                 audio_source: str | Path | None = None, audio_codec: str | None = None,
                 crf: int = 18, preset: str = "medium", codec: str = "libx264",
                 pixel_format: str = "yuv420p"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(width), int(height)
        if self.width <= 0 or self.height <= 0:
            raise VideoError(f"Invalid output size {self.width}x{self.height}")
        self.fps = float(fps) if fps and fps > 0 else 25.0

        cmd = [
            _tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:.6f}",
            "-i", "-",
        ]
        has_audio = audio_source is not None
        if has_audio:
            cmd += ["-i", str(audio_source)]
        # yuv420p needs even dimensions; pad by at most one pixel rather than
        # silently cropping the result. FFV1 keeps RGB at 8 bits per channel
        # (bgr0), so lossless segments survive the round trip untouched.
        if pixel_format == "bgr0":
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=bgr0"]
        else:
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        cmd += ["-map", "0:v:0"]
        if has_audio:
            cmd += ["-map", "1:a:0?"]
            if audio_codec in _MP4_AUDIO_COPYABLE:
                cmd += ["-c:a", "copy"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
        if codec == "ffv1":
            cmd += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", pixel_format]
        else:
            cmd += ["-c:v", codec, "-crf", str(int(crf)), "-preset", preset, "-pix_fmt", pixel_format]
        cmd += ["-movflags", "+faststart", str(self.path)]
        log.debug("ffmpeg encode: %s", " ".join(cmd))
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise VideoError(f"Could not start ffmpeg for {self.path}: {exc}") from exc
        self.frames_written = 0
        self._closed = False

    def write(self, frame_bhwc01: torch.Tensor) -> None:
        """Append one frame or a batch of frames (``(H,W,3)`` / ``(B,H,W,3)``) in [0, 1]."""
        x = frame_bhwc01
        if x.dim() == 3:
            x = x.unsqueeze(0)
        arr = (x.detach().float().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
        try:
            self.proc.stdin.write(arr.tobytes())
        except BrokenPipeError:
            self._raise_with_stderr("ffmpeg stopped accepting frames")
        self.frames_written += x.shape[0]

    def _raise_with_stderr(self, prefix: str):
        err = b""
        try:
            if self.proc.stderr:
                err = self.proc.stderr.read() or b""
        except Exception:
            pass
        raise VideoError(f"{prefix}: {err.decode('utf-8', 'replace').strip()[:500]}")

    def close(self) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        err = self.proc.stderr.read() if self.proc.stderr else b""
        code = self.proc.wait()
        if self.proc.stderr:
            self.proc.stderr.close()
        if code != 0:
            raise VideoError(
                f"ffmpeg encoding failed (exit {code}): "
                f"{err.decode('utf-8', 'replace').strip()[:500]}"
            )
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            return False
        self.close()
        return False
