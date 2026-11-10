"""Gradio web UI for standalone VOSR 2.0.

Launched with ``python -m vosr2 web`` (or ``vosr2 web`` once pip-installed).
The model loads in a background thread at startup so the page is immediately
reachable; the first run simply waits for loading to finish.
"""
from __future__ import annotations

import contextlib
import datetime
import logging
import threading
import time
import traceback
from pathlib import Path

import gradio as gr

from . import device as dev
from . import imageio
from .inference import run_vosr2
from .loader import KNOWN_MODEL, load_vosr2, model_options

log = logging.getLogger("vosr2.webui")

_OUTPUT_DIR = Path("outputs")


class _State:
    """Shared model handle; Gradio may call handlers from several threads."""

    def __init__(self):
        self.model = None
        self.lock = threading.Lock()
        self.error: str | None = None
        self.meta = ""

    def set(self, model, meta: str):
        self.model = model
        self.meta = meta
        self.error = None


def _fmt_bytes(n: int) -> str:
    return f"{n / 2**30:.2f} GiB"


def _load_async(state: _State, model_name: str, dtype: str, device_index: int,
                cpu_offload: bool, allow_download: bool):
    """Load in a daemon thread; the UI reports status via ``state``."""
    def work():
        try:
            t = time.time()
            model = load_vosr2(
                model_name=model_name,
                dtype=dtype,
                device_index=device_index,
                cpu_offload=cpu_offload,
                allow_download=allow_download,
            )
            mem = dev.memory_reserved(model.device)
            state.set(
                model,
                f"**{dev.describe()}** · dtype `{dtype}` · {model_name} · "
                f"加载用时 {time.time() - t:.1f}s · 显存 {_fmt_bytes(mem)}",
            )
            log.info("model ready on %s", model.device)
        except Exception as exc:  # surface it in the UI instead of dying silently
            state.error = f"{type(exc).__name__}: {exc}"
            log.error("model load failed:\n%s", traceback.format_exc())

    th = threading.Thread(target=work, daemon=True)
    th.start()
    return th


def build_ui(state: _State, defaults: dict, load_kwargs: dict):
    """Assemble the Blocks app. Handlers close over ``state`` and ``load_kwargs``."""

    def do_load(model_name, dtype, device_index):
        with state.lock:
            state.model = None
        th = _load_async(state, model_name, dtype, int(device_index),
                         load_kwargs.get("cpu_offload", False),
                         not load_kwargs.get("no_download", False))
        th.join(timeout=1.0)  # let the status text update, don't block the UI
        if state.error:
            raise gr.Error(f"模型加载失败：{state.error}")
        if state.model is not None:
            return state.meta
        return f"⏳ 正在后台加载模型（{model_name}）……页面可用后自动就绪。"

    class _ProgressBridge:
        """Translate progress callbacks into ``gr.Progress`` updates safely.

        Two traps this exists to avoid:

        * ``gr.Progress`` resolves its callback through contextvars that only
          exist on the Gradio event thread. Called from a worker thread it is a
          silent no-op, so a multi-device run would show 0% until it finished.
        * A ``None`` fraction means "indeterminate". Passing 0.0 for it (as an
          earlier version did) drags the bar back to zero mid-run, which reads
          as the job restarting.

        It also never lets the reported fraction go backwards: several stages
        (per-frame decode/encode, per-tile denoise) each count from zero, and
        without this the bar oscillates instead of advancing.
        """

        def __init__(self, gr_progress):
            self.gr = gr_progress
            self.lock = threading.Lock()
            self.frac = 0.0
            self.msg = ""

        def update(self, msg: str, frac: float | None = None, flush: bool = True):
            """Record progress; optionally push it to Gradio immediately.

            ``flush=False`` is for callbacks that may run on a worker thread,
            where ``gr.Progress`` would silently do nothing -- the owning thread
            is expected to call :meth:`pump` instead.
            """
            with self.lock:
                if frac is not None:
                    self.frac = max(self.frac, min(1.0, float(frac)))
                self.msg = msg
                cur_frac, cur_msg = self.frac, self.msg
            if flush:
                self.gr(cur_frac, desc=cur_msg)

        def worker_update(self, msg: str, frac: float | None = None):
            """Thread-safe record-only variant for worker threads."""
            self.update(msg, frac, flush=False)

        def pump(self):
            """Flush the latest state to Gradio from the thread that owns it."""
            with self.lock:
                cur_frac, cur_msg = self.frac, self.msg
            if cur_msg:
                self.gr(cur_frac, desc=cur_msg)

    def _wait_for_model(bridge):
        if state.error:
            raise gr.Error(f"模型加载失败：{state.error}")
        if state.model is None:
            bridge.update("等待模型加载……")
            while state.model is None and state.error is None:
                time.sleep(0.25)
                bridge.pump()
            if state.error:
                raise gr.Error(f"模型加载失败：{state.error}")
        return state.model

    @contextlib.contextmanager
    def _acquire(bridge, waiting_msg: str):
        """Take the model lock, keeping the UI alive while another job holds it.

        A bare ``with state.lock`` blocks the Gradio event thread with no
        feedback; the progress bar just freezes. Polling lets us say why.
        """
        if not state.lock.acquire(blocking=False):
            bridge.update(waiting_msg)
            bridge.pump()
            while not state.lock.acquire(timeout=0.25):
                bridge.pump()
        try:
            yield
        finally:
            state.lock.release()

    def _save_output(out, suffix, ext="png") -> Path:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = _OUTPUT_DIR / f"vosr2_{stamp}_{suffix}.{ext}"
        imageio.save_image(out, path)
        return path

    def run_upscale(image, upscale, seed, color_alignment, tile_size, tile_overlap,
                    vae_tile_size, vae_tile_overlap, progress=gr.Progress(track_tqdm=False)):
        if image is None:
            raise gr.Error("请先上传一张图片。")
        bridge = _ProgressBridge(progress)
        model = _wait_for_model(bridge)

        img = imageio.from_numpy(image)
        h, w = img.shape[1], img.shape[2]

        with _acquire(bridge, "等待上一个任务结束……"):
            out = run_vosr2(
                model, img,
                upscale=int(upscale),
                seed=int(seed),
                color_alignment=color_alignment,
                tile_size=int(tile_size),
                tile_overlap=int(tile_overlap),
                vae_tile_size=int(vae_tile_size),
                vae_tile_overlap=int(vae_tile_overlap),
                progress=bridge.update,
            )

        path = _save_output(out, f"x{int(upscale)}")
        before = imageio.to_pil(img[0])
        after = imageio.to_pil(out[0])
        mem = dev.memory_reserved(model.device)
        note = (
            f"输入 {w}×{h} → 输出 {out.shape[2]}×{out.shape[1]} · seed {seed} · "
            f"{color_alignment} · 显存 {_fmt_bytes(mem)}"
        )
        return (before, after), str(path), note

    def run_video(video_path, upscale, seed, color_alignment, tile_size, tile_overlap,
                  vae_tile_size, vae_tile_overlap, frame_batch, max_frames, keep_audio,
                  device_spec, crf, progress=gr.Progress(track_tqdm=False)):
        if not video_path:
            raise gr.Error("请先上传一个视频。")
        from .video import VideoError, probe
        from .video_infer import upscale_video
        from .parallel import upscale_video_parallel
        from .cli import _resolve_devices

        bridge = _ProgressBridge(progress)
        try:
            info = probe(video_path)
        except VideoError as exc:
            raise gr.Error(str(exc))

        model = _wait_for_model(bridge)
        limit = int(max_frames) if max_frames and int(max_frames) > 0 else None

        class _DevArgs:
            devices = device_spec
            device = load_kwargs.get("device_index", 0)
        devices = _resolve_devices(_DevArgs())
        if len(devices) > 1 and load_kwargs.get("cpu_offload"):
            devices = devices[:1]

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = _OUTPUT_DIR / f"vosr2_{stamp}_video_x{int(upscale)}.mp4"
        try:
            with _acquire(bridge, "等待上一个任务结束……"):
                if len(devices) == 1:
                    result = upscale_video(
                        model, video_path, path,
                        upscale=int(upscale),
                        seed=int(seed),
                        color_alignment=color_alignment,
                        tile_size=int(tile_size),
                        tile_overlap=int(tile_overlap),
                        vae_tile_size=int(vae_tile_size),
                        vae_tile_overlap=int(vae_tile_overlap),
                        frame_batch=int(frame_batch),
                        max_frames=limit,
                        keep_audio=bool(keep_audio),
                        crf=int(crf),
                        progress=bridge.update,
                    )
                else:
                    extra = {}

                    def factory(device_index):
                        if device_index == devices[0]:
                            return model
                        if device_index not in extra:
                            bridge.update(f"在设备 {device_index} 上加载模型副本……")
                            bridge.pump()
                            extra[device_index] = load_vosr2(
                                model_name=load_kwargs.get("model_name", KNOWN_MODEL),
                                dtype=load_kwargs.get("dtype", "default"),
                                device_index=device_index,
                                cpu_offload=False,
                                allow_download=not load_kwargs.get("no_download", False),
                            )
                        return extra[device_index]

                    result = upscale_video_parallel(
                        factory, video_path, path, devices,
                        upscale=int(upscale),
                        seed=int(seed),
                        color_alignment=color_alignment,
                        tile_size=int(tile_size),
                        tile_overlap=int(tile_overlap),
                        vae_tile_size=int(vae_tile_size),
                        vae_tile_overlap=int(vae_tile_overlap),
                        frame_batch=int(frame_batch),
                        max_frames=limit,
                        keep_audio=bool(keep_audio),
                        crf=int(crf),
                        progress=bridge.worker_update,
                        on_tick=bridge.pump,
                    )
        except VideoError as exc:
            raise gr.Error(str(exc))

        fps = result.frames / result.seconds if result.seconds > 0 else 0.0
        n_dev = getattr(result, "devices", 1)
        note = (
            f"输入 {info.width}×{info.height} @ {info.fps:.3g} fps → "
            f"输出 {result.width}×{result.height} · {result.frames} 帧 · "
            f"{n_dev} 卡 · 处理 {result.seconds:.1f}s ({fps:.2f} fps) · 每卡每批 {int(frame_batch)} 帧"
        )
        return str(path), str(path), note

    d = defaults

    def inference_controls(prefix: str, upscale_max: int = 16, upscale_default: int | None = None):
        """Shared upscale/seed/colour/tiling controls, prefixed for uniqueness."""
        up = gr.Slider(1, upscale_max,
                       value=upscale_default if upscale_default is not None else d.get("upscale", 4),
                       step=1, label="放大倍数（>4x 质量不保证）")
        seed = gr.Number(value=d.get("seed", 42), precision=0, label="seed")
        colour = gr.Radio(["wavelet", "adain", "none"],
                          value=d.get("color_alignment", "wavelet"), label="色彩对齐")
        with gr.Accordion("分块（大图/大视频必需）", open=False):
            ts = gr.Slider(0, 4096, value=d.get("tile_size", 512), step=64,
                           label="DiT 分块大小（0=关闭；输出 >512 px 必须开启）")
            to = gr.Slider(0, 512, value=d.get("tile_overlap", 64), step=8, label="DiT 分块重叠")
            vts = gr.Slider(0, 8192, value=d.get("vae_tile_size", 1024), step=64,
                            label="VAE 分块大小（0=整图一次解码）")
            vto = gr.Slider(0, 512, value=d.get("vae_tile_overlap", 128), step=8, label="VAE 分块重叠")
        return up, seed, colour, ts, to, vts, vto

    with gr.Blocks(title="VOSR 2.0 超分辨率") as demo:
        gr.Markdown(
            f"# VOSR 2.0 图像 / 视频超分辨率\n"
            f"一步 (one-step) 1.4B DiT · LightningDiT + Qwen-Image 2D VAE + DINOv2-L · "
            f"当前设备：**{dev.describe()}**\n\n"
            f"> 输出超过 512 px 时 DiT 分块（tile）是必需的；超过 ~1024 px 建议同时开启 VAE 分块，"
            f"否则显存占用会快速失控。"
        )
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Accordion("模型", open=False):
                    model_dd = gr.Dropdown(model_options(), value=load_kwargs.get("model_name", KNOWN_MODEL),
                                           label="模型 bundle")
                    dtype_radio = gr.Radio(["default", "fp16", "bf16", "fp32"],
                                           value=load_kwargs.get("dtype", "default"),
                                           label="DiT / 视觉编码器精度（VAE 始终 fp32）")
                    dev_num = gr.Number(value=load_kwargs.get("device_index", 0),
                                        precision=0, label="设备序号")
                    load_btn = gr.Button("重新加载模型", variant="secondary")
            with gr.Column(scale=7):
                status_md = gr.Markdown(value=state.meta or "⏳ 模型正在后台加载……")

        with gr.Tabs():
            with gr.Tab("图片"):
                with gr.Row():
                    with gr.Column(scale=5):
                        input_image = gr.Image(type="numpy", label="输入图像", height=380)
                        (img_up, img_seed, img_colour, img_ts, img_to,
                         img_vts, img_vto) = inference_controls("img")
                        img_run = gr.Button("开始超分", variant="primary")
                    with gr.Column(scale=7):
                        img_slider = gr.ImageSlider(label="对比（拖动滑块：左输入 / 右结果）", height=520)
                        img_file = gr.File(label="下载结果（PNG）")
                        img_note = gr.Markdown()

            with gr.Tab("视频"):
                with gr.Row():
                    with gr.Column(scale=5):
                        input_video = gr.Video(label="输入视频", height=380, sources=["upload"])
                        (vid_up, vid_seed, vid_colour, vid_ts, vid_to,
                         vid_vts, vid_vto) = inference_controls("vid", upscale_max=4, upscale_default=2)
                        with gr.Accordion("视频参数", open=True):
                            vid_batch = gr.Slider(1, 16, value=d.get("frame_batch", 4), step=1,
                                                  label="同时处理帧数（每卡每批：越大越快，显存占用越高）")
                            n_dev = max(1, dev.device_count())
                            vid_devices = gr.Dropdown(
                                ["1"] + (["all"] if n_dev > 1 else []) +
                                [str(i) for i in range(n_dev)],
                                value="1",
                                label=f"使用设备数（共 {n_dev} 张可用；多卡按帧分流）",
                                info="all = 全部设备；数字 = 指定单张卡。多卡会各加载一份模型",
                            )
                            vid_max = gr.Number(value=0, precision=0,
                                                label="只处理前 N 帧（0=整个视频；设为 N 会丢弃音频）")
                            vid_audio = gr.Checkbox(value=True, label="保留原音频（整段处理时）")
                            vid_crf = gr.Slider(12, 30, value=18, step=1,
                                                label="H.264 质量 CRF（越小越清晰、文件越大）")
                        vid_run = gr.Button("开始视频超分", variant="primary")
                    with gr.Column(scale=7):
                        vid_out = gr.Video(label="结果视频", height=420)
                        vid_file = gr.File(label="下载结果（MP4）")
                        vid_note = gr.Markdown()
                        gr.Markdown(
                            "> 视频逐帧流式处理，内存占用与视频长度无关。帧与帧共用同一份噪声，"
                            "避免逐步换种子造成的闪烁；改变「同时处理帧数」只影响速度与显存，"
                            "不改变画质。"
                        )

        img_run.click(
            run_upscale,
            inputs=[input_image, img_up, img_seed, img_colour,
                    img_ts, img_to, img_vts, img_vto],
            outputs=[img_slider, img_file, img_note],
            concurrency_limit=1,
        )
        vid_run.click(
            run_video,
            inputs=[input_video, vid_up, vid_seed, vid_colour, vid_ts, vid_to,
                    vid_vts, vid_vto, vid_batch, vid_max, vid_audio, vid_devices, vid_crf],
            outputs=[vid_out, vid_file, vid_note],
            concurrency_limit=1,
        )
        load_btn.click(
            do_load,
            inputs=[model_dd, dtype_radio, dev_num],
            outputs=[status_md],
            concurrency_limit=1,
        )
    return demo


def launch(model_dir=None, model_name: str = KNOWN_MODEL, dtype: str = "default",
           device_index: int = 0, cpu_offload: bool = False, no_download: bool = False,
           host: str = "0.0.0.0", port: int = 7860, share: bool = False,
           open_browser: bool = True, defaults: dict | None = None) -> None:
    import os
    if model_dir:
        os.environ["VOSR2_MODEL_DIR"] = str(Path(model_dir).expanduser().resolve())

    state = _State()
    load_kwargs = dict(model_name=model_name, dtype=dtype, device_index=device_index,
                       cpu_offload=cpu_offload, no_download=no_download)
    _load_async(state, model_name, dtype, device_index, cpu_offload, not no_download)

    demo = build_ui(state, defaults or {}, load_kwargs)
    demo.queue(max_size=16)
    demo.launch(server_name=host, server_port=port, share=share, inbrowser=open_browser,
                show_error=True, favicon_path=None)
