# VOSR 2.0 — 独立超分辨率程序

VOSR 2.0（one-step 1.4B 图像超分辨率：LightningDiT + Qwen-Image 2D VAE + DINOv2-L）
的**独立运行版**，已从 ComfyUI 自定义节点改造而来，可直接在命令行或浏览器中使用，
原生支持 **华为昇腾 NPU（Ascend 910 / CANN / torch_npu）**，同时兼容 CUDA 与 CPU。

- 上游模型：https://github.com/cswry/VOSR ，权重 https://huggingface.co/CSWRY/VOSR
- 原 ComfyUI 节点包：https://github.com/ylchen333/ComfyUI-VOSR2

---

## 环境

本机环境（已配置完毕）：

| 项 | 版本 |
| --- | --- |
| Python | 3.12.16（`/usr/bin/python3.12`） |
| torch | 2.10.0+cpu（PyTorch CPU 源，**不可用 PyPI CUDA 版**） |
| torch_npu | 2.10.0 |
| CANN | 9.0.0（driver 26.1.1，Ascend910 ×2，各 61.3 GiB HBM） |
| 虚拟环境 | `.venv/`（本目录） |

> torch 必须用 CPU 源（如上海交大镜像 `https://mirror.sjtu.edu.cn/pytorch-wheels/cpu`）
> 安装：PyPI 上的 `torch==2.10.0` 是 CUDA 构建，与 `torch_npu` 同时加载会因
> allocator 配置键重复直接 abort（`ValueError: Duplicated key
> 'pinned_reserve_segment_size_mb'`）。安装 torch / torch_npu 时都要加 `--no-deps`。

## 安装（从零）

```bash
cd VOSR2
/usr/bin/python3.12 -m venv .venv
printf '[global]\nindex-url = https://mirrors.cloud.tencent.com/pypi/simple\ntimeout = 60\n' > .venv/pip.conf
source .venv/bin/activate

# NPU 栈（关键：CPU 源 + --no-deps）
pip install --no-deps --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu "torch==2.10.0+cpu"
pip install --no-deps "torch_npu==2.10.0"
# 其余依赖
pip install -r requirements.txt
```

CUDA 机器上把前两行换成 CUDA 版 torch 即可；纯 CPU 机器跳过 torch_npu。

## 模型权重

首次运行会自动从 `CSWRY/VOSR`（HuggingFace）下载缺失文件（共约 6.5 GB），
已下载的部分不会重复下载；`VOSR2_NO_DOWNLOAD=1` 或 `--no-download` 可禁网。

权重放在 `models/VOSR2/`（相对项目根，或用 `VOSR2_MODEL_DIR` 指定别处）：

```
models/VOSR2/
├── args.json
├── checkpoints/ema_model.safetensors          # 5.2 GB，DiT
├── Qwen-Image-vae-2d/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors    # Qwen 2D VAE
└── dinov2_vitl14.safetensors                  # DINOv2-L 视觉条件编码器
```

## 使用

本项目**不需要安装**，在项目目录内直接运行即可。先激活环境并 source CANN（NPU 必须）：

```bash
cd /workspace/VOSR2
source .venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

> 若不想激活虚拟环境，也可以始终用 `.venv/bin/python` 代替 `python`。
> `source .../set_env.sh` 只需在当前 shell 执行一次（新开终端要重新执行）。

### 命令行

```bash
# 单图 4x（输出超过 512px 必须开 DiT 分块；超过 ~1024px 建议再开 VAE 分块）
python -m vosr2 upscale -i photo.jpg -o photo_4x.png -u 4 \
    --tile-size 512 --tile-overlap 64 --vae-tile-size 1024 --vae-tile-overlap 128

# 整个文件夹（递归）
python -m vosr2 upscale -i inputs/ -o outputs/ -r -u 4

# 关闭分块（仅当输出 ≤512×512）
python -m vosr2 upscale -i small.png -o small_4x.png -u 4 --tile-size 0 --vae-tile-size 0

# 显存紧张时把权重留在内存、按阶段搬运
python -m vosr2 upscale -i big.tif -o big_4x.png --cpu-offload

# 指定第二张 NPU / fp16 / 禁止下载
python -m vosr2 upscale -i a.png -o a.png -u 4 --device 1 --dtype fp16 --no-download

# 查看设备、dtype、bundle 状态
python -m vosr2 info
```

### 视频

```bash
# 2x 视频超分，每卡每批 6 帧
python -m vosr2 video -i clip.mp4 -o clip_2x.mp4 -u 2 --frame-batch 6

# 双卡按帧分流（本机 2 张 910）
python -m vosr2 video -i clip.mp4 -o clip_2x.mp4 -u 2 --devices all

# 指定用哪几张卡
python -m vosr2 video -i clip.mp4 -o clip_2x.mp4 -u 2 --devices 0,1

# 先试看前 48 帧（只处理片段时音频不会带过去）
python -m vosr2 video -i clip.mp4 -o preview.mp4 -u 2 --max-frames 48

# 指定编码质量（CRF 越小越清晰、文件越大）
python -m vosr2 video -i clip.mp4 -o clip_2x.mp4 -u 2 --crf 16 --preset slow
```

视频参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--frame-batch` | 4 | **同时处理帧数**（每卡每批）。越大吞吐略高、显存占用略高 |
| `--devices` | 单卡 | `all` = 全部设备；`0,1` / `0-1` = 指定卡 |
| `--start-frame` | 0 | 从第几帧开始 |
| `--max-frames` | 全部 | 只处理前 N 帧；片段不保留音频 |
| `--crf` | 18 | H.264 质量，12–30 |
| `--preset` | medium | x264 速度/体积档位 |
| `--no-audio` | 关 | 不保留原音频 |

视频逐帧**流式**处理：解码和编码各走一个 ffmpeg 子进程管道，内存占用与视频长度无关。
所有帧共用同一份潜噪声（`same_seed`），否则每帧换种子会让画面闪烁；
因此改变 `--frame-batch` 只影响速度与显存，**不改变画质**（已用逐位相同的结果验证）。
音频在整段处理时按原样流复制（mp4 兼容编码）或转 AAC，不重新编码视频以外的内容。

### 多卡（按帧分流）

`--devices all` 会把帧序列切成连续几段，每张卡跑一份模型副本、负责其中一段。
不是张量并行，没有卡间通信——帧与帧本来就独立，这样最简单也最稳。

- 每张卡要放一份完整模型（bf16 约 4.8 GiB），显存不够时请减卡。
- 各段先写成**无损中间文件**（FFV1/Matroska），最后合并并只做一次 H.264 编码。
  这样多卡输出与单卡**逐位相同**；如果每段各自编码再拼接，后续段的 P 帧会从不同的
  参考帧预测，实测会产生最高 0.13/1.0 的画面漂移。
- 本机 2×Ascend910 实测端到端约 **1.4x**（1.12 → 1.54 fps）。纯算子并发能到 1.94x，
  但两块 910 在整条 VAE+DiT 混合管线下共享资源，1.4x 是这套硬件的实际上限。

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-u/--upscale` | 4 | 精确放大倍数（训练见过的退化最高 16x，>4x 质量不保证） |
| `--seed` | 42 | 潜噪声种子（批量时第 i 张用 seed+i） |
| `--color-alignment` | wavelet | `wavelet` / `adain` / `none` |
| `--tile-size` | 512 | DiT 像素分块，0 关闭；输出 >512px 必须开启 |
| `--vae-tile-size` | 1024 | VAE 像素分块，0 整图一次解码 |
| `--dtype` | 自动 | DiT/视觉编码器精度（`default`≈bf16），VAE 恒为 fp32 |
| `--device` | 0 | NPU/CUDA 卡号 |
| `--cpu-offload` | 关 | 权重驻留内存、按阶段搬上卡 |

### 网页界面

```bash
python -m vosr2 web --host 0.0.0.0 --port 7860
```

浏览器打开 `http://<机器IP>:7860`。界面分「图片」「视频」两个标签页：

- **图片**：上传图片 → 调参数 → 开始超分，结果以拖动滑块的前后对比展示，可下载 PNG。
- **视频**：上传视频 → 设置放大倍数、**同时处理帧数**、**使用设备数**、可选「只处理前 N 帧」→
  开始视频超分，结果可直接在页面预览并下载 MP4。

模型在启动时于后台加载，页面先可用、首次任务等待加载完成。

### 视频说明

- **同时处理帧数**（每卡每批，1–16）：每批送入模型的帧数。调大通常只带来很小的
  吞吐提升（DiT 分块本身就是顺序处理），但会略微增加显存；默认 4 是稳妥值。
- **使用设备数**：`1` = 单卡；`all` = 全部设备按帧分流；也可指定某一张卡。
  多卡时每张卡各加载一份模型（约 4.8 GiB/卡），首帧前会有一次加载等待。
- **只处理前 N 帧**：填 0 处理整段。设为 N 时只输出前 N 帧，且**不保留音频**，
  适合先快速试看效果。
- 视频帧共用同一份噪声，避免逐帧换种子导致的闪烁；因此改批大小不改变画质。
- 多卡输出与单卡逐位相同（分段用无损中间文件，最后统一编码一次）。
- 竖屏手机视频的旋转元数据会自动识别（90°/270° 交换宽高）。

### 设备选择

默认自动探测：有 `torch_npu` 且 NPU 可用 → **npu**；否则 CUDA；否则 CPU。
用环境变量强制指定：

```bash
VOSR2_DEVICE=npu python -m vosr2 upscale ...   # 强制 NPU（不可用时直接报错）
VOSR2_DEVICE=cpu python -m vosr2 upscale ...   # 纯 CPU
```

## 性能参考（Ascend 910，bf16）

| 任务 | 耗时 |
| --- | --- |
| 模型加载（三组件上卡） | ~12–15 s |
| 256×214 → 512×428（无分块） | ~0.55 s |
| 1280×1072 修复（DiT 512 分块，6 块） | ~2.2 s |
| 2688×1520 修复（DiT 512 分块） | ~3.2 s |
| 视频 410×232 → 820×464（每批 4 帧） | ~2.3 fps |
| 视频 1280×720 → 2560×1440（DiT 512 分块） | ~0.3 fps |
| 视频 640×360 → 1280×720，单卡 / 双卡 | 1.12 / **1.54 fps**（1.4x） |

显存：整包 bf16 常驻约 4.8 GiB；VAE 内部恒为 fp32。**2560×1440 不分块会直接 OOM**
（实测申请 28 GiB 失败），大图/大视频务必保持分块开启。

批大小对吞吐的影响（8 帧 410×232 → 820×464）：batch 1 / 2 / 4 / 8 分别为
2.21 / 2.63 / 2.51 / 2.58 fps，峰值显存都是 3.94 GiB。分块路径逐帧处理，
所以批大小主要受限于解码/编码流水，而不是模型并行度。

多卡按帧分流的加速比（2×Ascend910，640×360→1280×720，144 帧）：

| 方式 | 耗时 | 吞吐 |
| --- | --- | --- |
| 单卡 | 128.1 s | 1.12 fps |
| 双卡 | 93.4 s | 1.54 fps（**1.4x**） |

纯算子级并发（两个独立 matmul 流）能到 1.94x，但整条 VAE+DiT 混合管线下两块 910
共享资源，1.4x 是实测上限。两块卡之间是 SIO 互联（`npu-smi info -t topo`），
不是独立计算单元。

## 与 ComfyUI 版本的差异

- 移除 `comfy.ops` / `ModelPatcher` / `folder_paths` / `comfy_api` 依赖，
  模型三组件直接管理（支持 `--cpu-offload`）。
- `models/VOSR2/` 即 bundle 目录（原来在 `ComfyUI/models/vosr2/` 下）。
- 新增 CLI（图片 + 视频）、Gradio UI（双标签页）、EXIF 方向校正、透明图合成白底。
- 新增视频超分：ffmpeg 流式解码/编码、帧批处理、音频流复制、旋转元数据处理，
  帧间共享噪声避免闪烁。
- 新增多卡按帧分流（`--devices all`）：每卡一份模型副本，帧序列分段并行；
  分段用无损中间文件，合并后统一编码一次，多卡输出与单卡逐位相同。
- **修复**：非方形图无分块推理时按需把短边反射补到方形（CANN 的
  `ReflectionPad2d` 与 PyTorch 一样要求 pad 量小于对应边长，超过 2:1 长宽比时
  原实现会直接崩），等价周期反射索引回退，数学不变。
- **修复**：分块路径改为逐帧缩放，避免整批上采样结果同时驻留显存。
- 保留：一步去噪推理契约、DiT/VAE 双重分块与高斯混合、seed 批处理语义、
  wavelet/adain 色彩对齐、checkpoint 严格校验、首用自动下载。

## 目录结构

```
VOSR2/
├── vosr2/            # 独立程序包
│   ├── cli.py        #   命令行入口（python -m vosr2 upscale|video|web|info）
│   ├── webui.py      #   Gradio 界面（图片 / 视频）
│   ├── loader.py     #   bundle 发现/下载/严格加载
│   ├── inference.py  #   推理管线（含分块）
│   ├── video.py      #   ffmpeg 流式视频读写
│   ├── video_infer.py#   视频超分驱动（帧批处理、音频）
│   ├── parallel.py   #   多卡按帧分流（分段 + 无损合并）
│   ├── tiled_vae.py  #   VAE 分块编解码
│   ├── color.py      #   色彩对齐
│   ├── device.py     #   NPU/CUDA/CPU 设备层
│   ├── ops.py        #   no-init Linear/Conv2d/LayerNorm（替代 comfy.ops）
│   ├── imageio.py    #   图像读写
│   └── models/       #   LightningDiT / Qwen VAE / DINOv2 架构
├── models/VOSR2/     # 权重 bundle（首次运行自动下载）
├── VOSR_cases/       # 官方示例对照图（lr.png / vosr_2_0.png）
├── requirements.txt
└── pyproject.toml
```

视频功能依赖系统的 `ffmpeg`/`ffprobe`（本机已装 ffmpeg 4.4.2 + libx264），
不需要 PyAV 或 OpenCV。

## 自检

`VOSR_cases/` 下有 5 组官方对照图，每组是同一场景的输入 `lr.png` 与官方输出
`vosr_2_0.png`（同尺寸，属于 1x 修复）。用它们可以快速确认环境与权重是否正常：

```bash
# 单张验证：PSNR 应明显高于直接 bicubic
python -m vosr2 upscale -i VOSR_cases/landscape/lr.png -o /tmp/landscape_fix.png -u 1
```

在 Ascend 910 上的参考结果：

| 用例 | VOSR 2.0 输出 vs 官方 | bicubic vs 官方 |
| --- | --- | --- |
| tiny_text_english | 28.6 dB | 19.4 dB |
| landscape | 35.5 dB | 29.3 dB |

若 PSNR 明显低于上表，通常是 dtype/设备不匹配或权重文件不完整。

## 许可

代码遵循 Apache-2.0（见 `LICENSE`）。模型权重单独从 `CSWRY/VOSR` 下载，
含 DINOv2（Meta）与 Qwen-Image VAE 各自的条款，再分发/商用前请自行审阅。
