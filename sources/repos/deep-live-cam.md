# Deep-Live-Cam — hacksider/Deep-Live-Cam

> 官方代码/README 原文归档；归纳仅覆盖实际读到的主分支实现。

- **标题：** Deep-Live-Cam — Real-time face swap and one-click video deepfake with a single image
- **类型：** repo
- **作者 / 仓库：** hacksider / Deep-Live-Cam
- **官方代码：** <https://github.com/hacksider/Deep-Live-Cam>
- **核对 commit：** [`0bae1a549014d86457548b05a01a3f7682cdda65`](https://github.com/hacksider/Deep-Live-Cam/tree/0bae1a549014d86457548b05a01a3f7682cdda65)（2026-11-08 `main`）
- **模型：** <https://huggingface.co/hacksider/deep-live-cam/tree/main>
- **官网：** <https://deeplivecam.net/>
- **许可证：** [AGPL-3.0](https://github.com/hacksider/Deep-Live-Cam/blob/main/LICENSE)（代码）；模型另行核对用途限制
- **入库日期：** 2026-11-08
- **核对范围：** 官方 README 全文、LICENSE、requirements.txt、run.py、modules/core.py、modules/ui.py 中 Live / webcam worker 路径、modules/video_capture.py、modules/face_analyser.py、modules/processors/frame/{core.py,face_swapper.py,face_enhancer.py}、modules/predicter.py；GitHub Releases 与官网版本边界。
- **通读：** 已完成
- **开源结论：** **公开可运行代码**（本地 Python / ONNX Runtime 推理），但完整的付费预构建版本/Ultimate 特性和关联模型许可不等同于开源仓库。

## 核实的主要入口与模块

| 入口（官方文件） | 实际职责 |
|----------------|---------|
| [`run.py`](https://github.com/hacksider/Deep-Live-Cam/blob/main/run.py) | 添加 ffmpeg 及平台 GPU 动态库路径，进入 `modules.core.run()` |
| [`modules/core.py`](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/core.py) | 命令行参数、环境检查、GPU execution provider；图片、视频处理入口 |
| [`modules/ui.py`](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/ui.py) | PySide6 GUI；`MainWindow._on_live()`、`WebcamPreviewWindow`、采集与推理 QThread；Qt 预览输出 |
| [`modules/video_capture.py`](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/video_capture.py) | `VideoCapturer.start/read/release`；按平台开启摄像头 |
| [`modules/face_analyser.py`](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/face_analyser.py) | InsightFace 人脸分析，源脸提取、目标脸检测、可选多脸与关键点 |
| [`modules/processors/frame/face_swapper.py`](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_swapper.py) | `get_face_swapper()`、`swap_face()`、`process_frame()`，模型推理及回贴、可选 mouth mask / Poisson / opacity |
| [`modules/processors/frame/face_enhancer.py`](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/face_enhancer.py) | 可选人脸增强后处理 |
| [`modules/processors/frame/core.py`](https://github.com/hacksider/Deep-Live-Cam/blob/main/modules/processors/frame/core.py) | 帧处理模块动态加载、线程池视频处理及 FFmpeg 管道内存路径 |

## 已核实的实时调用链

`python run.py` → `core.run()` → `ui.init()` → GUI `_on_live()` → 预加载 analyser/swapper → `WebcamPreviewWindow` → `VideoCapturer` → `_CaptureWorker` 的有界队列 → `_ProcessingWorker` 按间隔检测目标人脸并缓存 → `swap_face()`（模型推理 + 回贴 + 选配掩膜/增强）→ 输出有界队列 → Qt `_tick()` 绘制实时预览。

- 实时的两个队列 `maxsize=2`，溢出时丢旧帧；这是降低积压延迟的实现，**不是固定 FPS 保证**。
- `_ProcessingWorker` 以约 `round(camera_fps * 0.08)` 的间隔重新检测人脸，其余帧可复用检测缓存；人脸快速运动、遮挡、角度变化可能造成失配。
- 单图/视频渲染走 `modules/core.py::start()`。视频默认优先 FFmpeg 内存管道，失败时回退磁盘抽帧；有 `map_faces` 时走磁盘流程。
- **OBS 边界**：本次读取的 GitHub `main` Live 路径实际输出到 Qt 预览，README 指引使用 OBS 等屏幕采集工具。未在该路径确认直接写入系统虚拟摄像头设备的实现；2.6d Release 宣布的 Virtual Camera 属另一个具体版本声明，不应混写成当前开源主分支已验证能力。

## 配置、性能、许可边界

- README 手动安装建议 Python 3.14（支持 3.11–3.14），要求 pip、git、ffmpeg、模型文件；`requirements.txt` 含 InsightFace 0.7.3、PySide6、ONNX Runtime 等。CPU 可运行但慢。
- NVIDIA 加速需与 CUDA/cuDNN、`onnxruntime-gpu` 的版本组合匹配；不可拿网站订阅版推荐配置等同于开源版的确定性能。
- 所需模型在 README 列为 `inswapper_128_fp16.onnx` 和 `gfpgan-1024.onnx`；模型文件需单独获取。
- 仓库代码 AGPL-3.0；README 特别提醒 InsightFace 相关模型**仅限非商业研究用途**。商用前应单独审查权重授权及素材/人物肖像许可。
- 项目要求获得真实人脸当事人同意，并清晰标注合成内容；不得用于未授权冒充。
- 2026-11-08 通过 GitHub Repository API 核验 Stars 为 **97,780**，并非“10 万+”；动态指标需注明日期。
- **未复现**：本次为文档与源码静态核验，未执行 GPU 端到端跑分，不宣称稳定 FPS、嘴型精度、无遮挡效果。

## 相关归档

- [项目官网与发布版本核对](../sites/deep-live-cam-official.md)
- [Deep-Live-Cam 实体节点](../../wiki/entities/deep-live-cam.md)
