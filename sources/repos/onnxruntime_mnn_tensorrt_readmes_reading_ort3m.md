# ONNX Runtime / MNN / TensorRT 官方仓库 README 通读

> 来源归档（原文通读 · 支撑 `wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`）

- **标题：** microsoft/onnxruntime、alibaba/MNN、NVIDIA/TensorRT 三个上游仓库 README
- **类型：** repo / inference-runtime
- **原文：**
  - <https://raw.githubusercontent.com/microsoft/onnxruntime/main/README.md>
  - <https://raw.githubusercontent.com/alibaba/MNN/master/README.md>
  - <https://raw.githubusercontent.com/NVIDIA/TensorRT/main/README.md>（2026-11 获取）
- **一句话说明：** 三个机载推理 runtime 的官方自述：ORT 定位跨平台推理加速器（MIT），MNN 定位端侧轻量引擎（Apache 2.0，ARM/GPU/NPU 多后端），TensorRT-OSS 仓库是 NVIDIA 推理 SDK 的开源部分（插件、ONNX 解析器、示例）。
- **核对范围：** 三份 README 全文逐段读完（ORT 约 3 KB、MNN 约 15 KB、TensorRT 约 21 KB，含构建章节）。未阅读各项目的完整文档站（ORT EP 文档、TensorRT Developer Guide、MNN readthedocs）与 MLSys 2020 / OSDI 2022 论文，也未做任何延迟实测；wiki 页中的延迟 / 机器人使用频次判断仍依赖页面既有内容。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **ORT：** 自称"cross-platform inference and training machine-learning accelerator"；推理端接受 PyTorch、TensorFlow/Keras 以及 scikit-learn、LightGBM、XGBoost 等经典 ML 模型，利用硬件加速器并叠加图优化；许可 MIT；另有独立的 QNN Plugin EP 仓库（`onnxruntime-qnn`）。
- **MNN：** 许可 Apache 2.0（注意与 wiki 旧表 "BSD-3（PyPI 标注）" 不同，以仓库 README 为准）；集成于阿里 30+ App、70+ 场景，也用于 IoT；转换器支持 TensorFlow(lite) / Caffe / ONNX / TorchScript（README 数字：158 个 ONNX OP）；iOS 静态库约 12 MB、Android 核心 so 约 800 KB；FP16 / Int8 量化可减模型体积 50%-70%；ARMv8.2 FP16 约 2 倍、sdot / VNNI 约 2.5 倍提速。
- **MNN 后端矩阵：** CPU（x86 SSE/AVX2/AVX512、ARMv7/v8/v9、RISC-V）、GPU（OpenCL / Vulkan / Metal / CUDA）、NPU（CoreML、Hexagon、NeuroPilot、HIAI、NNAPI、QNN），按 S/A/B/C 标注成熟度；INT8/INT4 评级含仅权重量化，不代表全整数执行。
- **MNN 扩展：** MNN-LLM、MNN-Diffusion、MNN-Compress / Express / CV / Train；新闻栏提到 3.6.1 的 Hexagon 后端。
- **TensorRT-OSS：** 仓库含插件、ONNX 解析器与示例，是 GA 版本的子集；README 公告称 TensorRT 11.x 移除弱类型网络（改强类型）、隐式量化（改显式量化）、IPluginV2（改 IPluginV3）、TREX 工具；构建前提 TensorRT 11.4.0.106、CUDA 13.4 / 12.9；提供 Jetson Thor、DriveOS、QNX 的交叉编译示例；许可 Apache 2.0（SDK 本体另有许可）。

## 对 wiki 的映射

- [`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`](../../wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md)

## 参考来源

- ONNX Runtime：<https://github.com/microsoft/onnxruntime>
- MNN：<https://github.com/alibaba/MNN>
- TensorRT：<https://github.com/NVIDIA/TensorRT>
