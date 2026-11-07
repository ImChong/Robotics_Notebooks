---
type: entity
title: TensorRT
date: 2026-06-25
tags: [framework, deployment, tensorrt, inference, nvidia, jetson, gpu]
status: complete
summary: "NVIDIA TensorRT 是将 ONNX/PyTorch 等模型编译为 GPU 专属 engine 的推理加速 SDK（当前 11.4.x，强类型网络为默认），通过融合、量化与 kernel 调优在 Jetson 与 dGPU 上实现机器人感知与策略的毫秒级 onboard 推理；引擎绑定 GPU 与版本，重建后须验证数值。"
updated: 2026-11-07
original:
  - ../../sources/repos/tensorrt-official.md
sources:
  - ../../sources/repos/tensorrt-official.md
  - ../../sources/repos/onnxruntime-official.md
  - ../../sources/repos/apxinf-robo.md
---

# TensorRT

**NVIDIA TensorRT** 是 NVIDIA 的 **深度学习推理加速 SDK 生态**：在 **build 阶段**将训练好的网络（常见入口为 [ONNX](./onnx.md) 或 PyTorch）编译为针对目标 GPU 优化的 **serialized engine（plan）**；在 **runtime 阶段**由应用加载 engine 执行前向。它基于 **CUDA**，官方文档列出的精度有 **FP32/FP16/BF16/FP8/INT8**，另有仅权重的 INT4。覆盖 **数据中心 GPU、GeForce RTX、Jetson、DRIVE** 等平台。在机器人研究与工程中，TensorRT 是 **NVIDIA 机载算力**（Orin、Thor、4090 等）上把感知网络与 tracker 策略压到 **毫秒级** 的首选 runtime。本页以官方文档（docs.nvidia.com/deeplearning/tensorrt/latest，当前 11.4.1）与 NVIDIA/TensorRT 仓库 README 为一手依据，2026-11-07 核对。

## 一句话定义

**NVIDIA GPU 上的推理编译器 + runtime**：把通用 ONNX 图变成 **目标 GPU 专属高速 engine**，以延迟与吞吐换取平台绑定。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TRT | TensorRT | NVIDIA 推理优化 SDK |
| ONNX | Open Neural Network Exchange | 最常见上游交换格式，TensorRT 11.x 的 parser 支持 opset 9–25 |
| ORT | ONNX Runtime | 可通过 TensorRT EP 调用 TRT（配 TensorRT 11.x 须 ORT ≥1.27） |
| EP | Execution Provider | ORT 中的算子后端插件 |
| engine / plan | Serialized Engine | build 阶段产物，序列化的 GPU 专属可执行网络 |
| FP16 / BF16 / FP8 | 16-bit float / bfloat16 / 8-bit float | 低精度推理格式；FP8 在 Orin 上不被支持（见下） |
| INT8 | 8-bit Integer Quantization | 整数量化，需校准或 Q/DQ 显式量化 |
| PTQ | Post-Training Quantization | 训练后量化；NVIDIA Model Optimizer 提供 PTQ 与 QAT |
| QDQ | Quantize/DeQuantize | 显式量化节点；TensorRT 11 用显式量化取代隐式量化 |
| DLA | Deep Learning Accelerator | Jetson AGX Orin 上的专用加速器；Thor 不支持 |
| Jetson | NVIDIA 嵌入式 AI 平台 | Orin/Thor 等人形机载板卡 |
| CUDA | Compute Unified Device Architecture | TRT 依赖的 GPU 并行平台 |
| MIG / MPS | Multi-Instance GPU / Multi-Process Service | GPU 分区 / 多进程共享，用于故障隔离与吞吐 |
| LLM | Large Language Model | TensorRT-LLM 子生态覆盖 |

## 为什么重要

- **机载延迟标杆**：本库 [Humanoid-GPT](./paper-humanoid-gpt.md) 报告 **ONNX + TensorRT <1.5 ms**（4090）；[RF-DETR](./rf-detr.md) 在 T4/Jetson 上以 **TensorRT FP16** 达实时检测（这两条来自本库其他页，本页未回查原论文）。
- **Jetson 生态默认优化路径**：JetPack 7 把 TensorRT 与 CUDA、cuDNN 一起列为 AI 计算栈组件；感知、VLA、扩散去噪等 GPU 密集型 onboard 模块在 Orin 上文献普遍写 **TRT** 或 **ORT+TensorRT EP**。
- **与 ONNX 分工清晰**：训练侧导出 `.onnx`；部署侧 `trtexec` 或 Builder API 生成 **engine**——engine **绑定 GPU 架构**，换卡常需重编。
- **工具链完整**：官方称 TensorRT 生态包含 TensorRT SDK、TensorRT-LLM、**Model Optimizer**（量化、剪枝、蒸馏）与 TensorRT for RTX，并可经 Triton / Dynamo 服务化；开发者主页的「相对仅 CPU 最高 36×」是厂商口径，具体取决于模型与硬件。

## 核心原理

### 两阶段模型

1. **Build Phase**：Builder 编译网络，为每层在目标 GPU 上选择最快的 **CUDA kernel**（靠计时选择），做层融合与精度处理；输出 **engine 二进制**（plan）。创建 engine 后，builder、network、parser、build config 都可以销毁，engine 仍可用。
2. **Runtime Phase**：运行时加载 engine，`ICudaEngine` 创建 `IExecutionContext` 执行推理。工厂对象的生命周期须覆盖其创建的对象。
3. **入口（Quick Start）**：转换有四条路径——Torch-TensorRT、ONNX 自动转换（`trtexec` 或 ONNX parser，要求算子全覆盖，缺失算子需插件）、Nsight Deep Learning Designer 图形工具、手工用 TensorRT 网络 API 搭建；其中 ONNX 路径被官方称为"最通用且性能较好的自动转换路径"。最简命令：`trtexec --onnx=model.onnx --saveEngine=model.engine`。

### TensorRT 11 的变化（OSS README 公告）

弱类型网络及相关 API 已移除，**所有网络均为强类型**；隐式量化移除，改为显式量化（Q/DQ）；IPluginV2 移除，改为 IPluginV3；TREX 工具移除，改用 Nsight Deep Learning Designer；Python 3.9 及以下绑定移除。升级须读 Migration Guide。强类型网络下精度由模型里各张量的数据类型决定，而不是 builder 的精度标志（推测的读法；具体的 Precision Control 文档本次未读）。

### 数值与确定性

builder 靠计时选 kernel，会受 GPU 上其他负载和时钟波动影响，**同一模型每次重建可能选出不同实现、数值略有差异**；因此每次重建后都要用容差验证精度。引擎构建后，同一环境、同一 batch 槽位、同一邻居输入给出相同输出。

### 运行时选项

| 运行时 | 库 | 取舍 |
|--------|----|------|
| Full（默认） | `libnvinfer` / `tensorrt` | 算子最全，体积最大 |
| Lean | `libnvinfer_lean` / `tensorrt_lean` | 体积小，须 version-compatible 引擎，算子实现少，可能更慢 |
| Dispatch | `libnvinfer_dispatch` | 跨版本加载 lean 运行时的垫片 |

## 工程实践

### 构建与部署流程

1. **导出**：训练框架导出 ONNX，opset 落在 9–25（TensorRT 11.x 的 parser 范围），先用 `trtexec --onnx=...` 验证能否导入；报错时可用 Polygraphy 常量折叠、ONNX-GraphSurgeon 简化，或自写插件。
2. **在目标板上构建**：engine 不可跨平台（Linux/Windows），默认不可跨 GPU 架构；`version-compatible` 引擎可在更新的运行时上跑，但跨大版本只支持 10→11 与 8→10；`hardware compatibility` 模式可跨架构（可能更慢）。实际项目如 [APXInf](./apxinf.md) 的构建会按所在 GPU 的计算能力编译 kernel，Orin 为 `sm_87`、Thor 为 `sm_110`（APXInf README），因此"在部署机上构建"是通用做法。
3. **验证数值**：用固定输入对拍 ONNX（ORT CPU）与 engine 输出，设容差；引擎重建后重复。
4. **固定输入形状**：locomotion 策略应固定 batch=1 与输入 shape 再 build，避免动态 shape 的 profile 切换抖动；若必须动态，需设置 min/opt/max 优化 profile。
5. **运行**：启动时一次性反序列化 engine、创建 context，控制环里只做 `enqueue`；**每线程一个 `IExecutionContext`**，同一 engine 可并发创建多个 context；`CUDA_MODULE_LOADING=LAZY` 可降低峰值显存与初始化时间（官方要求在目标配置上实测三项指标）。
6. **基准**：官方最佳实践是"先测量再优化"，用 `trtexec` 给出可复现的延迟/吞吐基线，并控制 GPU 时钟、功耗/热节流等环境变量。

### 内存与稳定性（Jetson 尤其重要）

- builder workspace 默认等于**整块设备显存**，生产构建应用 `setMemoryPoolLimit(kWORKSPACE, N)` 设上限，在显存与 CPU 共享的 Jetson 上尤其如此（统一内存设备上 `cudaMemGetInfo` 的结果可能不准，官方文档）。
- 构建期主机内存：同一网络最少会同时驻留两份权重；个别大 ONNX 网络在 TensorRT 11.1 的主机内存占用明显高于 11.0，11.2.1 已修复；用 `trtexec --monitorMemory` 观察。
- **CUDA 错误是"粘性"的**：一次异步 CUDA 错误后同进程同设备的后续调用都会失败，try/catch 无法恢复；官方建议进程级隔离（由 systemd 等监督者重启进程）、MPS 或 MIG。对机器人而言意味着：**急停与关节保护不能依赖推理进程存活**（应独立于 GPU 进程，推理进程崩溃后由监督者拉起）。
- **引擎文件是可执行代码**：反序列化不受信任的引擎等同于执行不受信任的原生代码；只加载自己构建或经签名校验的引擎，不要把第三方 `.engine` 当模型文件分发。
- **DLA**：TensorRT 11.4.1 在 Jetson AGX Orin（JetPack 7.2.2+）上支持 DLA；Jetson Thor 与 DriveOS **不支持** DLA。

### 精度选择

TensorRT 11 文档列出 FP32/FP16/FP8/BF16/INT8，INT4 仅权重量化，量化采用对称方案。**Orin 没有 FP8 Tensor Core，FP8 仅 Thor 可用**（出处是 [APXInf](./apxinf.md) README，TensorRT 文档的交互式 GPU 精度对照表本次未能取得）。INT8/FP8 须代表性校准数据，闭环行走须吊架复测。

### 与 ORT 的搭配

经 ORT 的 TensorRT EP 调用时，**TensorRT 11.x 须配 ORT 1.27 或更新**（TensorRT 快速入门页）；注册顺序用 `['TensorrtExecutionProvider','CUDAExecutionProvider']`；务必打开 engine cache / timing cache，否则首次建引擎可能要几分钟（见 [ONNX Runtime](./onnxruntime.md)）。

### 与机器人研究的关系

- **全身控制**：[Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md) 真机层与 [ONNX Runtime](./onnxruntime.md) 并列；**极致延迟**场景优先 TRT（[Booster RoboCup demo](./booster-robocup-demo.md)：仿真 ORT / 真机 TRT）。
- **扩散/生成中间件**：[ETH G1 扩散 locomotion](./paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md) 等用 **TensorRT ~20 ms** 两步去噪（来自本库其他页）。注意 [PredActor](./paper-predactor.md) 在 G1 的 Orin NX 上是以 TorchScript + LibTorch 的 C++ 路径、FP32 达到 p50 16.79 ms / p95 19.38 ms，并非 TensorRT，说明 Orin NX 上 20 ms 预算并不必然要靠 TensorRT。
- **感知**：[RF-DETR](./rf-detr.md)、[object-detection 选型](../queries/object-detection-model-selection.md) — Jetson 上 TRT 为默认 benchmark 后端。
- **非 NVIDIA 硬件**：**不适用**；应选 [ONNX Runtime](./onnxruntime.md)、[MNN](./mnn.md)、[OpenVINO](./openvino.md) 等。

## 局限与常见误区

- **「ONNX 文件 = 已优化部署」**：TRT 还需 **build engine**；且 engine 与 **GPU 架构、平台、TensorRT 版本** 绑定。
- **ORT TensorRT EP ≠ 独立 engine**：API 与调试体验不同；极致性能常直接链 TRT runtime API。
- **动态 shape 与固定控制环**：locomotion 策略应 **固定输入 shape** 再 build，避免 runtime 重编译抖动。
- **重建即可能换数值**：builder 计时选 kernel 会随负载变化，升级驱动/TensorRT 或重新构建后必须重验输出。
- **量化回归**：INT8/FP8 须代表性校准集；闭环行走须在吊架复测（与 MNN/TFLite 同理）。
- **粘性错误与引擎信任**：见"内存与稳定性"；推理进程崩溃不能让机器人失去保护。
- **算子覆盖**：ONNX 路径要求全部算子可被 TensorRT 或插件处理，否则转换失败。
- **本页没有核对**：Performance Benchmarking 各章的具体数字、插件开发细节、DLA 调优、TensorRT-LLM、各版本 Release Notes 的逐条变更、Torch-TensorRT 的行为。

## 流程总览（ONNX → Engine → 机载）

```mermaid
flowchart LR
  onnx[.onnx 或 PyTorch]
  build[TensorRT Builder<br/>融合 / 量化 / kernel 选型]
  engine[.plan / engine 文件]
  rt[Runtime API / trtexec]
  loop[机载 C++ 控制或感知环]
  onnx --> build --> engine --> rt --> loop
```

## 关联页面

- [ONNX](./onnx.md)
- [ONNX Runtime](./onnxruntime.md)
- [MNN](./mnn.md)
- [OpenVINO](./openvino.md)
- [RF-DETR](./rf-detr.md)
- [Humanoid-GPT](./paper-humanoid-gpt.md)
- [APXInf](./apxinf.md) — Thor/Orin 上 VLA 的专用推理引擎，对照 TensorRT 的取舍
- [ONNX Runtime vs MNN vs TensorRT](../comparisons/onnxruntime-vs-mnn-vs-tensorrt.md)
- [机器人运控开发板选型](../comparisons/robot-policy-deployment-dev-board-selection.md) — 各板推荐的模型格式
- [NVIDIA CUDA](./nvidia-cuda.md) — TRT 依赖的 GPU 平台与驱动/CTK 对齐

## 参考来源

- [NVIDIA TensorRT 官方站点与文档索引](../../sources/repos/tensorrt-official.md)（原文，已通读）
- [ONNX Runtime 官方站点与文档索引](../../sources/repos/onnxruntime-official.md)（TensorRT EP 页）
- [APXinf-robo 仓库归档](../../sources/repos/apxinf-robo.md)（Orin 无 FP8、按 `sm_` 架构编译）

## 推荐继续阅读

- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- [How TensorRT Works](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html)
- [Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [NVIDIA/TensorRT（GitHub）](https://github.com/NVIDIA/TensorRT)
