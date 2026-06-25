---
type: entity
title: OpenVINO
date: 2026-06-25
tags: [framework, deployment, openvino, inference, edge-ai, intel, physical-ai]
summary: "OpenVINO 是 Intel 开源推理工具包，优化 ONNX 等模型在 Intel CPU/GPU/NPU 上的部署，含 GenAI 与 Physical AI（机器人 VLA）专章，亦常作为 ONNX Runtime 的 OpenVINO EP。"
updated: 2026-06-25
---

# OpenVINO

**OpenVINO**（Open Visual Inference and Neural network Optimization）是 **Intel** 开源的 **AI 推理部署工具包**，面向 **云、AI PC、边端与 Physical AI（机器人等）**。它将 **ONNX、TensorFlow、PaddlePaddle** 等模型转换或直连为 OpenVINO 表示，在 **Intel CPU / GPU / NPU** 上执行优化推理，并通过 **NNCF** 做压缩。2026 文档将 **OpenVINO Physical AI** 单列为机器人/VLA 部署路径；同时 OpenVINO 也是 [ONNX Runtime](./onnxruntime.md) 的 **OpenVINO Execution Provider** 之一。

## 一句话定义

**Intel 硅上的推理优化栈**：一次开发、多 Intel 设备部署，并可选 **Model Server** 服务化。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OV | OpenVINO | Intel 开源推理工具包 |
| IR | Intermediate Representation | OpenVINO 内部模型表示 |
| NNCF | Neural Network Compression Framework | 量化/压缩工具 |
| OVMS | OpenVINO Model Server | 服务端推理组件 |
| ONNX | Open Neural Network Exchange | 可直接加载或转换的输入格式 |
| ORT | ONNX Runtime | 可通过 OpenVINO EP 调用 |
| VLA | Vision-Language-Action | Physical AI 专章覆盖的机器人策略类型 |
| NPU | Neural Processing Unit | Intel AI Boost 等加速单元 |

## 为什么重要？

- **Intel 机器人/Physical AI 叙事**：官方 **Physical AI** 文档直接面向 **机器人 VLA onboard**，与纯 Jetson/CUDA 栈形成硬件对照。
- **ORT 生态一环**：同一 `.onnx` 可在 ORT 内注册 **OpenVINO EP**，无需重写应用即可触达 Intel 优化。
- **PC/工控机常见**：无独显或 Intel Arc 的 **开发机、小型工控** 上，OpenVINO 是 ONNX 落地的自然选项。
- **GenAI 扩展**：**OpenVINO GenAI** 覆盖本地大模型，与机载 VLA 探索相关。

## 核心结构（2026.2 文档归纳）

1. **OpenVINO Runtime**：C++/Python API；支持 **CPU 先行编译 + 异步切换 GPU/NPU** 降低首帧延迟。
2. **模型导入**：直连 ONNX/TF/Paddle，或转为 OpenVINO IR。
3. **OpenVINO GenAI** — 生成式模型管线
4. **OpenVINO Physical AI** — 机器人等物理智能部署
5. **OpenVINO Model Server（OVMS）** — Kubernetes/微服务推理
6. **PyTorch**：`torch.compile` OpenVINO 后端

## 与机器人研究与工程的关系

- **硬件对照**：[TensorRT](./tensorrt.md) 绑定 NVIDIA；OpenVINO 绑定 **Intel**——选型先锁板卡。
- **感知 on Intel**：IoT/工控视觉在 Intel CPU+VPU 上常用 OpenVINO；与 [ncnn](./ncnn.md)/[MNN](./mnn.md) 的 ARM 移动路径不同。
- **跨框架**：与 [ONNX](./onnx.md) 标准格式互补；不必替换 PyTorch 训练栈。

## 常见误区或局限

- **非 Intel 硬件收益有限**：AMD/NVIDIA 独显场景应优先 ORT CUDA/TRT。
- **IR 转换可能有数值差**：须在目标设备上用固定输入回归。
- **与人形 WBC 默认栈距离较远**：本库人形高频控制文献更多写 **ORT/TRT**；OpenVINO 更常出现在 **Intel 边端感知 / VLA Physical AI** 叙事。

## 关联页面

- [ONNX](./onnx.md)
- [ONNX Runtime](./onnxruntime.md)
- [TensorRT](./tensorrt.md)
- [MNN](./mnn.md)
- [ncnn](./ncnn.md)
- [ONNX Runtime vs MNN vs TensorRT](../comparisons/onnxruntime-vs-mnn-vs-tensorrt.md)

## 参考来源

- [Intel OpenVINO 官方文档索引](../../sources/repos/openvino-official.md)

## 推荐继续阅读

- [OpenVINO 文档](https://docs.openvino.ai/)
- [Physical AI 专章](https://docs.openvino.ai/2024/openvino-workflow/physical-ai.html)
- [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
