# Intel OpenVINO 官方文档索引

> 来源归档（以 docs.openvino.ai 2026.2 文档叙述为准）

- **标题：** OpenVINO Toolkit
- **类型：** 开源推理部署工具包 + GenAI + Physical AI 扩展
- **文档：** https://docs.openvino.ai/
- **代码：** https://github.com/openvinotoolkit/openvino
- **Hugging Face 预优化模型：** OpenVINO 模型集（文档首页入口）
- **入库日期：** 2026-06-25
- **一句话说明：** Intel 开源的 **跨云/PC/边端/Physical AI** 推理工具包：将 ONNX、TensorFlow、Paddle 等模型 **转换或直连** 为 OpenVINO IR，在 **Intel CPU/GPU/NPU** 上优化推理；2026 文档突出 **OpenVINO GenAI**、**OpenVINO Physical AI（机器人 VLA 部署）** 与 **OpenVINO Model Server**；亦常作为 [ONNX Runtime](../../wiki/entities/onnxruntime.md) 的 **OpenVINO EP** 后端。
- **沉淀到 wiki：** [OpenVINO](../../wiki/entities/openvino.md)（延伸 runtime，与 TensorRT/MNN 并列选型）

---

## 文档首页要点（OpenVINO 2026.2）

1. **定位**：「Open-source toolkit for deploying high-performance AI solutions across cloud, AI PCs, edge devices, and **Physical AI**」。
2. **四大工具**：
   - **OpenVINO Base Package** — 常规 AI 模型推理
   - **OpenVINO GenAI** — 生成式模型
   - **OpenVINO Physical AI** — **机器人 VLA 等 Physical AI 部署**（专章 `./physical-ai.html`）
   - **OpenVINO Model Server** — 服务端推理（OVMS）
3. **框架兼容**：可 **直接加载** TensorFlow、ONNX、PaddlePaddle，或转换为 OpenVINO 格式。
4. **压缩**：**NNCF** 后训练/训练时压缩。
5. **PyTorch 集成**：**torch.compile** OpenVINO 后端，PyTorch 原生应用内调用。
6. **部署特性**：一次编写多硬件部署；**CPU 先行编译 + 热切换设备** 降低首帧延迟；模型缓存加速启动。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [ONNX](../../wiki/entities/onnx.md) | 可直接加载 ONNX 或转 IR |
| [ONNX Runtime](../../wiki/entities/onnxruntime.md) | OpenVINO 作为 ORT 的 EP 之一 |
| [TensorRT](../../wiki/entities/tensorrt.md) | 对照：NVIDIA GPU 极致优化 vs Intel 硅优化 |
| [MNN](../../wiki/entities/mnn.md) | 对照：ARM 移动边端 |
| 机器人 VLA | Physical AI 专章 — Intel 侧机器人部署叙事 |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/openvino.md`**（延伸 runtime 实体，篇幅适中）
- 纳入 **`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`** 的延伸 runtime 表

---

## 外部参考

- [OpenVINO 文档](https://docs.openvino.ai/)
- [Physical AI 专章](https://docs.openvino.ai/2024/openvino-workflow/physical-ai.html)
- [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
