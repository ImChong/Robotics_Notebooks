# openvinotoolkit/openvino — OpenVINO 主仓库

> 来源归档（以 GitHub README 与仓库元数据为准；版本以 Release 标签为准）

- **标题：** OpenVINO™ Toolkit
- **类型：** repo / inference-runtime / deployment
- **仓库：** https://github.com/openvinotoolkit/openvino
- **许可：** Apache License 2.0
- **入库日期：** 2026-09-09
- **开源状态：** **已开源**（完整 C++/Python/C/Node.js 运行时、转换器、样例与开发者文档）
- **一句话说明：** Intel 开源的 **深度学习推理优化与部署工具包**：将 PyTorch、TensorFlow、ONNX、PaddlePaddle、JAX/Flax、TFLite 等训练产物 **转换或直连** 为 OpenVINO 模型，在 **Intel CPU / GPU / NPU**（及 ARM CPU）上执行硬件加速推理；配套 **NNCF 压缩**、**GenAI**、**Physical AI 机器人运行时** 与 **Model Server** 生态。
- **沉淀到 wiki：** [OpenVINO](../../wiki/entities/openvino.md)

---

## 仓库要点（2026-09-09 抓取）

### 定位

「Open-source software toolkit for optimizing and deploying deep learning models」—— 覆盖计算机视觉、ASR、GenAI、NLP/LLM 等常见任务；强调 **一次编写、多硬件部署** 与 **最小外部依赖** 的轻量部署 footprint。

### 核心能力

| 能力 | 说明 |
|------|------|
| **推理优化** | 在 Intel 硅上提升吞吐、降低延迟，兼顾精度 |
| **灵活模型支持** | PyTorch / TF / ONNX / Keras / Paddle / JAX；Hugging Face **Optimum Intel** 直连 transformers / diffusers |
| **跨平台** | CPU（x86、ARM）、GPU（Intel 集显与独显 Arc）、NPU（AI Boost 等） |
| **多语言 API** | C++、Python、C、Node.js；GenAI 专用 API |
| **快速安装** | `pip install -U openvino`；亦支持 conda、brew、系统包 |

### 典型推理路径（README 示例）

**PyTorch → OpenVINO → CPU 推理：**

```python
import openvino as ov
import torch
import torchvision

model = torch.hub.load("pytorch/vision", "shufflenet_v2_x1_0", weights="DEFAULT")
example = torch.randn(1, 3, 224, 224)
ov_model = ov.convert_model(model, example_input=(example,))

core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')
output = compiled_model({0: example.numpy()})
```

**TensorFlow → OpenVINO：**

```python
import openvino as ov
import tensorflow as tf
import numpy as np

model = tf.keras.applications.MobileNetV2(weights='imagenet')
ov_model = ov.convert_model(model)
compiled_model = ov.Core().compile_model(ov_model, 'CPU')
output = compiled_model({0: np.random.rand(1, 224, 224, 3)})
```

### 生态子仓库（README 列举）

| 子项目 | 仓库 | 角色 |
|--------|------|------|
| NNCF | [openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf) | 量化、稀疏化等压缩 |
| GenAI | [openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai) | LLM / 生成式管线 |
| Tokenizers | [openvinotoolkit/openvino_tokenizers](https://github.com/openvinotoolkit/openvino_tokenizers) | GenAI 分词器 |
| Model Server | [openvinotoolkit/model_server](https://github.com/openvinotoolkit/model_server) | OVMS 服务化推理 |
| Physical AI | [openvinotoolkit/physicalai](https://github.com/openvinotoolkit/physicalai) | 机器人 VLA 部署运行时 |
| Notebooks | [openvinotoolkit/openvino_notebooks](https://github.com/openvinotoolkit/openvino_notebooks) | 教程与示例 |
| Awesome | [openvinotoolkit/awesome-openvino](https://github.com/openvinotoolkit/awesome-openvino) | 社区项目合集 |

### 集成（README 列举）

- **Optimum Intel** — Hugging Face 模型一键 OpenVINO 后端
- **torch.compile** — PyTorch 原生 JIT 到 OpenVINO kernel
- **ExecuTorch** — PyTorch 移动端后端
- **vLLM OpenVINO** — [vllm-openvino](https://github.com/vllm-project/vllm-openvino) 服务加速
- **ONNX Runtime EP** — 现有 ORT 代码注册 OpenVINO 后端
- **LangChain / LlamaIndex / LLMWare / Keras 3** — GenAI 框架绑定

### 开发者文档

- 用户文档：https://docs.openvino.ai/
- 开发者文档：仓库内 `docs/dev/`（架构、构建、贡献）

### 遥测

默认收集性能与使用遥测（GA4）；可 `opt_in_out --opt_out` 退出。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [ONNX](../../wiki/entities/onnx.md) | 可直接加载 ONNX 或经 `convert_model` 转 IR |
| [ONNX Runtime](../../wiki/entities/onnxruntime.md) | OpenVINO 作为 ORT **OpenVINO EP** |
| [TensorRT](../../wiki/entities/tensorrt.md) | 对照：NVIDIA GPU 极致优化 vs Intel 硅 |
| [Ultralytics](../../wiki/entities/ultralytics.md) | YOLO 导出 `format=openvino` |
| [LeRobot](../../wiki/entities/lerobot.md) | Physical AI 支持 LeRobot 模型导出与 Intel 部署 |
| [VLA 方法页](../../wiki/methods/vla.md) | Physical AI 专章面向 VLA onboard |

---

## 对 wiki 的映射

- 深化 **`wiki/entities/openvino.md`**：主仓推理路径、生态表、Physical AI 交叉引用
- 链入 **`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`** 延伸 runtime 表（已有）
- 在 **`wiki/entities/lerobot.md`** 补充 Physical AI 部署对照一句

---

## 外部参考

- [openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
- [OpenVINO 文档](https://docs.openvino.ai/)
- [OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
- [Release Notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html)
