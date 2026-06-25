# MNN 官方文档与仓库索引

> 来源归档（以 GitHub README、mnn-docs.readthedocs.io 与 PyPI 包页叙述为准）

- **标题：** MNN — Mobile Neural Network
- **类型：** 轻量推理引擎 + 模型转换工具 + LLM/Diffusion 扩展
- **主页（Workbench）：** http://www.mnn.zone
- **文档：** https://mnn-docs.readthedocs.io/en/latest/
- **核心代码：** https://github.com/alibaba/MNN
- **PyPI：** https://pypi.org/project/mnn/（`pip install MNN`）
- **维护方：** 阿里巴巴 MNN Team
- **入库日期：** 2026-06-25
- **一句话说明：** 阿里开源的 **高效轻量深度学习推理引擎**，在移动端与嵌入式场景经大规模生产验证（淘宝、天猫、优酷等 30+ App）；支持 **CNN / Transformer / LLM / Diffusion**；提供 **`mnnconvert`** 从 **ONNX / TensorFlow / PyTorch / Caffe** 等转入 **`.mnn`**，并配套 **权重量化、离线量化、NPU 后端** 与 **MNN-LLM** 本地大模型运行时。
- **沉淀到 wiki：** [MNN](../../wiki/entities/mnn.md)

---

## GitHub README 要点（alibaba/MNN）

1. **定位**：「A blazing-fast, lightweight inference engine battle-tested by Alibaba, powering high-performance on-device LLMs and Edge AI.」
2. **能力**：支持推理与（实验性）训练；强调 **on-device** 性能领先。
3. **生产规模**：集成于阿里 30+ 应用、70+ 场景（直播、短视频、搜索推荐、图搜、营销、风控等）；亦用于 IoT 嵌入式。
4. **扩展子项目**：
   - **MNN-LLM**：基于 MNN 的本地 LLM 运行时（手机/PC/IoT）；支持通义、百川、智谱、LLAMA 等（见 [LLM 文档](https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html)）。
   - **MNN Diffusion**：端侧 Stable Diffusion 类方案。
5. **工具链**：
   - **MNN Workbench**（官网下载）：预训练模型、可视化训练、一键下发设备。
   - **mnnconvert / mnnquant / mnncompress**：转换、量化与压缩。

---

## 文档要点（Read the Docs）

### 安装

```bash
pip install MNN
```

源码编译见 [Pymnn 构建](https://mnn-docs.readthedocs.io/en/latest/compile/pymnn.html)。

### ONNX → MNN 转换（快速开始示例）

```bash
mnnconvert -f ONNX --modelFile mobilenet_v1.onnx --MNNModel mobilenet_v1.mnn --weightQuantBits 8
```

### Python Module API（推荐）

```python
import MNN.nn as nn
net = nn.load_module_from_file("model.mnn", ["input"], ["output"])
output = net.forward([input_var])
```

### 运行时后端（文档归纳）

后端选项示例：**0=CPU, 1=Metal, 2=CUDA, 3=OpenCL, 7=Vulkan**；另文档专章说明 **NPU 及相应后端**。

### API 层次

- **Module API**（推荐）：高层 `nn.load_module_from_file`
- **Expr API**：表达式构图
- **Session API**：已标记 deprecated，新代码应避免

### 配套 Python 模块

- **MNN**：推理、训练、图像处理、数值计算（`expr` / `nn` / `cv` / `numpy` 等）
- **MNNTools**：封装 `mnnconvert`、`mnnquant` 等 CLI

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [ONNX](../../wiki/entities/onnx.md) | MNN 常见入口格式；`mnnconvert -f ONNX` |
| [ONNX Runtime](../../wiki/entities/onnxruntime.md) | 同为推理 runtime；ORT 在 x86/Jetson C++ 人形栈更常见，MNN 在 **移动/国产 ARM/NPU** 侧更强 |
| [TensorFlow](../../wiki/entities/tensorflow.md) | 支持 TF 图转入 MNN |
| [PyTorch](../../wiki/entities/pytorch.md) | 训练后常先导出 ONNX 再转 MNN |
| [Sim2Real](../../wiki/concepts/sim2real.md) | 边端算力受限时，`.mnn` + 量化是可选 onboard 路径 |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/mnn.md`**：边端推理引擎实体页。
- 参与 **`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`** 与 ONNX 实体互链。

---

## 外部参考（便于复核）

- [MNN 文档（Read the Docs）](https://mnn-docs.readthedocs.io/en/latest/)
- [Python 快速开始（5 分钟）](https://mnn-docs.readthedocs.io/en/latest/start/quickstart_python.html)
- [模型转换工具](https://mnn-docs.readthedocs.io/en/latest/tools/convert.html)
- [MNN-LLM 用户指南](https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html)
- [alibaba/MNN（GitHub）](https://github.com/alibaba/MNN)
- [PyPI: mnn](https://pypi.org/project/mnn/)
