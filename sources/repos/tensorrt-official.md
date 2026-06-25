# NVIDIA TensorRT 官方站点与文档索引

> 来源归档（以 developer.nvidia.com、docs.nvidia.com/deeplearning/tensorrt 与 GitHub OSS 叙述为准；版本与 JetPack 对应关系以目标平台 Release Notes 为准）

- **标题：** NVIDIA TensorRT
- **类型：** 推理优化 SDK 生态（编译器 + runtime + 量化工具链）
- **开发者主页：** https://developer.nvidia.com/tensorrt
- **文档：** https://docs.nvidia.com/deeplearning/tensorrt/latest/
- **快速开始：** https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html
- **架构说明：** https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/how-trt-works.html
- **OSS 仓库：** https://github.com/NVIDIA/TensorRT（plugins、ONNX parser、samples 等开源子集）
- **入库日期：** 2026-06-25
- **一句话说明：** NVIDIA 的 **深度学习推理加速 SDK 生态**：将 PyTorch / ONNX / TensorFlow 等训练产物经 **build 阶段**（算子融合、kernel 选型、量化）编译为 **GPU 专属 engine（plan）**，再由 **runtime 阶段**加载执行；支持 **FP32/FP16/BF16/FP8/INT8** 等精度，覆盖 **数据中心 GPU、RTX、Jetson、DRIVE** 等；机器人栈中常见于 **Jetson Orin/Thor 感知与策略低延迟部署**。
- **沉淀到 wiki：** [TensorRT](../../wiki/entities/tensorrt.md)

---

## 开发者主页要点（2026-06-25 抓取归纳）

1. **定位**：「Ecosystem of tools for high-performance deep learning inference」—— 含 **TensorRT 编译器/runtime**、**TensorRT-LLM**、**Model Optimizer**、**TensorRT for RTX**、**TensorRT Cloud** 等子产品。
2. **工作原理**：基于 **CUDA**；通过 **量化、层/张量融合、kernel 调优** 降低延迟与显存带宽；相对 CPU-only 官方宣称可达数量级加速（以具体模型与硬件为准）。
3. **框架接入**：
   - **ONNX Parser** 导入 ONNX 图
   - **Torch-TensorRT** 与 PyTorch 集成（「一行代码」叙事）
   - **MATLAB GPU Coder** 面向 Jetson / DRIVE / 数据中心
4. **部署形态**：
   - **Runtime API**：C++/Python，最低开销、细粒度控制
   - **`trtexec` CLI**：从 ONNX 生成 engine 并 benchmark
   - **Triton Inference Server**：TensorRT backend，动态 batch、ensemble、流式输入
5. **垂直场景**：DeepStream、Riva、JetPack、DRIVE、TAO 等 NVIDIA 栈均集成 TensorRT。

---

## 文档架构要点（How TensorRT Works）

**两阶段模型**：

| 阶段 | 产物 | 说明 |
|------|------|------|
| **Build** | Serialized **engine**（plan 文件） | Builder 为每层在目标 GPU 上选择最快 kernel |
| **Runtime** | 加载 engine 执行推理 | 须注意 **对象生命周期**（Builder/Runtime/ExecutionContext 归属） |

**关键概念**：

- **Strongly typed networks**（11.x 默认）：弱类型 API（`setPrecision`、`IInt8Calibrator` 隐式量化等）已移除；升级须查 Migration Guide。
- **Lean / Dispatch Runtime**：生产部署可裁剪 runtime 体积。
- **动态 shape**：支持但机载实时控制环通常 **固定 batch=1 与输入 shape** 以利于优化。

**安装路径（文档归纳）**：

- Python：`pip install tensorrt`
- Debian/RPM、tar/zip、容器镜像
- Jetson / JetPack 随系统或 SDK Manager 分发

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [ONNX](../../wiki/entities/onnx.md) | 最常见上游格式；`trtexec --onnx=...` 或 ONNX Parser |
| [ONNX Runtime](../../wiki/entities/onnxruntime.md) | 可通过 **TensorRT EP** 在 ORT 内调用 TRT；亦可独立 TRT engine |
| [Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md) | 真机 tracker：**ONNX + TensorRT**，RTX 4090 **<1.5 ms** |
| [RF-DETR](../../wiki/entities/rf-detr.md) | `export(onnx)` → TensorRT FP16，Jetson 感知 |
| [Booster RoboCup demo](../../wiki/entities/booster-robocup-demo.md) | 仿真 ORT / 真机 **TensorRT** |
| [htwk-gym](../../wiki/methods/htwk-gym.md) | 对照：同栈亦支持 TFLite 边端路径 |
| [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md) | 真机推理层 ONNX / TensorRT @ ~50 Hz |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/tensorrt.md`**
- 更新 **`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`**：链至 TensorRT 实体，并补充 OpenVINO / NCNN / LiteRT 等延伸 runtime 一览
- 交叉更新 **`wiki/entities/onnx.md`**、**`wiki/entities/rf-detr.md`**

---

## 外部参考（便于复核）

- [TensorRT Developer Home](https://developer.nvidia.com/tensorrt)
- [TensorRT 最新文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- [NVIDIA/TensorRT（GitHub OSS）](https://github.com/NVIDIA/TensorRT)
- [Torch-TensorRT](https://pytorch.org/TensorRT/)
