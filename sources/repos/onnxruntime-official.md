# ONNX Runtime 官方站点与文档索引

> 来源归档（以 onnxruntime.ai 官网与 GitHub README 叙述为准；Execution Provider 列表以目标版本文档为准）

- **标题：** ONNX Runtime — Accelerated Edge Machine Learning
- **类型：** 跨平台推理/训练引擎 + 多语言 API
- **主页：** https://onnxruntime.ai/
- **文档：** https://onnxruntime.ai/docs/
- **核心代码：** https://github.com/microsoft/onnxruntime
- **最新跟踪版本：** [v1.28.0（2026-07-25）](./onnxruntime-v1.28.0.md) — CUDA 13 与轻量 CUDA 部署
- **Generative AI 扩展：** https://onnxruntime.ai/docs/genai/（`onnxruntime-genai`）
- **入库日期：** 2026-06-25（索引）；**续更：** 2026-07-26（接入 1.28.0）；**2026-09-09**（文档首页 / Python 入门 / EP API 复核）
- **复核日期：** 2026-11-07（onnxruntime.ai 文档页、raw.githubusercontent.com 上 microsoft/onnxruntime 的 README 与 cmake/deps.txt、PyPI 元数据）
- **核对范围：** `README.md`；文档首页；Get started with Python；Execution Providers 总览；NVIDIA CUDA EP 全页（安装要求、配置项、CUDA Graph 约束）；NVIDIA TensorRT EP 全页（要求表、用法、配置项、缓存与 EPContext、动态形状）；Graph Optimizations 全页；Thread management 全页；Compatibility 页；各 tag（v1.28.0、v1.31.1、main）的 `cmake/deps.txt`。**未读**：github.com 上的 v1.28.0 / v1.29–v1.31 Release Notes（github.com 与 api.github.com 对本环境返回 403）、onnxruntime 源码实现、OpenVINO/QNN/CoreML 等其他 EP 页、量化与 Mobile 文档、ORT Web 教程。
- **通读：** 已完成
- **开源状态：** **已开源**（MIT；完整源码 + Release 预编译资产）
- **一句话说明：** 微软主导的 **生产级 ONNX 推理与训练加速引擎**，支持 **Python / C++ / C# / Java / JavaScript** 等语言，覆盖 **Linux / Windows / macOS / iOS / Android / Web**；通过 **Execution Provider（EP）** 对接 CPU、CUDA、TensorRT、OpenVINO、CoreML、NNAPI 等后端；广泛用于 **Windows、Office、Azure、Bing** 及机器人 **C++ 机载策略推理**（如 Unitree G1 WBC、AMP_mjlab 部署链）。**1.28.0** 起官方并行提供 **CUDA 12 / CUDA 13** GPU 包，且 CUDA EP 可将 cuDNN/cuFFT 作运行时可选依赖以缩小 redistributable。
- **沉淀到 wiki：** [ONNX Runtime](../../wiki/entities/onnxruntime.md)

---

## 版本锚点：v1.28.0（2026-07-25）

详见专档 [onnxruntime-v1.28.0.md](./onnxruntime-v1.28.0.md)。工程侧优先记住三点：

1. **CUDA 13 打包线**：Release 资产含 `gpu_cuda12` 与 `gpu_cuda13`；NPM 改走 CUDA 13 pipeline。
2. **轻量 GPU 部署**：CUDA EP **cuDNN / cuFFT 运行时可选**，**不再链接 `nvrtc`**。
3. **格式依赖**：捆绑 **ONNX 1.22.0**（+ protobuf 6.33.5）；升级机载钉扎版本时须回归。

---

## 文档首页要点（2026-09-09 复核）

### 定位

- **跨平台 ML 模型加速器**：灵活接口集成硬件专用库；可消费 PyTorch、TensorFlow/Keras、TFLite、scikit-learn 等框架导出的模型。
- 推理侧支撑 Office、Azure、Bing 及大量社区项目；机器人栈常见 **训练 Python → 导出 ONNX → C++/Java 机载 ORT 推理**。

### 推理三步（官方叙述）

1. **获取模型** — 自各框架导出/转换为 ONNX。
2. **加载并运行** — `InferenceSession` + `session.run()`；多语言入门见 [Get started with Python](https://onnxruntime.ai/docs/get-started/with-python.html) 等。
3. **（可选）调优** — Session/EP 配置、图优化、硬件加速器；见 Performance 文档。

### 运行时机制（归纳）

- ORT 对模型图做 **图优化**，再按可用 EP **划分子图** 到 CPU/CUDA/TensorRT 等后端执行。
- **模型验证责任**：ORT 仅校验 ONNX 规范合规；**精度、性能与恶意模型风险** 由应用方负责（官方明确提醒超大算力/内存消耗型恶意图）。

### 训练分支

- **Large Model Training** — 大模型训练加速。
- **On-Device Training** — 端侧训练叙事（机器人控制环较少直接用）。

### 快速安装与模板

- CPU：`pip install onnxruntime`；GPU 默认 CUDA 12.x：`pip install onnxruntime-gpu`；CUDA 11.8 须用 Azure DevOps feed（见 Python 入门页）。
- 生成式：`pip install onnxruntime-genai`（[GenAI 文档](https://onnxruntime.ai/docs/genai/)）。
- 官方 QuickStart 模板：**ORT Web JavaScript Site**、**ORT C# Console App**（文档首页链接）。

### 最小 Python 推理

```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"input": input_tensor})
```

### Python 多框架导出速查（文档入门页）

| 来源 | 导出方式 |
|------|----------|
| PyTorch | `torch.onnx.export(...)`（PyTorch 已内置 ONNX） |
| TensorFlow/Keras | `tf2onnx.convert.from_keras(...)` |
| scikit-learn | `skl2onnx.convert_sklearn(...)` |

> **包互斥**：`onnxruntime` 与 `onnxruntime-gpu` 同一环境只装其一；GPU 包涵盖大部分 CPU 能力。

---

## 与 ONNX 格式的关系

- **ONNX**（[onnx.ai](https://onnx.ai/)）定义 **`.onnx` 文件与算子规范**。
- **ONNX Runtime** 是 **执行该格式的运行时** 之一（另有 TensorRT 直接 ingest ONNX、MNN 经 convert 等路径）。
- 机器人栈常见分工：**PyTorch/JAX 训练 → 导出 ONNX → ORT（CPU/GPU EP）或 ORT+TensorRT EP 上机**。

---

## Execution Provider（EP）概念（归纳）

ORT 通过 **SessionOptions** / `providers` 参数注册 EP；`GetCapability()` 将节点或子图分配给 EP 库在对应硬件执行。

| EP（示例） | 典型场景 |
|------------|----------|
| CPU | 通用回退、x86/ARM 机载 |
| CUDA | NVIDIA GPU 数据中心 / Jetson |
| TensorRT | NVIDIA 上进一步图优化与 INT8/FP16 |
| OpenVINO | Intel CPU/GPU/VPU |
| CoreML / NNAPI | iOS / Android 移动 |

### Python EP API（2026-09 文档）

```python
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
sess = ort.InferenceSession("model.onnx", providers=EP_list)
# 运行时改优先级（会重建 session）：
sess.set_providers(['CPUExecutionProvider'])
```

- `get_providers()` — 已注册 EP 列表。
- `get_provider_options()` — 各 EP 配置。
- `set_providers([...])` — 按 **优先级顺序** 重注册 EP。

> 具体 EP 可用性与算子覆盖须以 [官方 EP 文档](https://onnxruntime.ai/docs/execution-providers/) 与目标 `.onnx` 图为准。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [ONNX](../../wiki/entities/onnx.md) | 格式规范；ORT 是最常用的兼容 runtime |
| [wbc-fsm](../../sources/repos/wbc_fsm.md) | G1 **纯 C++ + ONNX Runtime 1.22** 部署 LAFAN1 WBC 策略 |
| [AMP_mjlab](../../wiki/entities/amp-mjlab.md) | 训练导出 ONNX → C++ ORT 推理 |
| [jackhan-feap-mujoco-deployment](../../wiki/entities/jackhan-feap-mujoco-deployment.md) | README 固定 `onnxruntime==1.19.2` 类版本提示 |
| [BotLab MotionCanvas](../../sources/sites/botlab_motioncanvas.md) | 浏览器端 **ONNX Runtime WASM/WebGPU** 编排 obs→policy |
| [Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md) | 真机对比提及 ONNX + TensorRT 低延迟部署 |

---

## 对 wiki 的映射

- 维护 **`wiki/entities/onnxruntime.md`**：runtime 实体页（EP、语言绑定、机器人 C++ 部署；含 **1.28.0** 版本锚点）。
- 参与 **`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`** 选型对比。
- 版本专档：**`sources/repos/onnxruntime-v1.28.0.md`**。

---

## 外部参考（便于复核）

- [ONNX Runtime 官网](https://onnxruntime.ai/)
- [文档首页](https://onnxruntime.ai/docs/)
- [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [microsoft/onnxruntime（GitHub）](https://github.com/microsoft/onnxruntime)
- [Release v1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)

## 2026-11-07 全文复核补充（官方文档 + PyPI + deps.txt）

- **最新版本：** PyPI `onnxruntime` 最新 **1.31.1**（2026-10-26）；1.28.0（2026-07-25）之后依次 1.29.0（2026-08-17）、1.30.0（2026-09-10）、1.31.0（2026-10-09）、1.31.1（2026-10-26）；`requires_python >= 3.11`。tag `v1.28.0` 与 `v1.31.1` 的 `cmake/deps.txt` 均为 **onnx v1.22.0**；`main` 已升到 onnx v1.23.2。
- **与 [v1.28.0 专档](./onnxruntime-v1.28.0.md) 的差异：** 专档摘自 Release Notes 的「protobuf 6.33.5」，而 tag v1.28.0 的 `deps.txt` 里 protobuf 为 **v21.12**（v1.31.1 与 main 为 v33.6），二者口径不同（可能前者指 Python 侧依赖）；本次因 Release Notes 无法取得，未能判定。
- **CUDA 包线（CUDA EP 文档）：** **自 1.27 起** PyPI `onnxruntime-gpu` 与 NuGet GPU 包默认用 CUDA 13.0 构建；另发 CUDA 12.x 包（用 CUDA 12.8 构建，要求 12.8 或更新）；更早版本默认 CUDA 12.8。cuDNN 8.x 与 9.x 不兼容，须与环境主版本一致。Python 入门页仍写「默认 CUDA 12.x」，两页口径不一致，以所装版本的 CUDA EP 页为准。
- **EP 机制：** 通过 `GetCapability()` 把节点/子图分给 EP；`providers=[...]` 按优先级排列，`set_providers()` 会重建 session；官方建议注册 TensorRT EP 时同时注册 CUDA EP 兜底不支持的节点。同一环境只装一个 Python 包（CPU 或 GPU）。
- **图优化三级：** Basic（常量折叠、冗余节点消除、Conv-Add/Mul/BatchNorm 融合等，EP 划分之前，适用所有 EP）、Extended（划分之后，只对 CPU/CUDA/ROCm 分到的节点）、Layout（NCHWc，仅 CPU）；默认全开，可离线保存优化后的图。
- **线程（Thread management）：** `intra_op_num_threads` 默认 0 = 物理核数；`execution_mode` 默认 ORT_SEQUENTIAL（分支多的图可试 ORT_PARALLEL）；线程池默认自旋等待（`session.intra_op.allow_spinning`），更快但更耗 CPU/功耗，可用 spin_duration_us / spin_backoff_max 调节。
- **CUDA Graph（CUDA EP，Preview）：** 去掉逐个 kernel 启动的 CPU 开销；要求整个模型都能划到 CUDA EP、输入输出的形状与地址在多次调用间不变、必须用 IOBinding。
- **TensorRT EP：** 依赖表（main/1.22 对应 TensorRT 10.9、CUDA 12.0–12.8）；`trt_fp16_enable`、`trt_int8_enable`（INT8 非 QDQ 模型需校准表）、`trt_engine_cache_enable`、`trt_timing_cache_enable`、`trt_cuda_graph_enable`、`trt_profile_{min,opt,max}_shapes` 等；**引擎缓存不可跨硬件/ORT 版本/TensorRT 版本移植**，这些变化后须清缓存；官方示例 SD UNet 的会话创建时间：无缓存 384 s → timing cache 42 s → engine cache 9 s → 嵌入式引擎模型（EPContext，`_ctx.onnx`）1.9 s；`trt_engine_hw_compatible` 不适用于 Turing 及更早与 **Jetson Orin**；Jetson 的预编译包与镜像见 Jetson Zoo。TensorRT 官方快速入门页另注：**与 TensorRT 11.x 配合须用 ONNX Runtime 1.27 或更新**，更早版本引用了 TensorRT 11.0 移除的 API。
- **兼容性页：** 新版 ORT 支持旧版 ORT 能跑的所有模型；支持最新发布 ONNX 规范的所有 opset，且所有 ORT 版本至少支持 opset 7+（ONNX 1.2.1+）；该页的「ORT–ONNX 版本对照表」只更新到 ORT 1.20（ONNX 1.16.1 / opset 21 / IR 10），已落后于现状。
