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
- **复核日期：** 2026-11-07（docs.nvidia.com/deeplearning/tensorrt/latest 文档页、developer.nvidia.com/tensorrt、raw.githubusercontent.com 上 NVIDIA/TensorRT 的 README、PyPI 元数据）
- **核对范围：** 开发者主页；文档《How TensorRT Works》全页（对象生命周期、引擎反序列化信任边界、错误隔离分级、内存、CUDA lazy loading、L2 持久缓存、线程安全表、确定性、Full/Lean/Dispatch 运行时）；《Quick Start Guide》与《Example Deployment Using ONNX》（含导出 PyTorch→ONNX 一节开头）；《Support Matrix》文字部分（引擎可移植性、平台列表；版本/GPU 精度对照表为前端交互渲染，未能取得）；《ONNX Opset Guide》；《Best Practices》开头与生产检查清单；《Working with Quantized Types》开头；C API 页中与强类型网络有关的段落；OSS 仓库 README。**未读**：Performance Benchmarking 各章、Plugin/自定义层、DLA 指南、Torch-TensorRT、TensorRT-LLM、各版本 Release Notes 的逐条内容、TensorRT 源码。
- **通读：** 已完成
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

## 2026-11-07 全文复核补充（官方文档）

- **版本：** PyPI `tensorrt` 最新 **11.4.0.106**；文档 Release Notes 适用 **11.4.1**（Latest），支持矩阵列出 11.4.1 / 11.4.0 / 11.3.0 / 11.2.1 / 11.1.0 / 11.0.0 与 10.x；平台含 NVIDIA JetPack AArch64（Jetson AGX Orin / Thor，JetPack 7.2.2+；Orin 另有 JetPack 6.x 条目）。OSS 仓库 README：TensorRT GA 构建为 v11.4.0.106；插件与 ONNX parser 为开源子集（Apache-2.0）；推荐 CUDA 13.4.0 / 12.9.0；有 Jetson/Thor 交叉编译与原生构建示例。
- **两阶段：** Build（Builder 编译网络并为每层选当前 GPU 上最快的 kernel，输出序列化 engine / plan）与 Runtime（加载 engine 并执行）。创建 engine 之后，builder、network、parser、build config 可销毁，engine 仍可用；工厂对象的生命周期须覆盖其创建的对象。
- **TensorRT 11 破坏性变化（OSS README 公告）：** 弱类型网络及相关 API 已移除，改为**强类型网络**（文档：所有网络都是强类型，无需设置 kSTRONGLY_TYPED）；隐式量化移除，改为显式量化；IPluginV2 移除，改为 IPluginV3；TREX 移除，改用 Nsight Deep Learning Designer；Python 3.9 及以下绑定移除。
- **ONNX 路径：** TensorRT 11.x 的 ONNX parser 对应 ONNX 包 1.20.0，**最大 opset 25，向后兼容到 opset 9**；转换要求算子全覆盖（或提供插件）；官方推荐导出时用框架所能支持的最新 opset，并用 `trtexec --onnx=...` 先验证导入，必要时用 Polygraphy 常量折叠、ONNX-GraphSurgeon 简化图；最简命令 `trtexec --onnx=model.onnx --saveEngine=model.engine`（TensorRT 11 默认强类型，无需再加标志）。支持精度：FP32、FP16、FP8、BF16、INT8，INT4 仅权重；量化采用对称方案。
- **引擎信任边界：** 序列化引擎含编译好的 CUDA tactic 与插件调用指针，**反序列化不受信任的引擎等同于执行不受信任的原生代码**，只应反序列化自己构建的引擎或经签名、完整性校验的引擎。
- **可移植性（Support Matrix）：** 引擎不可跨平台（如 Linux/Windows）；默认不可跨 GPU 架构（需 hardware compatibility 模式）；version-compatible 标志构建的引擎可在更新的 TensorRT 运行时上运行，但跨大版本只支持 TensorRT 10→11 与 8→10。
- **线程与错误：** TensorRT 对象一般非线程安全；约定「每线程一个 IExecutionContext」，同一 engine 可并发创建多个 context；CUDA 异步错误可能是「粘性」的，进程内 try/catch 无法恢复，官方建议进程级隔离（监控重启）、MPS 或 MIG；builder workspace 默认等于整块显存，生产应设 `setMemoryPoolLimit(kWORKSPACE, N)`；同一 GPU 上并行多 builder 会互相干扰计时，导致次优引擎。
- **确定性：** builder 靠计时选 kernel，受其他负载和时钟波动影响，**每次重建可能选出不同 kernel、数值略有差异，须在每次重建后验证精度**；构建后同一环境、同一 batch 槽位、同一邻居输入则输出一致（IFillLayer 随机填充与重复索引 scatter 除外）。可用可编辑 timing cache 固定选择。
- **运行时库：** Full（默认，覆盖最全）、Lean（体积小，需 version-compatible 引擎，可能更慢）、Dispatch（跨版本加载 lean 的垫片）。
- **DLA：** TensorRT 11.4.1 支持 Jetson AGX Orin（JetPack 7.2.2+）上的 DLA；**Jetson Thor 与 DriveOS 不支持 DLA**。
- **开发者主页：** 「TensorRT 生态」包含 TensorRT SDK、TensorRT-LLM、NVIDIA Model Optimizer、TensorRT for RTX；主页宣称相对仅 CPU 平台最高 36× 加速（厂商口径，以具体模型与硬件为准）；TensorRT for RTX 的引擎可跨 OS、跨 GPU，构建 15–30 秒。
