# ONNX Runtime v1.28.0（GitHub Release）

> 来源归档（以 [microsoft/onnxruntime `v1.28.0` Release Notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0) 为准；发布日 2026-07-25）

- **标题：** ONNX Runtime v1.28.0
- **类型：** 开源推理/训练引擎版本发布（repo release）
- **核心代码：** https://github.com/microsoft/onnxruntime
- **项目页 / Release：** https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0
- **对照区间：** [v1.27.1...v1.28.0](https://github.com/microsoft/onnxruntime/compare/v1.27.1...v1.28.0)
- **发布日期：** 2026-07-25（UTC）
- **入库日期：** 2026-07-26
- **开源状态：** **已开源**（MIT；完整训练/推理源码与预编译资产均公开）
- **一句话说明：** ORT **1.28.0** 主线强化 **CUDA 13 打包与轻量 GPU 部署**：CUDA EP 运行时 **cuDNN / cuFFT 可选**、不再链接 `nvrtc`，显著缩小 CUDA redistributable；官方同时提供 **CUDA 12** 与 **CUDA 13** GPU 包；依赖升级至 **ONNX 1.22.0** + protobuf 6.33.5，并含大量安全加固与 EP/量化更新。
- **沉淀到 wiki：** [ONNX Runtime](../../wiki/entities/onnxruntime.md)
- **关联索引：** [ONNX Runtime 官方站点与文档索引](./onnxruntime-official.md)

---

## 开源与资产核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 源码 | **已开源** — `microsoft/onnxruntime`（MIT） |
| Release 资产 | 公开预编译包（CPU / GPU）；GPU 分 **`cuda12`** 与 **`cuda13`** 两线 |
| 数据/权重 | 不适用（通用 runtime，非模型仓） |
| 项目页性质 | GitHub Release Notes = 本条「项目页」 |

**Linux x64 GPU 包体积对照（Release Assets，约）：**

| 资产 | 约体积 | 读法 |
|------|--------|------|
| `onnxruntime-linux-x64-gpu_cuda12-1.28.0.tgz` | ~404 MB | CUDA 12 线 |
| `onnxruntime-linux-x64-gpu_cuda13-1.28.0.tgz` | ~230 MB | CUDA 13 线；与「轻量部署」叙事一致 |

Windows 亦同步提供 `win-x64-gpu_cuda12` / `cuda13` zip。

---

## Announcements & Breaking Changes（归纳）

1. **依赖升级：** ONNX **1.22.0**、protobuf **6.33.5**；图优化器 opset 检查同步更新。
2. **轻量 CUDA 部署（核心）：** CUDA EP 运行时 **cuDNN、cuFFT 可选**；**不再链接 `nvrtc`** → 显著降低所需 CUDA redistributable 体积。
3. **实验性 C/C++ API：** `OrtModelPackageApi` 进入 experimental C API，后续可能变更。
4. **废弃 / 移除：**
   - SkipLayerNorm strict mode 废弃
   - CUDA EP 中 TensorRT fused causal attention kernels 移除
   - 动态 WGSL generator（duktape/Node）路径移除，改 Python `wgsl-gen`
   - `CUDA_QUANT_PREPROCESS` 默认关闭
5. **打包：** NPM 包改为从 **CUDA 13** pipeline 发布；CUDA 12.8 包架构列表刷新。

---

## 对机器人部署的读法

| 场景 | 1.28.0 影响 |
|------|-------------|
| Jetson / 工控机 **瘦镜像** | 优先评估 **CUDA 13 GPU 包** + 可选 cuDNN/cuFFT：镜像与运维体积更小 |
| 仍钉 **CUDA 12 / JetPack 旧栈** | 继续用官方 **`cuda12`** 资产；勿盲目升 13 |
| 机载策略 C++ Session | API 主路径稳定；勿依赖 experimental `OrtModelPackageApi` |
| 版本钉扎工程（如 ORT 1.19 / 1.22） | 升级须回归 EP 注册、`session.get_providers()`、数值对齐 |
| 依赖 ONNX opset | 与格式页对齐：本 runtime 捆绑 **ONNX 1.22.0** |

---

## 其它值得记的变更（摘要）

- **安全：** FlatBuffer 加载器硬化、多处 OOB / 整数溢出防护；protobuf CVE 缓解（含 CVE-2026-0994）。
- **CUDA EP：** XQA 默认开启（FP16/BF16 GQA）、cuDNN SDPA、QMoE / `MatMulNBits` 路径优化；修复 CUDA 13 wheel 布局与 CPU-only Linux 上误依赖 `libcudart.so.13` 的 import 问题。
- **WebGPU / Web：** KV cache 量化、FlashAttention decode 融合；NPM 走 CUDA 13 pipeline。
- **量化工具：** 新增 `CudaQuantizer`；FP8 / Direct8Bit 等修补。
- **CPU / MLAS：** AVX512/ARM64 **2-bit** 权重量化 kernel；RISC-V RVV INT8 等。

---

## 对 wiki 的映射

- 更新 **`wiki/entities/onnxruntime.md`**：写入 1.28.0 版本锚点（CUDA 13 + 轻量 CUDA 部署、ONNX 1.22.0）。
- 交叉 **`wiki/entities/onnx.md`**、**`wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md`**：版本/打包选型提示。
- 回链 **`sources/repos/onnxruntime-official.md`** 最新版本指针。

---

## 外部参考（便于复核）

- [Release v1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)
- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- [ONNX Runtime 官网](https://onnxruntime.ai/)
- [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
