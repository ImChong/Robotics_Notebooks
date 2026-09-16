# Depth Anything 3: Recovering the Visual Space from Any Views

> 来源归档（ingest · 上游模型）

- **标题：** Depth Anything 3: Recovering the Visual Space from Any Views
- **缩写：** **DA3** / Depth Anything 3
- **类型：** paper / monocular-depth / multi-view-geometry / pose-estimation / 3d-reconstruction / foundation-model
- **arXiv：** <https://arxiv.org/abs/2511.10647>（HF Papers: <https://huggingface.co/papers/2511.10647>）
- **代码：** <https://github.com/ByteDance-Seed/Depth-Anything-3>（**已开源**）
- **权重：** <https://huggingface.co/depth-anything>（DA3 全系列；许可因型号而异）
- **机构：** 字节跳动 Seed（ByteDance Seed）
- **作者：** Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, Bingyi Kang（†project lead）
- **状态：** arXiv 2025-11；代码/权重 2025-11 发布；含 DA3-Streaming（2025-12）
- **入库日期：** 2026-09-16
- **一句话说明：** 用 **单一 plain transformer（vanilla DINO）** + **depth-ray 统一表征**，从任意数量、有无位姿的视图预测 **空间一致几何**；单目深度超越 DA2，多视图深度/位姿超越 VGGT，并支持 3D Gaussian 与 metric 分支。

## 摘录 1：建模极简主义

- **骨干：** 单个 plain transformer（如 vanilla DINO encoder）即可，**无需架构特化**。
- **表征：** 统一的 **depth-ray prediction target**，避免复杂多任务头分裂。
- **训练：** teacher-student 范式；细节与泛化与 **Depth Anything 2** 同级。
- **任务面：** 单目深度、多视图一致深度、位姿条件深度、相机内外参估计、3D Gaussian 预测、metric depth（独立分支）。

**对 wiki 的映射：** 作为 [`depth-anything.cpp`](../../wiki/entities/depth-anything-cpp.md) 的上游；交叉见 [`wiki/entities/paper-r3-relative-regression.md`](../../wiki/entities/paper-r3-relative-regression.md)、[`wiki/entities/paper-track4world.md`](../../wiki/entities/paper-track4world.md)

## 摘录 2：模型族（官方 README 摘要）

| 系列 | 代表 checkpoint | 能力 | 许可（典型） |
|------|-----------------|------|--------------|
| Main | DA3-GIANT/LARGE/BASE/SMALL | 深度 + 位姿 +（GIANT）3DGS | Large/Giant: CC BY-NC；Small/Base: Apache-2.0 |
| Metric | DA3METRIC-LARGE | 单目 metric depth + sky | Apache-2.0 |
| Mono | DA3MONO-LARGE | 高质量相对单目深度 + sky | Apache-2.0 |
| Nested | DA3NESTED-GIANT-LARGE | any-view + metric 对齐（米制） | CC BY-NC 4.0 |

`-1.1` 后缀为修复训练 bug 后的重训版；街景优先选 `-1.1`。

## 摘录 3：开源边界（步骤 2.5 · 2026-09-16）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)；Python API + CLI + Gradio |
| **权重** | **已发布** — Hugging Face `depth-anything/*` |
| **工程栈** | 推理默认 PyTorch + xformers；可选 gsplat 高斯头、Gradio app |
| **社区移植** | [depth-anything.cpp](../repos/depth-anything-cpp.md)（C++/ggml GGUF，MIT） |
| **商用** | 须逐模型查 HF model card；Large/Giant/Nested 多为 **非商业** |

## 对 wiki 的映射

- C++ 运行时：**[`wiki/entities/depth-anything-cpp.md`](../../wiki/entities/depth-anything-cpp.md)**
- 仓库归档：**[`sources/repos/depth-anything-cpp.md`](../repos/depth-anything-cpp.md)**（推理引擎，非官方仓）
