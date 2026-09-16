# depth-anything.cpp（Depth Anything 3 的 C++/ggml 推理引擎）

> 来源归档（ingest）

- **标题：** depth-anything.cpp — C++/ggml inference engine for Depth Anything 3
- **类型：** repo / perception / monocular-depth / 3d-reconstruction / deployment / ggml / gguf
- **作者 / 组织：** [LocalAI](https://github.com/mudler/LocalAI) 团队（Ettore Di Giacinto [@mudler](https://github.com/mudler)、Richard Palethorpe）；GitHub 组织 [localai-org](https://github.com/localai-org)
- **代码：** <https://github.com/localai-org/depth-anything.cpp>（**已开源**，MIT）
- **GGUF 权重：** <https://huggingface.co/mudler/depth-anything.cpp-gguf>
- **上游模型：** [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)（论文 [arXiv:2511.10647](https://arxiv.org/abs/2511.10647)）
- **入库日期：** 2026-09-16
- **一句话说明：** 从零实现的 C++17/ggml 移植，把 DA3（及 DA2）整族 checkpoint 转为自包含 GGUF，推理期 **零 Python/PyTorch/CUDA 依赖**；单图输出 metric depth、置信度、相机内外参、天空掩码、点云与 glb/COLMAP/PLY，CPU 上比 PyTorch **更快且 bit-exact**（correlation 1.0）。

## 开源状态（步骤 2.5 · 2026-09-16）

| 项 | 核查结论 |
|----|----------|
| **GitHub** | 公开仓 [localai-org/depth-anything.cpp](https://github.com/localai-org/depth-anything.cpp)；MIT 许可证；含 CLI、`da_capi.h`、转换脚本、37 项 ctest parity 套件 |
| **HF GGUF** | [mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf) 发布量化权重（f16 / q8_0 / q4_k 等） |
| **上游 DA3** | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) **已开源**（代码 + HF 权重）；部分大模型 CC BY-NC 4.0 |
| **转换依赖** | 权重转 GGUF 与 parity 检查需 Python venv（`requirements.txt`）；**推理不需要** |
| **LocalAI 集成** | README 提供 LocalAI backend 接入路径 |
| **结论** | **已开源**（引擎 + 转换工具 + 预转换 GGUF）；商用须按各 checkpoint 原许可（Small/Base Apache-2.0；Large/Giant/Nested 多为 CC BY-NC） |

## 技术栈快照

| 模块 | 实现 |
|------|------|
| 运行时 | C++17 + [ggml](https://github.com/ggml-org/ggml)（git submodule） |
| 权重 | 自包含 GGUF（维度/超参/预处理常数 baked in） |
| 量化 | f16 / q8_0 / q6_k / q5_k / q4_k（q4_k ~99 MB，near-lossless） |
| 后端 | CPU（tinyBLAS、Winograd、flash-attention）；可选 CUDA / Metal / Vulkan |
| API | Flat C API `include/da_capi.h`；CLI `da3-cli` |
| 导出 | 无依赖 glb / COLMAP / PLY / PFM（parity-checked） |

## 支持模型族

| 系列 | 模型 | 输出 |
|------|------|------|
| **DA3 Main** | SMALL / BASE / LARGE / GIANT | depth + conf + pose（GIANT 含 3D Gaussians） |
| **DA3 Mono** | DA3MONO-LARGE | depth + sky |
| **DA3 Metric** | DA3METRIC-LARGE | metric depth + sky |
| **DA3 Nested** | DA3NESTED-GIANT-LARGE | 对齐 metric depth + pose（双分支） |
| **DA2** | V2 Small/Base/Large + Metric Hypersim/VKITTI | 相对或 metric depth（无 pose/conf） |

## 性能要点（README · Ryzen 9 9950X3D · 504×336）

| engine | quant | infer ms | peak RAM MB | vs PyTorch |
|--------|-------|---------:|------------:|-----------:|
| PyTorch | f32 | 416.9 | 1328 | 1.00x |
| C++/ggml | f32 | 346.4 | 614 | **1.20x** |
| C++/ggml | q8_0 | 319.4 | 363 | **1.31x** |
| C++/ggml | q4_k | 395.2 | 320 | 1.05x |

加载约 **6.7× 更快**；端到端 depth 与参考 DA3 forward **correlation 1.0**。

## 典型 CLI 入口

```sh
da3-cli depth --model models/depth-anything-base-f32.gguf --input photo.jpg --pfm depth.pfm --png depth.png
da3-cli depth --model $M --input photo.jpg --pose pose.json          # 内外参 JSON
da3-cli depth --model $M --input a.jpg --input b.jpg --out-prefix scene  # 多视图
da3-cli depth --model $M --input photo.jpg --glb scene.glb --colmap colmap_out/
```

## 对 wiki 的映射

- 主实体页：**[`wiki/entities/depth-anything-cpp.md`](../../wiki/entities/depth-anything-cpp.md)**
- 上游论文摘录：**[`sources/papers/depth_anything_3_arxiv_2511_10647.md`](../papers/depth_anything_3_arxiv_2511_10647.md)**
- 交叉：**[`wiki/entities/paper-r3-relative-regression.md`](../../wiki/entities/paper-r3-relative-regression.md)**（DA3 骨干）、[`wiki/entities/kimodo-cpp.md`](../../wiki/entities/kimodo-cpp.md)（同 LocalAI C++/ggml 部署范式）
