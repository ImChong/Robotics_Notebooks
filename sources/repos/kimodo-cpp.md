# kimodo.cpp — 原始资料归档

- **来源**：https://github.com/localai-org/kimodo.cpp
- **类型**：repo
- **机构**：LocalAI（GitHub `localai-org`；权重托管 Hugging Face `LocalAI-io`）
- **归档日期**：2026-08-28
- **上游**：NVIDIA [nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) / [项目页](https://research.nvidia.com/labs/sil/projects/kimodo/) / [arXiv:2603.15546](https://arxiv.org/abs/2603.15546)
- **许可**：移植代码 **Apache-2.0**；GGML 子模块与模型权重各保留原许可
- **开源结论（步骤 2.5）**：**已开源** — GitHub 公开 C++/GGML 推理仓（无独立 `*.github.io` 项目页，以仓库 README / `PORTING.md` / `docs/IMPLEMENTATION.md` 为项目入口）；SOMA / G1 的 F32 GGUF 已发布；SMPL-X 权重因 NVIDIA Internal R&D 许可禁止再分发，须用户自行转换
- **快照**：约 511 stars、43 forks；C++ 为主；创建于 2026-08-22

## 一句话说明

**kimodo.cpp** 把 NVIDIA [Kimodo](../../wiki/entities/kimodo.md) 文本→运动扩散去噪器移植到 **C++ / GGML**：CPU 或 Vulkan 上加载原生 GGUF，吃 UTF-8 prompt 或预计算 LLM2Vec 嵌入，输出局部旋转 + root 平移；不依赖 Python/PyTorch 运行时。

## 为什么值得保留

- 官方 Kimodo Python 栈全 GPU 约 **17 GB VRAM**（文本编码器占大头）；本仓把 **8B LLM2Vec 与运动去噪器串行加载**，把「本地 / 嵌入式 / 无 Python」推理从研究栈里拆出来
- 与 llama.cpp **不是** 简单包一层：文本侧自实现 **双向 Llama + mean pooling**（LLM2Vec 改了因果注意力），运动侧是独立 root/body Transformer + DDIM
- 机器人选型上它是 Kimodo **部署档**：能出骨架 GLB / C ABI 运动缓冲，但 **一般约束输入、77 关节 SOMA 展开、蒙皮 GLB、量化权重尚未实现**

## 能力边界（README，截至 2026-08）

已实现：GGUF 校验加载、safetensors→GGUF 转换、DDIM 采样、C/C++ API、多 prompt 过渡、CPU/Vulkan 对等测试、仅骨架 GLB 导出、本地文生运动 Demo。

未实现：通用运动学约束输入、SOMA 30→77 关节展开、蒙皮网格 GLB、量化模型。

## 模型与权重

可运行检查点接受 UTF-8 prompt 或预计算 **4096-d F32 LLM2Vec** 嵌入：

| 变体 | 骨架（原生 API 实际预测） | HF（`LocalAI-io`） | 上游许可 |
|------|---------------------------|---------------------|----------|
| SMPL-X RP v1 | 22 关节 | **不发布**（须本地转换 gated 权重） | NVIDIA Internal Scientific R&D（禁止商用与衍生模型再分发） |
| SOMA RP / SEED v1.1 | 紧凑 **30** 关节控制骨架（官方 Python 再展开到 77） | [Kimodo-SOMA-RP-v1.1-GGML](https://huggingface.co/LocalAI-io/Kimodo-SOMA-RP-v1.1-GGML) 等 | NVIDIA Open Model License（模型许可允许商用） |
| G1 RP / SEED v1 | 34 个 Unitree G1 关节 | [Kimodo-G1-RP-v1-GGML](https://huggingface.co/LocalAI-io/Kimodo-G1-RP-v1-GGML) 等 | NVIDIA Open Model License |

文本编码器单独发布：[Llama-3-Kimodo-GGML](https://huggingface.co/LocalAI-io/Llama-3-Kimodo-GGML)（含转换后的 Meta Llama 3 材料，条款独立）。SOMA-RP 运动 GGUF 约 **0.3B / F32 1.13 GB**。安装器：`scripts/download_gguf_weights.sh`（校验 manifest + SHA-256）；`--motion-only` 用于只喂预计算嵌入。

## 运行时接口（工程入口）

| 入口 | 位置 / 命令 |
|------|-------------|
| C ABI | `include/kimodo/kimodo_capi.h`：`kimodo_model_load` / `kimodo_generate` / `kimodo_generate_embedding`；输出 `local_rotations_xyzw [T,J,4]` 与 `root_positions [T,3]` |
| 设备 | `KIMODO_DEVICE_AUTO` / `CPU` / `VULKAN`；文本层分块 `KIMODO_TEXT_LAYER_CHUNK=1..32` |
| 构建 | C++23 + CMake 3.25+ + Ninja；`cmake --preset debug`；可选 Nix flake；另有 `release` / `asan-ubsan` / `fuzz` |
| 权重 | `scripts/download_gguf_weights.sh --output "$PWD" --model soma-rp-v1.1`；测试套件**不会**自行下载 |
| Demo | `go run ./demo -addr 0.0.0.0:8094` → 本地 WebGL；成功样本写 `demo-output/.../animation.glb` |
| 源码主路径 | `src/text_encoder` + `llama_bi`（双向 Llama）→ `src/denoiser`（root/body）→ `src/diffusion`（CFG/DDIM）→ `src/motion_rep` 逆变换 → `src/skeleton` FK |

GGML 以 **pinned git submodule**（`ggml-org/ggml`）引入；**不** vendoring 整棵 llama.cpp。

## 对 wiki 的映射

1. **[kimodo.cpp（实体页）](../../wiki/entities/kimodo-cpp.md)** — 本地 GGML 运行时、许可边界与缺失能力
2. **[Kimodo（上游实体）](../../wiki/entities/kimodo.md)** — Python 官方栈、约束编辑、Benchmark
3. **[Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md)** — 部署侧对照（Python 17GB vs C++ CPU/Vulkan）
4. **[HY-Motion vs GENMO vs Kimodo](../../wiki/comparisons/hy-motion-vs-genmo-vs-kimodo.md)** — 选型里补「本地推理档」

## 关联原始资料

- [Kimodo 官方仓](./kimodo.md)
- [Kimodo 项目页](../sites/kimodo-project.md)
- [Kimodo 论文摘录](../papers/kimodo_arxiv_2603_15546.md)
