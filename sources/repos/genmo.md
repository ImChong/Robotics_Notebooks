# GENMO / GEM（NVlabs/GENMO 代码仓与权重发布）

- **标题**: GEM: A Generalist Model for Human Motion（原名 **GENMO**，2025-12 更名为 **GEM**）
- **论文**: <https://arxiv.org/abs/2505.01425>（ICCV 2025 Highlight）
- **机构页**: <https://research.nvidia.com/labs/dair/gem/>（旧地址 `…/genmo/` 已 302 重定向）
- **代码**: <https://github.com/NVlabs/GENMO>
- **类型**: code-release（与 `sources/papers/genmo.md` 分工：本文件聚焦仓库与权重侧；论文方法走 papers/）
- **机构**: NVIDIA Research（DAIR 等）
- **License**: Apache-2.0（第三方组件见 `ATTRIBUTIONS.md`）
- **首次入库**: 2026-05-07
- **最后更新**: 2026-05-16

## 一句话摘要

NVIDIA 官方实现，把扩散式的「带观测约束的运动生成」做成统一权重，支持视频、2D 关键点、文本、音乐、3D 关键帧条件下的全身 SMPL 序列恢复与合成；CLI 演示 / 训练 / 实时 webcam 全套脚本齐备。

## 仓库时间线（README News，截至 2026-03）

- **2026-03** — 发布 **GEM-SMPL** checkpoint 与 multi-modal demo 脚本（HuggingFace `nvidia/GEM-X`）。
- **2025-12** — GENMO 正式更名为 **GEM**；项目主页与 README 全面切换品牌（旧检索词仍可用）。
- **2025-10** — GEM 代码库首次开源。

## 仓库结构与可复现入口（README 摘要）

- **环境**：Python 3.10 + `uv` + Torch + 自带 `scripts/install_env.sh`。
- **预训练权重**：`GEM-SMPL`（SMPL body model，regression + generation 一体；text/audio/music/video 条件）→ HuggingFace `nvidia/GEM-X`。
- **核心 demo**
  - `scripts/demo/demo_smpl.py` — 多模态条件演示（mp4 / `.txt` prompt / `text:...` 字面文本可任意串联）；默认每段文本预算 300 帧 ≈ 10 s。
  - `scripts/demo/demo_smpl_hpe.py` — 纯视频 → SMPL 位姿估计的简化路径。
  - `scripts/demo/demo_webcam.py` — 实时 webcam 管线 YOLOX → ViTPose-H → HMR2 → GEM 去噪，ONNX Runtime；可选 OpenCV 叠绘或 Viser 3D 视图，支持 `--no_imgfeat` 跳过 HMR2 提速。
- **训练入口**
  - 回归模型：`python scripts/train.py exp=gem_smpl_regression`（body=SMPLx, AdamW lr=2e-4, fp16-mixed, max=500K steps, grad-clip=0.5）。
  - 完整模型：`python scripts/train.py exp=gem_smpl`。
  - 多 GPU：`pl_trainer.devices=4`；SLURM：`scripts/train_slurm.py`。
- **输出文件**：`outputs/<name>_mix/` 下 `1_incam.mp4`（相机系 mesh 叠加）、`2_global.mp4`（世界系）、`3_incam_global_horiz.mp4`（左右对比）、`smpl_params.pt`（含 `body_params_global` / `body_params_incam` / `K_fullimg` / `segment_info`）。

## 与 NVIDIA 人形栈的关系（README "Related Humanoid Work"）

- [GEM-X](https://github.com/NVlabs/GEM-X) — GEM 的全身（含手、脸）扩展。
- [SOMA-X](https://github.com/NVlabs/SOMA-X) — SOMA body model。
- BONES-SEED 数据集（README 中标记将发布）。
- [ProtoMotions](https://github.com/NVlabs/ProtoMotions) — 大规模并行人形仿真与 RL。
- SOMA Retargeter（README 中标记将发布）。
- [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) — 规模化运动跟踪的全身控制策略。
- [Kimodo](https://github.com/nv-tlabs/kimodo) — 文本 + 运动学约束的运动扩散与生成工具链。

## 引用 / 命名

- **论文 BibTeX（README 主推 ICCV 版本）**：`@inproceedings{genmo2025, title={GENMO: A GENeralist Model for Human MOtion}, author={Li, Jiefeng and Cao, Jinkun and Zhang, Haotian and Rempe, Davis and Kautz, Jan and Iqbal, Umar and Yuan, Ye}, booktitle={ICCV}, year={2025}}`。
- **项目页另一个 BibTeX 变体（journal=arXiv）** 仍保留早期 title「GENMO: Generative Models for Human Motion Synthesis」，检索两种 title 均可命中同一工作。
- **品牌**：论文 / arXiv 仍为 **GENMO**；仓库与权重发布以 **GEM** 为新名；检索代码与 checkpoint 时两个关键词都要试。

## 对 Wiki 的映射

- **`wiki/methods/genmo.md`**：人体运动估计/生成方法页（本次同步补全 dual-mode 训练、multi-text 注入、NVIDIA 人形栈关联）。
- **`sources/papers/genmo.md`**：配套的论文级摘录，承载方法学细节。
- **`wiki/methods/exoactor.md`**：作为「生成视频 → SMPL 全身轨迹」环节的下游消费者。
- **`wiki/methods/sonic-motion-tracking.md`**：官网与 README 将 GEM/GENMO 与 SONIC 拼成「人体运动 → 通用跟踪」演示。
- **`wiki/methods/diffusion-motion-generation.md`**：扩散范式在人体运动域的代表实现交叉引用。
- **`wiki/entities/protomotions.md`**：同属 NVIDIA 人形栈的训练框架；本次在 GENMO 方法页中显式建立横向关联。
