# Robbyant 技术官网与 LingBot 项目页（technology.robbyant.com）

> 来源归档（ingest · 官方技术站首页 + 尚未单独归档的项目页）

- **标题：** Robbyant - Exploring the Frontiers of Embodied Intelligence | 蚂蚁灵波科技 - 探索具身智能上限，打造物理世界的 AGI 平台
- **类型：** site / project-page 索引
- **机构：** 蚂蚁灵波科技（Robbyant），蚂蚁集团旗下
- **官方入口：** <https://technology.robbyant.com/>
- **入库日期：** 2026-09-29
- **一句话说明：** 官方把 LingBot 七个模型按「感知 → 空间 → 视频 → 世界 → 动作」组织成全栈具身智能叙事；本档补齐 Vision / Depth / Video / VA 2.0 四个此前未单独归档的项目页。
- **为什么值得保留：** 版本号（如 VA 2.0、World 2.0）与每个模型的一句话定位以官网为准；VA 2.0 的架构细节目前只在项目页和仓内 PDF 中公开。

## 首页叙事（2026-09-29 抓取，引号内为原文）

- 公司定位："An embodied AI company under Ant Group, building one brain for all robots."
- 使命："We build full‑stack embodied intelligence — from spatial perception and action models to environmental reward — to create one brain for all robots."

| 模型（官网版本号） | 官网一句话 |
|--------------------|------------|
| LingBot-Vision 1.0 | "A spatial‑perception‑native vision foundation model by masked boundary modeling" |
| LingBot-Depth 1.0 | "High‑accuracy depth perception that helps robots see transparent and reflective objects" |
| LingBot-Map 1.0 | "A pure autoregressive streaming 3D reconstruction foundation model for high‑accuracy reconstruction over 10,000+ video frames" |
| LingBot-World 2.0 | "A high‑fidelity, controllable, and logically consistent world model for interactive physical simulation" |
| LingBot-VLA 2.0 | "A cross‑embodiment, multi‑task open‑source embodied foundation model for robot operation" |
| LingBot-VA 2.0 | "A world‑action model for dynamic physical interaction, improving robot generalization and real‑time inference" |
| LingBot-Video 1.0 | "A video foundation model for multi‑scene dynamic generation, spanning physical motion, human action, robot tasks, and open‑ended creation" |

## 项目页要点

### LingBot-Vision — <https://technology.robbyant.com/lingbot-vision>

- **论文：** *Vision Pretraining for Dense Spatial Perception*，[arXiv:2607.05247](https://arxiv.org/abs/2607.05247)
- **代码 / 权重：** [robbyant/lingbot-vision](https://github.com/robbyant/lingbot-vision) · [HF 集合](https://huggingface.co/collections/robbyant/lingbot-vision)
- **摘要要点：** 提出 **masked boundary modeling**：先学习亚像素边界表示，再把含边界的 token 当作掩码目标来学习稠密视觉 token；以 DINOv3 为强基线评测；论文称 LingBot-Vision 推动 LingBot-Depth 从 1.0 升级到 2.0（深度补全）。

### LingBot-Depth — <https://technology.robbyant.com/lingbot-depth>

- **论文：** *Masked Depth Modeling for Spatial Perception*，[arXiv:2601.17895](https://arxiv.org/abs/2601.17895)
- **代码 / 权重：** [robbyant/lingbot-depth](https://github.com/robbyant/lingbot-depth) · [HF](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5)
- **摘要要点：** 把深度传感器的缺失与噪声视为天然的「掩码」信号，用视觉上下文补全和精修深度；配自动化数据整理管线；论文自报在深度精度与像素覆盖率上超过顶级 RGB-D 相机；公开代码、checkpoint 与 300 万 RGB-深度对（200 万真实 + 100 万仿真）。

### LingBot-Video — <https://technology.robbyant.com/lingbot-video>

- **论文：** *Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence*，[arXiv:2607.07675](https://arxiv.org/abs/2607.07675)
- **代码 / 权重：** [robbyant/lingbot-video](https://github.com/robbyant/lingbot-video) · [HF 集合](https://huggingface.co/collections/robbyant/lingbot-video) · [ModelScope](https://www.modelscope.cn/collections/Robbyant/LingBot-Video)
- **摘要要点：** 指出通用视频生成模型面向内容创作，偏重画质与创意，而非计算效率与物理真实；LingBot-Video 是 **DiT** 架构的视频预训练范式，用 **MoE**（非 dense）从零扩展；数据侧用 data profiling engine 在互联网视频之外加入 **manipulation / navigation / egocentric** 机器人向视频；训练侧用多维 reward 对齐 **physical rationality** 与 **task completion**，超出 aesthetics、prompt-following、motion consistency 等常规标准。
- **项目页自报：** "70,000+ hours of embodiment-oriented data"；Single-Stream Diffusion Transformer；MoE 30B-A3B 在 1M tokens 下相对 dense 基线约 3.18× 加速；定位为具身 AI 的 "physical-world simulator"，用于数据合成、策略评测与动作规划。

### LingBot-VA 2.0 — <https://technology.robbyant.com/lingbot-va-v2>

- **技术报告：** [LingBot_VA2_paper.pdf](https://github.com/Robbyant/lingbot-va/blob/main/LingBot_VA2_paper.pdf)（位于 `lingbot-va` 仓；入库日未查到独立 arXiv 编号）
- **项目页原文要点：**
  - **Tokenizer：** "The visual tokenizer is semantically aligned with a frozen perception encoder, and latent actions are self‑supervised by IDM / FDM from adjacent visual latents"
  - **Causal pretraining：** "A causal DiT jointly predicts future visual latents and latent actions under language and planner context"
  - **MoE：** "The video stream uses sparse MoE routed layers (top-8 of 128 experts) while the action stream stays dense"
  - **Multi-chunk prediction：** 联合预测 next-1 / next-2 / next-3 未来 chunk，减少短视 rollout 与误差累积
  - **推理：** 异步预测与执行 + re-grounding；FP8 TensorRT 等优化后端到端 "over 4x" 加速；可从单条示范视频做 in-context 适配；低频任务规划 + 高频动作控制的分层结构
- **开源状态：** 项目页只链到组织级 GitHub / HF / ModelScope；截至 2026-09-29 HF 未见 VA 2.0 专属权重，**权重未确认开源**。

## 对 wiki 的映射

- [Robbyant（蚂蚁灵波）公司实体](../../wiki/entities/robbyant.md)
- [LingBot-Vision](../../wiki/entities/cn-os-lingbot-vision.md) · [LingBot-Depth](../../wiki/entities/cn-os-lingbot-depth.md) · [LingBot-Video](../../wiki/entities/cn-os-lingbot-video.md)
- [LingBot-VA](../../wiki/entities/paper-sa-2601-21998-lingbot-va-causal-video-action-world-model-for-g.md)
