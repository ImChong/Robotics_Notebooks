# Open Dreamer 项目页（next-state）

> 来源归档

- **标题：** How to train a frontier-level world model
- **类型：** site / 项目页 + 工程博客
- **URL：** <https://next-state.github.io/open-dreamer/>
- **Zenodo DOI：** [10.5281/zenodo.21475232](https://doi.org/10.5281/zenodo.21475232)
- **训练代码：** <https://github.com/next-state/open-dreamer>
- **推理代码：** <https://github.com/reactor-team/open-dreamer>
- **权重：** <https://huggingface.co/reactor-team/open-dreamer>
- **上游 Dreamer 4：** <https://danijar.com/project/dreamer4/> · arXiv:[2509.24527](https://arxiv.org/abs/2509.24527)
- **入库日期：** 2026-07-25
- **一句话说明：** next-state 团队复现 Dreamer 4 的工程叙事：CoinRun 单卡起步 → Minecraft 规模化；公开 tokenizer / dynamics 训练教训、实时浏览器 demo（Reactor 赞助）。

---

## 页面要点（ingest 快照）

### 目标与搜索空间

- **目标：** 复现 Dreamer 4 论文，构建可交互的 **Minecraft 世界模型**。
- **起点：** CoinRun（单 GPU、快迭代，跑通 tokenizer → dynamics → 简易 BC/RL）。
- **约束：** 有意不引入论文外方法，避免搜索空间膨胀。

### 架构两件套（block-causal transformer）

| 模块 | 作用 |
|------|------|
| **Tokenizer（MAE）** | Transformer Masked Autoencoder，约 100× 压缩；无 KL/对抗损失；masking 使 latent 更易扩散 |
| **Dynamics** | 动作条件下一帧预测；**diffusion forcing + flow matching + shortcut models** |

空间层：单帧内 token 交互；因果时间层：跨帧传播。Dynamics 将 rollout 折成块 \(B_t=(a_{t-1}, s_t, \pi_t)\)，空间注意实现 \(a_{t-1}\to s_t\to\pi_t\)（世界 token 不可读 \(\pi\)，任务信息只能经下一动作影响未来）。

### 规模化与稳定性（博客强调）

- **算力：** JAX/XLA；约 **57–58% MFU**；256 frames/GPU 进入 compute-bound；倾向 **纯 DP** 而非激进模型并行；activation checkpointing。
- **数据：** 视频解码跟不上 GPU → **预 tokenize** 为 ArrayRecord + Grain + GPU prefetch。
- **稳定：** Muon 优于 LaProp（损失尖峰）；生成必须用 **EMA**；BF16/FP32 精度边界；**x-prediction + v-space loss**（相对 Dreamer4 权重项分母平方）；minibatch barycentric OT；认为 \(\mu\)P 非必需。

### 开源边界（页面自述）

- 宣称开源 **模型与训练代码**；浏览器可玩实时 demo。
- CoinRun 用过的 **BC/RL 代码未发布**（未用于 Minecraft，故不随仓放出）。
- 完整 Dreamer 4 agent 训练环仍在训练仓 Roadmap。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Open Dreamer](../../wiki/entities/open-dreamer.md) | 主实体：训练+推理+权重+demo |
| [DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md) | 系列前代；沙盒路线对照 |
| [虚拟沙盒路线](../../wiki/overview/world-models-route-03-virtual-sandbox.md) | Dreamer 谱系外延复现 |
| [Latent Imagination](../../wiki/concepts/latent-imagination.md) | Dreamer 核心机制演进语境 |
| [Model-Based RL](../../wiki/methods/model-based-rl.md) | MBRL / 想象训练对照 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 可交互视频级 WM 开源基线 |

## 对 wiki 的映射

- 训练仓：[`sources/repos/open-dreamer.md`](../repos/open-dreamer.md)
- 推理仓：[`sources/repos/reactor-team-open-dreamer.md`](../repos/reactor-team-open-dreamer.md)
- 实体页：**`wiki/entities/open-dreamer.md`**
