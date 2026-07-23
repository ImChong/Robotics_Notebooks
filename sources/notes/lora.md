# lora

> 来源归档（ingest）：**LoRA（Low-Rank Adaptation）** 参数高效微调方法的公开资料索引，用于支撑 wiki 概念页对该术语的统一定义。

- **入库日期：** 2026-07-22
- **沉淀到 wiki：** 是 → [`wiki/concepts/lora.md`](../../wiki/concepts/lora.md)

## 一句话说明

LoRA 冻结预训练权重 $W_0$，只对其增量训练一对低秩矩阵 $B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$（$r\ll\min(d,k)$），前向变为 $W_0x+BAx$，从而以极小可训练参数量适配下游任务或新形态/动力学。

## 为什么值得保留

本库多篇 VLA / world-model / 视频微调页面（FADA、Any2Any、M4World、WAM-TTT、RLDX、mimic-video 等）都把 LoRA 当作**动力学敏感模块的低成本适配算子**在引用，但此前没有一处统一定义。收束到一条 ingest 线后，各页只需回链概念页，避免重复解释 $W'=W+BA$、rank 选择与「只训 A/B」等要点。

## 可引用来源

- 原始论文：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685（2021）。
- 官方实现：microsoft/LoRA — <https://github.com/microsoft/LoRA>
- 常见衍生：QLoRA（4-bit 量化 + LoRA）、DoRA（权重分解）、LoRA rank/alpha 缩放实践。

## 库内引用页面

- [`wiki/methods/mimic-video.md`](../../wiki/methods/mimic-video.md)
- [`wiki/entities/paper-fada-humanoid.md`](../../wiki/entities/paper-fada-humanoid.md)
- [`wiki/entities/paper-any2any-cross-embodiment-wbt.md`](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md)
- [`wiki/entities/paper-m4world.md`](../../wiki/entities/paper-m4world.md)
- [`wiki/entities/paper-wam-ttt-human-video-test-time-steering.md`](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md)
- [`wiki/entities/rldx-1.md`](../../wiki/entities/rldx-1.md)
