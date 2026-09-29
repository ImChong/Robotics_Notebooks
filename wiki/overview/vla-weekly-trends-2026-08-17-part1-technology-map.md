---
type: overview
tags: [overview, survey, vla, technology-map, duomo-space]
status: complete
updated: 2026-09-29
related:
  - ../entities/paper-geo-vla.md
  - ../entities/paper-inference-time-attention-steering-vla-driving.md
  - ../entities/paper-nebulavla.md
  - ../entities/paper-reuse-before-you-retrieve-tta-vla.md
  - ../entities/paper-maniguard.md
  - ../entities/paper-libero-vifo.md
  - ../entities/paper-safe-pruner.md
  - ../entities/paper-rcl-2608-17209-teach-and-grow-an-agent-centered-architecture-fo.md
  - ../entities/paper-haf-humanoid-vla-adaptation.md
  - ../entities/paper-expo-ft.md
  - ../entities/paper-prism-grpo.md
  - ../entities/paper-fabrimae-vla-self-eval.md
  - ../entities/paper-sparkvla.md
  - ../entities/paper-tau0-vla.md
  - ../entities/paper-baton-long-horizon-manipulation.md
  - ../entities/paper-us-vla-ultrasound.md
  - ../methods/vla.md
sources:
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md
summary: "多模空间策展：2026.08.17–08.23 一周 16 篇 VLA 论文按架构、诊断、训练、长程与医疗六轴索引；16/16 独立 canonical 节点。"
---

# 一周 VLA 研究趋势（2026.08.17–08.23 · 第一篇）

> **本页定位**：为 [多模空间公众号盘点](https://mp.weixin.qq.com/s/E_JW7JL5g0p4-q5S3MoQUw) 提供横切面索引；**16/16 各有独立 `paper-*` 详情节点**。

## 一句话观点

**本周 VLA 主线：在不大改 VLM 骨干的前提下，用测试时干预（注意力 steering / TTA / 自评）、训练后 RL（EXPO-FT、Prism-GRPO、HAF）、以及 agent/分层记忆（TGL、BATON、SparkVLA、τ₀）把「能做一次」推进到「能安全、长程、可诊断地跑」。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| TTA | Test-Time Augmentation / Adaptation | 测试时增强或适配 |
| GRPO | Group Relative Policy Optimization | 组相对策略优化 |
| TTC | Test-Time Computation | 测试时额外推理计算 |

## 节点索引（16/16）

### 架构模块

- [Geo-VLA](../entities/paper-geo-vla.md) — arXiv:2608.21440
- [Inference-Time Attention Steering](../entities/paper-inference-time-attention-steering-vla-driving.md) — arXiv:2608.17095
- [NebulaVLA](../entities/paper-nebulavla.md) — arXiv:2608.16503
- [FabriMAE](../entities/paper-fabrimae-vla-self-eval.md) — arXiv:2608.16697

### 分析诊断 · 安全评测

- [Reuse Before You Retrieve](../entities/paper-reuse-before-you-retrieve-tta-vla.md) — arXiv:2608.17484
- [MANIGUARD](../entities/paper-maniguard.md) — arXiv:2608.17386
- [LIBERO-VIFO](../entities/paper-libero-vifo.md) — arXiv:2608.17600

### 性能 · 训练范式

- [SAFE-Pruner](../entities/paper-safe-pruner.md) — arXiv:2605.29662
- [Teach and Grow (TGL)](../entities/paper-rcl-2608-17209-teach-and-grow-an-agent-centered-architecture-fo.md) — arXiv:2608.17209
- [HAF](../entities/paper-haf-humanoid-vla-adaptation.md) — arXiv:2608.16837
- [EXPO-FT](../entities/paper-expo-ft.md) — arXiv:2605.25477
- [Prism-GRPO](../entities/paper-prism-grpo.md) — arXiv:2608.17423

### 长程 · Agent · 医疗

- [SparkVLA](../entities/paper-sparkvla.md) — arXiv:2608.16172
- [τ₀-VLA](../entities/paper-tau0-vla.md) — arXiv:2608.16885
- [BATON](../entities/paper-baton-long-horizon-manipulation.md) — arXiv:2608.16889
- [US-VLA](../entities/paper-us-vla-ultrasound.md) — arXiv:2608.16074

## 关联页面

- [VLA 方法页](../methods/vla.md)
- [Manipulation 任务页](../tasks/manipulation.md)
- [Geo-VLA](../entities/paper-geo-vla.md) — 本周智驾几何内化代表

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-17_part1.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-17_part1.md)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [Universal Post-Training for Robotics](../concepts/universal-post-training-robotics.md) — EXPO-FT 谱系
