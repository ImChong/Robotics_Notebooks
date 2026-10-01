---
type: entity
tags:
  - paper
  - vla
  - diffusion
  - pretraining
  - rl
status: complete
updated: 2026-10-01
arxiv: "2605.12236"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md
sources:
  - ../../sources/papers/tmrl-diffusion-timestep-pretraining_arxiv_2605_12236.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md
summary: "TMRL（arXiv:2605.12236）：BC 预训练后 RL 微调时，对输入加噪扩展可采样动作，微调阶段再调节探索幅度，少数据适应复杂真机操作。"
---

# TMRL（arXiv:2605.12236）

**TMRL**（*TMRL: Diffusion Timestep-Modulated Pretraining Enables Exploration for Efficient Policy Finetuning*，[arXiv:2605.12236](https://arxiv.org/abs/2605.12236)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第二篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) **训练范式** 段。

## 一句话定义

**BC 预训练后 RL 微调时，对输入加噪扩展可采样动作，微调阶段再调节探索幅度，少数据适应复杂真机操作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| CoT | Chain-of-Thought | 链式推理 |

## 为什么重要

- BC 动作分布过窄限制 RL 探索；TMRL 用扩散步调制预训练拓宽分布。
- 策展机构：华盛顿大学（美）；亚马逊 FAR（美）
- 开源结论：**待核实**（步骤 2.5，2026-10-01）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2605.12236](https://arxiv.org/abs/2605.12236) |
| **开源** | **待核实** |
| **文内评测** | OG-Bench、LIBERO（IsaacLab）；WidowX 250、Franka Panda 实机 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** OG-Bench、LIBERO（IsaacLab）；WidowX 250、Franka Panda 实机
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md) | 同批 15 篇横向索引；本文属 **训练范式** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**TMRL 适合作为本期「训练范式」路线的快速索引页。**

1. 核心贡献：BC 预训练后 RL 微调时，对输入加噪扩展可采样动作，微调阶段再调节探索幅度，少数据适应复杂真机操作。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第二篇）](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part2.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md)
- [arXiv:2605.12236](https://arxiv.org/abs/2605.12236)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2605.12236)
