---
type: entity
tags:
  - paper
  - vla
  - representation
  - language-alignment
  - cmu
status: complete
updated: 2026-10-01
arxiv: "2608.10484"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md
sources:
  - ../../sources/papers/salt-vla-action-language-alignment_arxiv_2608_10484.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md
summary: "SALT（arXiv:2608.10484）：动作编码需同时可重建轨迹且让 VLM 猜回语言指令，避免 L1/L2 把语义不同但数值相近的动作混为一谈。"
---

# SALT（arXiv:2608.10484）

**SALT**（*Lost in Reconstruction: Aligning Action Representations with Language in Vision-Language-Action Models*，[arXiv:2608.10484](https://arxiv.org/abs/2608.10484)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第二篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) **架构模块** 段。

## 一句话定义

**动作编码需同时可重建轨迹且让 VLM 猜回语言指令，避免 L1/L2 把语义不同但数值相近的动作混为一谈。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| CoT | Chain-of-Thought | 链式推理 |

## 为什么重要

- 重建损失不等于语义对齐；SALT 显式绑动作方式与语言。
- 策展机构：卡内基梅隆大学（美）
- 开源结论：**待核实**（步骤 2.5，2026-10-01）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.10484](https://arxiv.org/abs/2608.10484) |
| **开源** | **待核实** |
| **文内评测** | SimplerEnv WidowX |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** SimplerEnv WidowX
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md) | 同批 15 篇横向索引；本文属 **架构模块** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**SALT 适合作为本期「架构模块」路线的快速索引页。**

1. 核心贡献：动作编码需同时可重建轨迹且让 VLM 猜回语言指令，避免 L1/L2 把语义不同但数值相近的动作混为一谈。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第二篇）](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part2.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md)
- [arXiv:2608.10484](https://arxiv.org/abs/2608.10484)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.10484)
