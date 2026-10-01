---
type: entity
tags:
  - paper
  - vla
  - humanoid
  - dual-arm
  - diagnostics
status: complete
updated: 2026-10-01
arxiv: "2608.11769"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md
sources:
  - ../../sources/papers/handprior-score-humanoid-dual-arm_arxiv_2608_11769.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md
summary: "HandPriorScore（arXiv:2608.11769）：诊断 VLA 在双手起始姿态不同下的「用手先验」：策略与姿态交互导致选错手；扩姿态覆盖与薄弱姿态补数据可提稳健性。"
---

# HandPriorScore（arXiv:2608.11769）

**HandPriorScore**（*Policy-Induced Hand Priors in Humanoid Dual-Arm Manipulation: Diagnosing and Mitigating Initial-Pose Dependence*，[arXiv:2608.11769](https://arxiv.org/abs/2608.11769)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第二篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) **末端操控** 段。

## 一句话定义

**诊断 VLA 在双手起始姿态不同下的「用手先验」：策略与姿态交互导致选错手；扩姿态覆盖与薄弱姿态补数据可提稳健性。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| CoT | Chain-of-Thought | 链式推理 |

## 为什么重要

- 同任务不同初始手位成功率差异大；HandPriorScore 量化 policy-induced hand prior。
- 策展机构：韩国科学技术研究院（KIST，韩）
- 开源结论：**待核实**（步骤 2.5，2026-10-01）。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.11769](https://arxiv.org/abs/2608.11769) |
| **开源** | **待核实** |
| **文内评测** | PickApple；Unitree G1-Dex3 采集 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** PickApple；Unitree G1-Dex3 采集
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md) | 同批 15 篇横向索引；本文属 **末端操控** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**HandPriorScore 适合作为本期「末端操控」路线的快速索引页。**

1. 核心贡献：诊断 VLA 在双手起始姿态不同下的「用手先验」：策略与姿态交互导致选错手；扩姿态覆盖与薄弱姿态补数据可提稳健性。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第二篇）](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part2.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md)
- [arXiv:2608.11769](https://arxiv.org/abs/2608.11769)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.11769)
