---
type: entity
tags:
  - paper
  - vla
  - adaptation
  - rl
  - cmu
status: complete
updated: 2026-10-03
arxiv: "2608.11363"
related:
  - ../methods/vla.md
  - ../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md
sources:
  - ../../sources/papers/midas-minimal-data-vla-adaptation_arxiv_2608_11363.md
  - ../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md
summary: "MiDAS（arXiv:2608.11363）：少量专家示范先模仿启动，再自主试错并按结果筛选更有效动作，从易失败状态逐步稳定并泛化到示范外情况。"
---

# MiDAS（arXiv:2608.11363）

**MiDAS**（*Adaptation of Generalist Robot Policies with Minimal Data*，[arXiv:2608.11363](https://arxiv.org/abs/2608.11363)）收录于 [多模空间 · 一周 VLA 研究趋势简析（2026.08.10–08.16）第二篇](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) **训练范式** 段。

## 一句话定义

**少量专家示范先模仿启动，再自主试错并按结果筛选更有效动作，从易失败状态逐步稳定并泛化到示范外情况。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| VLM | Vision-Language Model | 视觉–语言多模态模型 |
| RL | Reinforcement Learning | 强化学习 |
| CoT | Chain-of-Thought | 链式推理 |

## 为什么重要

- 通才 VLA 零样本探索弱；MiDAS 用极少人工示范启动后续自主学习。
- 策展机构：卡内基梅隆大学（美）
- 开源结论：**待核实**（步骤 2.5，2026-10-01）。

## 流程总览

以下按本页已归纳的机制与资料绘制，表示模块或阅读路径关系。

```mermaid
flowchart TD
    N0["少量专家示范"]
    N1["模仿初始化"]
    N2["自主探索"]
    N3["结果筛选"]
    N4["策略更新"]
    N5["示范外状态评测"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N2
    N4 --> N5
```

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2608.11363](https://arxiv.org/abs/2608.11363) |
| **开源** | **待核实** |
| **文内评测** | LIBERO、RoboCasa；YAM 双臂实机 |

## 源码运行时序图

**不适用**（截至入库日未提供可运行官方代码入口，或仓库尚无可辨识训练/推理入口）。

## 实验与评测

- **文内口径：** LIBERO、RoboCasa；YAM 双臂实机
- **读法：** 本页为 [公众号策展](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md) 摘要；逐项指标以 **原文 PDF** 为准。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md) | 同批 15 篇横向索引；本文属 **训练范式** |
| [VLA 方法页](../methods/vla.md) | 单篇机制细节以原文为准 |

## 结论

**MiDAS 适合作为本期「训练范式」路线的快速索引页。**

1. 核心贡献：少量专家示范先模仿启动，再自主试错并按结果筛选更有效动作，从易失败状态逐步稳定并泛化到示范外情况。
2. 开源结论：**待核实** — 以项目页/仓库实际链接为准。
3. 横向对照见 [第二篇技术地图](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [一周 VLA 趋势技术地图（第二篇）](../overview/vla-weekly-trends-2026-08-10-part2-technology-map.md)

## 参考来源

- [wechat_duomo_vla_weekly_trends_2026-08-10_part2.md](../../sources/blogs/wechat_duomo_vla_weekly_trends_2026-08-10_part2.md)
- [arXiv:2608.11363](https://arxiv.org/abs/2608.11363)

## 推荐继续阅读

- [VLA 方法页](../methods/vla.md)
- [arXiv PDF](https://arxiv.org/pdf/2608.11363)
