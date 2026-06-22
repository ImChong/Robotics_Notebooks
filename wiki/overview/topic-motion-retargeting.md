---
type: overview
tags: [topic, topic-motion-retargeting, motion-retargeting, mocap, humanoid]
status: complete
updated: 2026-06-17
summary: "动作重定向专题汇总：把人体/动物参考动作映射到人形与异构机器人骨架，衔接 MoCap、IK/优化重定向、AMP 先验与 WBT 训练数据的全链路导读。"
---

# 动作重定向（专题汇总）

> **图谱专题视图**：本页是知识图谱「🤸 动作重定向」专题的统一入口；在 [图谱专题视图](../../docs/graph.html?topic=motion-retargeting) 筛选时，本节点为汇总锚点。

## 一句话定义

**动作重定向（Motion Retargeting）** 把来自人体动捕、视频或动画的参考运动，转换成目标机器人可执行、且仍保留运动语义与风格的关节/末端轨迹。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Retargeting | Motion Retargeting | 跨骨架动作映射 |
| MoCap | Motion Capture | 最常见参考动作来源 |
| IK | Inverse Kinematics | 任务空间约束下的关节解算 |
| AMP | Adversarial Motion Prior | 可与重定向数据组合的风格先验 |
| WBT | Whole-Body Tracking | 重定向产物进入跟踪训练的主下游 |

## 为什么重要

- **模仿学习与 WBT** 几乎都依赖「像人的参考动作」；不重定向就无法直接喂给 RL/BC。
- **跨平台复用**：一次 MoCap 录制，可映射到不同人形/四足形态。
- **失败常发生在重定向**：比例差、接触不一致、关节超限会让后续训练「看起来在学、实际上在追不可行轨迹」。

## 本专题覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 概念 | 重定向解决什么、有哪些方法族 | [Motion Retargeting](../concepts/motion-retargeting.md) |
| 流水线 | 采集 → 清洗 → 重定向 → 训练输入 | [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md) |
| 选型 | GMR / NMR / Reactor 等路线差异 | [GMR vs NMR vs Reactor](../comparisons/gmr-vs-nmr-vs-reactor.md) |
| 数据 | 参考运动数据集与重定向就绪度 | [人形参考运动数据集选型](../comparisons/humanoid-reference-motion-datasets.md) |
| 下游 | 重定向后如何进入 WBT / AMP | [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md) |

## 与其他专题的关系

- **[WBT](./topic-wbt.md)**：消费重定向轨迹做全身跟踪策略。
- **[跨具身](./topic-cross-embodiment.md)**：重定向是跨形态迁移的前置步骤。
- **[IL/RL](./topic-learning.md)**：重定向数据常作为 BC 示范或 AMP 风格约束。
- **[训练数据](./topic-data-pipeline.md)**：重定向是训练数据管线的第③段，承接质量评估、产出策略输入。

## 关联页面

- [Character Animation vs Robotics](../concepts/character-animation-vs-robotics.md)
- [Humanoid AMP 运动先验综述](./humanoid-amp-motion-prior-survey.md)
- [人形 RL 运动控制身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)

## 参考来源

- 本库归纳自 [Motion Retargeting](../concepts/motion-retargeting.md)、[Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md)
- 图谱专题定义：[docs/topic-filters.js](../../docs/topic-filters.js)（`motion-retargeting` 命中规则）
