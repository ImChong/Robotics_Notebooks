---
type: overview
tags: [hub, hub-motion-retargeting, motion-retargeting, mocap, humanoid]
status: complete
updated: 2026-09-04
summary: "动作重定向知识链汇总：把人体/动物参考动作映射到人形与异构机器人骨架，衔接 MoCap、IK/优化重定向、动力学精炼（SBTO/DSMS/KDMR/SPARK）、AMP 先验与 WBT 训练数据的全链路导读；含人体→非人形腿式（X-Morph）扩展。"
---

# 动作重定向（知识链汇总）

> **知识链汇总**：本页是相关概念/方法的统一入口；对应策展纵深见图谱 [路线视图](../../docs/graph.html?depth=motion-retargeting) 与 [路线页](../../roadmap/depth-motion-retargeting.md)。

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

## 本知识链覆盖什么

| 层次 | 典型问题 | 站内入口 |
|------|----------|----------|
| 概念 | 重定向解决什么、有哪些方法族 | [Motion Retargeting](../concepts/motion-retargeting.md) |
| 流水线 | 采集 → 清洗 → 重定向 → 训练输入 | [Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md) |
| 选型 | GMR / NMR / Reactor 等路线差异 | [GMR vs NMR vs Reactor](../comparisons/gmr-vs-nmr-vs-reactor.md) |
| 动力学精炼 | 运动学参考 → 全身动力学可行 | [DynaRetarget / SBTO](../methods/dynaretarget-sbto-motion-retargeting.md)、[DSMS / Shooting for Contact](../entities/paper-shooting-for-contact.md)、[KDMR](../entities/paper-kdmr.md)、[SPARK](../entities/paper-spark-skeleton-aligned-retargeting.md) |
| 表面对应 | 不手写关键点、接触跟点走 | [UMR](../entities/paper-umr-unified-motion-retargeting.md)（点云对应；待发布） |
| 接触精炼（开源工具） | SOMA 人体 → 多人形 + 接触/自碰 | [CoRe v0.1.0](../entities/core-retarget.md)（论文 [CoRe](../entities/paper-core.md) / [RMR](../entities/paper-rmr.md)） |
| 数据 | 参考运动数据集与重定向就绪度 | [人形参考运动数据集选型](../comparisons/humanoid-reference-motion-datasets.md) |
| 下游 | 重定向后如何进入 WBT / AMP | [Whole-Body Tracking Pipeline](../concepts/whole-body-tracking-pipeline.md) |

## 与其他知识链的关系

- **[WBT](./hub-wbt.md)**：消费重定向轨迹做全身跟踪策略。
- **[跨具身](./hub-cross-embodiment.md)**：重定向是跨形态迁移的前置步骤。
- **[IL/RL](./hub-learning.md)**：重定向数据常作为 BC 示范或 AMP 风格约束。
- **[训练数据](./hub-data-pipeline.md)**：重定向是训练数据管线的第③段，承接质量评估、产出策略输入。

## 关联页面

- [Character Animation vs Robotics](../concepts/character-animation-vs-robotics.md)
- [Humanoid AMP 运动先验综述](./humanoid-amp-motion-prior-survey.md)
- [人形 RL 运动控制身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)
- [X-Morph](../entities/paper-xmorph.md) — 人体运动→非人形腿式（Go2/六足/带臂四足）重定向+校正+跟踪蒸馏
- [Shooting for Contact / DSMS](../entities/paper-shooting-for-contact.md) — 接触隐式多重打靶动力学重定向（Caltech/DePaul；G1 爬行与跳转）
- [KDMR](../entities/paper-kdmr.md) — GRF 锚定多接触全身 TO（Georgia Tech；BeyondMimic 下游）
- [SPARK（骨架对齐重定向）](../entities/paper-spark-skeleton-aligned-retargeting.md) — URDF 校准 + 渐进 KDTO（UW–Madison / Berkeley / SII）
- [UMR（学习点云对应）](../entities/paper-umr-unified-motion-retargeting.md) — 稠密表面对应替代手工关键点（HKUST-GZ / Noitom 等；arXiv:2609.02134）
- [CoRe v0.1.0](../entities/core-retarget.md) — Kimodo/GEM-X → 11 机接触精炼（高丽大学；Humanoids/IROS 2025）

## 参考来源

- 本库归纳自 [Motion Retargeting](../concepts/motion-retargeting.md)、[Motion Retargeting Pipeline](../concepts/motion-retargeting-pipeline.md)
- 知识链定义：[docs/depth-filters.js](../../docs/depth-filters.js)（`motion-retargeting` 命中规则）
- 上游原始资料（本链概念页共同的 ingest 来源）：[Make Tracking Easy：神经运动重定向](../../sources/papers/neural_motion_retargeting_nmr.md)、[ReActor：物理感知运动重定向](../../sources/papers/reactor_rl_physics_aware_motion_retargeting.md)、[AMP（2021）](../../sources/papers/amp.md)
