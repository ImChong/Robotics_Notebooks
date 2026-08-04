---

type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub, horizon-robotics]
status: stub
updated: 2026-07-01
arxiv: "2602.21723"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_lessmimic-long-horizon-humanoid-interaction-with.md
summary: "LessMimic 用距离场（Distance Field, DF）作为统一的交互表征——不依赖运动参考，单个策略就能在 0.4× ～ 1.6× 尺度变化下完成抓取、坐立、推拉、搬运，并支持最长 40 个连续技能的长时序组合。"
---

# LessMimic

**LessMimic: Long-Horizon Humanoid Interaction with Unified Distance Field Representations** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

LessMimic 用距离场（Distance Field, DF）作为统一的交互表征——不依赖运动参考，单个策略就能在 0.4× ～ 1.6× 尺度变化下完成抓取、坐立、推拉、搬运，并支持最长 40 个连续技能的长时序组合。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/LessMimic_Long-Horizon_Humanoid_Interaction_with_Unified_Distance_Field_Representations/LessMimic_Long-Horizon_Humanoid_Interaction_with_Unified_Distance_Field_Representations.html> |
| arXiv | <https://arxiv.org/abs/2602.21723> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**LessMimic 的取舍很清楚：放弃运动参考，换来一个统一的几何表征——用距离场把「抓、坐立、推拉、搬运」这些异质交互压进同一个策略里。**

- 真正起作用的机制是距离场（Distance Field）作为统一交互表征：不同物体、不同技能共享同一套输入描述，因而不必为每个任务准备参考轨迹。
- 两个可对照的能力口径是尺度鲁棒性（0.4×～1.6× 尺度变化下单策略仍可用）与长时序组合（最长 40 个连续技能）。
- 「不依赖运动参考」是它相对 mimic 系方法的核心区分点：省掉了参考数据这一环，代价与收益都要落在表征设计上。
- 边界：本页是索引级实体，消融、成功率与实机指标一律以深读笔记与论文 PDF 为准（见 [参考来源](#参考来源)），本页不构成可引用的评测证据。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_lessmimic-long-horizon-humanoid-interaction-with.md](../../sources/papers/humanoid_pnb_lessmimic-long-horizon-humanoid-interaction-with.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/LessMimic_Long-Horizon_Humanoid_Interaction_with_Unified_Distance_Field_Representations/LessMimic_Long-Horizon_Humanoid_Interaction_with_Unified_Distance_Field_Representations.html>
- 论文：<https://arxiv.org/abs/2602.21723>

## 推荐继续阅读

- [机器人论文阅读笔记：LessMimic](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/LessMimic_Long-Horizon_Humanoid_Interaction_with_Unified_Distance_Field_Representations/LessMimic_Long-Horizon_Humanoid_Interaction_with_Unified_Distance_Field_Representations.html)
