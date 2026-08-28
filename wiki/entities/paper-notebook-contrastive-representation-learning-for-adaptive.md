---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2509.12858"
related:
  - ../overview/paper-notebook-category-10-sim-to-real.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_contrastive-representation-learning-for-adaptive.md
summary: "主流人形 RL 行走面临两难选择——纯本体感知策略反应快但被动（只能\"踩到了再调\"），而依赖深度图/高程图的感知驱动策略主动但脆弱（深度噪声、外参漂移、视角遮挡都会让 sim-to-real 崩掉）。本文用对比学习把仿真侧的特权环境信息（地形高度、摩擦、质量、外力等）\"蒸馏\"到 actor 的隐状态里，同时引入一个自适应步态时钟让策略根据\"已感知到但实际看不见\"的地形主动调整步频，从而在不接任何外部感知模块的前提下兼具反应与主动性，全尺寸人形零样本通过 30 cm 台阶和 26.5° 斜坡。"
---

# Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion

**Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：10_Sim-to-Real）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

主流人形 RL 行走面临两难选择——纯本体感知策略反应快但被动（只能"踩到了再调"），而依赖深度图/高程图的感知驱动策略主动但脆弱（深度噪声、外参漂移、视角遮挡都会让 sim-to-real 崩掉）。本文用对比学习把仿真侧的特权环境信息（地形高度、摩擦、质量、外力等）"蒸馏"到 actor 的隐状态里，同时引入一个自适应步态时钟让策略根据"已感知到但实际看不见"的地形主动调整步频，从而在不接任何外部感知模块的前提下兼具反应与主动性，全尺寸人形零样本通过 30 cm 台阶和 26.5° 斜坡。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks 策展清单，便于与全库 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 深读笔记提供比摘要更贴近实现的阅读路径，适合作为后续 ingest 深化起点。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 10_Sim-to-Real |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion.html> |
| arXiv | <https://arxiv.org/abs/2509.12858> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**这篇的取舍很明确：不接任何外部感知模块，用特权信息蒸馏 + 自适应步态时钟把「主动性」塞进纯本体感知策略，从而绕开深度图/高程图路线的 sim-to-real 脆弱点。**

- 起作用的机制是两件事的组合：对比学习把仿真侧特权环境信息（地形高度、摩擦、质量、外力）蒸馏进 actor 隐状态；自适应步态时钟让策略按「已感知到但看不见」的地形主动调步频。缺任一件就退回「反应快但被动」。
- 关键结果是全尺寸人形**零样本**通过 30 cm 台阶与 26.5° 斜坡——它证明的是不用外感知也能拿到主动性，而不是性能上限。
- 适用边界与代价：换掉的是深度噪声、外参漂移、视角遮挡这些脆弱点，代价是策略只能靠本体信号与训练期特权信息去推断地形。
- 主要风险：特权信息只在仿真中可得，蒸馏质量直接决定隐状态是否可靠；本页为索引级摘要，消融与完整指标以深读笔记与论文 PDF 为准。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-10-sim-to-real](../overview/paper-notebook-category-10-sim-to-real.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_contrastive-representation-learning-for-adaptive.md](../../sources/papers/humanoid_pnb_contrastive-representation-learning-for-adaptive.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion.html>
- 论文：<https://arxiv.org/abs/2509.12858>

## 推荐继续阅读

- [机器人论文阅读笔记：Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/10_Sim-to-Real/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion/Contrastive_Representation_Learning_for_Adaptive_Humanoid_Locomotion.html)
