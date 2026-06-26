---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-26
arxiv: "2512.08500"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_mimic2dm.md
summary: "不再依赖 off-the-shelf 的 3D 重建，直接用从野生视频抽出的 2D 关键点轨迹 + 重投影误差，把\"物理仿真中的角色控制器\"从训练到生成端到端做穿；多视角聚合后还能涨成 3D 追踪能力 —— 给数据匮乏 / 物理可信度难保证的复杂动作（舞蹈、球类、动物步态）一条便宜得多的路。"
---

# Mimic2DM

**Mimic2DM: Generating and Mimicking 2D Motions for 3D Character Control** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：13_Physics-Based_Animation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

不再依赖 off-the-shelf 的 3D 重建，直接用从野生视频抽出的 2D 关键点轨迹 + 重投影误差，把"物理仿真中的角色控制器"从训练到生成端到端做穿；多视角聚合后还能涨成 3D 追踪能力 —— 给数据匮乏 / 物理可信度难保证的复杂动作（舞蹈、球类、动物步态）一条便宜得多的路。

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
| 分类 | 13_Physics-Based_Animation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/Mimic2DM__Generating_and_Mimicking_2D_Motions_for_3D_Character_Control/Mimic2DM__Generating_and_Mimicking_2D_Motions_for_3D_Character_Control.html> |
| arXiv | <https://arxiv.org/abs/2512.08500> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_mimic2dm.md](../../sources/papers/humanoid_pnb_mimic2dm.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/Mimic2DM__Generating_and_Mimicking_2D_Motions_for_3D_Character_Control/Mimic2DM__Generating_and_Mimicking_2D_Motions_for_3D_Character_Control.html>
- 论文：<https://arxiv.org/abs/2512.08500>

## 推荐继续阅读

- [机器人论文阅读笔记：Mimic2DM](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/Mimic2DM__Generating_and_Mimicking_2D_Motions_for_3D_Character_Control/Mimic2DM__Generating_and_Mimicking_2D_Motions_for_3D_Character_Control.html)
