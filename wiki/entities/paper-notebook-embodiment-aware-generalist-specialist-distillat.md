---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
arxiv: "2602.02960"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_embodiment-aware-generalist-specialist-distillat.md
summary: "EAGLE 把\"跨本体人形 WBC\"建成一个迭代的\"泛化—专家\"蒸馏循环：先在一个池子里同时训练多种本体的泛化策略；再为每个本体派生一个专家做精修；最后把各专家的新技能通过 DAgger 蒸馏回泛化策略，反复循环直至收敛——配合一套统一的高维指令接口（蹲、倾、底盘速度等同时支持），最终用一份策略驱动 H1 / G1 / N1 / T1 / Adam 等异构人形。"
---

# Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control

**Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

EAGLE 把"跨本体人形 WBC"建成一个迭代的"泛化—专家"蒸馏循环：先在一个池子里同时训练多种本体的泛化策略；再为每个本体派生一个专家做精修；最后把各专家的新技能通过 DAgger 蒸馏回泛化策略，反复循环直至收敛——配合一套统一的高维指令接口（蹲、倾、底盘速度等同时支持），最终用一份策略驱动 H1 / G1 / N1 / T1 / Adam 等异构人形。

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
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B.html> |
| arXiv | <https://arxiv.org/abs/2602.02960> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_embodiment-aware-generalist-specialist-distillat.md](../../sources/papers/humanoid_pnb_embodiment-aware-generalist-specialist-distillat.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B.html>
- 论文：<https://arxiv.org/abs/2602.02960>

## 推荐继续阅读

- [机器人论文阅读笔记：Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B/Embodiment-Aware_Generalist_Specialist_Distillation_for_Unified_Humanoid_Whole-B.html)
