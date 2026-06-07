---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-06-07
arxiv: "2509.13780"
related:
  - ../overview/paper-notebook-category-03-high-impact-selection.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_behavior-foundation-model-for-humanoid-robots.md
summary: "把各类 WBC 任务都看成「在合适目标下生成行为轨迹」，先用 AMASS 重定向 + 仿真里特权信息的 proxy 运动模仿策略在线产出大规模行为数据，再用 掩码在线蒸馏 + 条件 VAE（CVAE） 学到可跨速度指令、遥操作、参考动作等多种控制接口共享的生成式策略，并可用 残差学习在不大改网络的前提下快速学会新动作——在仿真与真机上都展示了对多种全身任务的泛化与可组合潜空间。"
---

# Behavior Foundation Model for Humanoid Robots

**Behavior Foundation Model for Humanoid Robots** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：03_High_Impact_Selection）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

把各类 WBC 任务都看成「在合适目标下生成行为轨迹」，先用 AMASS 重定向 + 仿真里特权信息的 proxy 运动模仿策略在线产出大规模行为数据，再用 掩码在线蒸馏 + 条件 VAE（CVAE） 学到可跨速度指令、遥操作、参考动作等多种控制接口共享的生成式策略，并可用 残差学习在不大改网络的前提下快速学会新动作——在仿真与真机上都展示了对多种全身任务的泛化与可组合潜空间。

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
| 分类 | 03_High_Impact_Selection |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Behavior_Foundation_Model_for_Humanoid_Robots/Behavior_Foundation_Model_for_Humanoid_Robots.html> |
| arXiv | <https://arxiv.org/abs/2509.13780> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-03-high-impact-selection](../overview/paper-notebook-category-03-high-impact-selection.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_behavior-foundation-model-for-humanoid-robots.md](../../sources/papers/humanoid_pnb_behavior-foundation-model-for-humanoid-robots.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Behavior_Foundation_Model_for_Humanoid_Robots/Behavior_Foundation_Model_for_Humanoid_Robots.html>
- 论文：<https://arxiv.org/abs/2509.13780>

## 推荐继续阅读

- [机器人论文阅读笔记：Behavior Foundation Model for Humanoid Robots](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/03_High_Impact_Selection/Behavior_Foundation_Model_for_Humanoid_Robots/Behavior_Foundation_Model_for_Humanoid_Robots.html)
