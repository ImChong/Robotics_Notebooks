---

type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub, horizon-robotics]
status: stub
updated: 2026-06-26
arxiv: "2509.22442"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-to-ball.md
summary: "把篮球里「运球 / 投篮 / 上篮 / 跑动 / 转身 / 捡球」这些差异极大、各自训好的子技能策略，用一套策略组合框架 + 一个高层「软路由器」拼起来——关键在于处理那些「目标说不清」的过渡段，让物理仿真角色能连贯打出 shoot-off-the-dribble（运球急停跳投）、catch-and-shoot（接球就投）、board-and-bang（抢下前场篮板立刻补篮）这类多阶段长程连招。"
---

# Learning to Ball

**Learning to Ball: Composing Policies for Long-Horizon Basketball Moves** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：13_Physics-Based_Animation）。本页为 **索引级实体**，链向深读笔记与原始论文；详细机制待从笔记消化后补充。

## 一句话定义

把篮球里「运球 / 投篮 / 上篮 / 跑动 / 转身 / 捡球」这些差异极大、各自训好的子技能策略，用一套策略组合框架 + 一个高层「软路由器」拼起来——关键在于处理那些「目标说不清」的过渡段，让物理仿真角色能连贯打出 shoot-off-the-dribble（运球急停跳投）、catch-and-shoot（接球就投）、board-and-bang（抢下前场篮板立刻补篮）这类多阶段长程连招。

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
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/Learning_to_Ball__Composing_Policies_for_Long-Horizon_Basketball_Moves/Learning_to_Ball__Composing_Policies_for_Long-Horizon_Basketball_Moves.html> |
| arXiv | <https://arxiv.org/abs/2509.22442> |

## 实验与评测

- 本页为 **策展索引级** 摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-to-ball.md](../../sources/papers/humanoid_pnb_learning-to-ball.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/Learning_to_Ball__Composing_Policies_for_Long-Horizon_Basketball_Moves/Learning_to_Ball__Composing_Policies_for_Long-Horizon_Basketball_Moves.html>
- 论文：<https://arxiv.org/abs/2509.22442>

## 推荐继续阅读

- [机器人论文阅读笔记：Learning to Ball](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/13_Physics-Based_Animation/Learning_to_Ball__Composing_Policies_for_Long-Horizon_Basketball_Moves/Learning_to_Ball__Composing_Policies_for_Long-Horizon_Basketball_Moves.html)
