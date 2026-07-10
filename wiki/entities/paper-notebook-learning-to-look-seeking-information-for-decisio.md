---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2410.18964"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-to-look.md
summary: "许多操作任务需要主动或交互式探索才能成功——智能体要主动寻找每一阶段所需的信息（如移动机器人的头去找操作相关信息；或多机器人里一个侦察机器人为另一个找信息）。本文把这类任务刻画为一种新问题：因子化上下文马尔可夫决策过程（factorized Contextual MDP），并提出 DISaM ——一个双策略解法：① 信息寻求策略（information-seeking）探索环境找到相关上下文信息；② 信息接收策略（information-receiving）利用上下文达成操作目标。这种因子化让两策略可分开训练（用接收策略给寻求策略提供奖励）。测试时，双智能体按操作策略对\"下一步最佳动作\"的不确定性来平衡探索与利用。在五个需信息寻求的操作任务（仿真 + 真机）上，DISaM 大幅优于已有方法。"
---

# Learning to Look

**Learning to Look: Seeking Information for Decision Making via Policy Factorization** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

许多操作任务需要主动或交互式探索才能成功——智能体要主动寻找每一阶段所需的信息（如移动机器人的头去找操作相关信息；或多机器人里一个侦察机器人为另一个找信息）。本文把这类任务刻画为一种新问题：因子化上下文马尔可夫决策过程（factorized Contextual MDP），并提出 DISaM ——一个双策略解法：① 信息寻求策略（information-seeking）探索环境找到相关上下文信息；② 信息接收策略（information-receiving）利用上下文达成操作目标。这种因子化让两策略可分开训练（用接收策略给寻求策略提供奖励）。测试时，双智能体按操作策略对"下一步最佳动作"的不确定性来平衡探索与利用。在五个需信息寻求的操作任务（仿真 + 真机）上，DISaM 大幅优于已有方法。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| DISaM | 双策略（信息寻求 + 接收）框架 |
| Factorized Contextual MDP | 因子化上下文 MDP |
| Information-Seeking | 信息寻求策略（探索） |
| Information-Receiving | 信息接收策略（利用） |
| Exploration/Exploitation | 探索 / 利用平衡 |
| Uncertainty | 操作策略对动作的不确定性 |

## 为什么重要

- **"找信息"与"用信息"解耦**是处理主动探索任务的优雅归纳偏置；
- **不确定性驱动的探索/利用平衡**是可迁移的测试时机制；
- 与 ViA、Learning to Look Around 的"主动视觉"互补（这里更偏决策层）；
- 对人形（转头/移动找信息）直接相关。

## 解决什么问题

许多操作要**先找信息再决策**： - 需**主动探索**（如转头找物体）； - 把"寻求信息"与"利用信息"混在一个策略里难学； - 测试时如何**平衡探索与利用**？

论文要：把任务**因子化**，分别学**寻求**与**接收**策略，并在测试时合理切换。

## 核心机制

1. **因子化上下文 MDP**：刻画"需主动找信息"的操作任务；
2. **DISaM 双策略**：信息寻求 + 信息接收，可分开训练；
3. **跨策略奖励**：用接收策略给寻求策略提供奖励；
4. **不确定性驱动探索/利用平衡**：五任务大幅优于基线。

方法拆解（深读笔记小节）：因子化上下文 MDP；DISaM 双策略；测试时按不确定性平衡探索/利用；结果；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization.html> |
| arXiv | <https://arxiv.org/abs/2410.18964> |
| 作者 | Shivin Dass、Jiaheng Hu、Ben Abbatematteo、Peter Stone、Roberto Martín-Martín（UT Austin） |
| 发表 | 2024 年 10 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-to-look.md](../../sources/papers/humanoid_pnb_learning-to-look.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization.html>
- 论文：<https://arxiv.org/abs/2410.18964>

## 推荐继续阅读

- [机器人论文阅读笔记：Learning to Look](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/06_Manipulation/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization/Learning_to_Look__Seeking_Information_for_Decision_Making_via_Policy_Factorization.html)
