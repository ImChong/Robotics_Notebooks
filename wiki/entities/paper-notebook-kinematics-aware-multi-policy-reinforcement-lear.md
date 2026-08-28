---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2511.21169"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_kinematics-aware-multi-policy-rl-for-force-capab.md
summary: "人形机器人有类人形态，在工业里潜力大。但现有 loco-manip 多聚焦灵巧操作，难满足高负载工业对「灵巧 + 主动力交互」的双重要求。本文提出一个基于 RL 的解耦三阶段训练流水线：上身策略、下身策略、delta 指令策略。为加速上身训练，设计一个启发式奖励——通过隐式嵌入前向运动学（FK）先验，让策略更快收敛且性能更优；为下身，开发一个基于力的课程学习策略，使机器人能主动施加并调节与环境的交互力。这样把「灵巧」与「主动发力」统一进同一框架，面向高负载工业搬运/推压等任务。"
---

# Kinematics-Aware Multi-Policy Reinforcement Learning for Force-Capable Humanoid Loco-Manipulation

**Kinematics-Aware Multi-Policy Reinforcement Learning for Force-Capable Humanoid Loco-Manipulation** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：04_Loco-Manipulation_and_WBC），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

人形机器人有类人形态，在工业里潜力大。但现有 loco-manip 多聚焦灵巧操作，难满足高负载工业对「灵巧 + 主动力交互」的双重要求。本文提出一个基于 RL 的解耦三阶段训练流水线：上身策略、下身策略、delta 指令策略。为加速上身训练，设计一个启发式奖励——通过隐式嵌入前向运动学（FK）先验，让策略更快收敛且性能更优；为下身，开发一个基于力的课程学习策略，使机器人能主动施加并调节与环境的交互力。这样把「灵巧」与「主动发力」统一进同一框架，面向高负载工业搬运/推压等任务。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Force-Capable | 有施力能力，能主动施加/调节接触力 |
| Multi-Policy | 多策略，上身/下身/delta 指令分设 |
| Delta-Command | 增量指令，对基础指令做微调 |
| FK Prior | Forward Kinematics 前向运动学先验 |
| Force Curriculum | 力课程，按力大小由易到难训练 |
| Loco-Manipulation | 移动操作 |

## 为什么重要

- **「主动发力」是工业 loco-manip 的关键短板**：纯位置/灵巧不够，需把力作为可控量；
- **嵌入运动学先验是高效训练的实用手段**：把结构知识写进奖励，胜过纯黑箱探索；
- **上下身解耦**契合人形结构，呼应 EGM 的上下身专家分设；
- **与 HAFO、FALCON、CHIP、HMC 等力/柔顺工作同向**，共同推进「会发力」的人形。

## 解决什么问题

**高负载工业**场景要求人形**既灵巧又能主动发力**，但现有 loco-manip： - 多只做**灵巧操作**，缺**主动力交互**； - 直接端到端学「又灵巧又发力」收敛慢、性能差。

论文要：一套能**同时**学到灵巧与主动发力、且**训练高效**的框架。

## 核心机制

1. **解耦三阶段多策略框架**：上身/下身/delta 指令分治，降低联合学习难度；
2. **FK 先验启发式奖励**：隐式嵌入前向运动学，加速上身收敛并提升性能；
3. **力课程学习**：让下身主动施加并调节交互力；
4. **面向工业高负载**：统一灵巧与主动发力。

方法拆解（深读笔记小节）：解耦三阶段多策略；上身：FK 先验的启发式奖励；下身：基于力的课程学习；面向工业高负载；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation.html> |
| arXiv | <https://arxiv.org/abs/2511.21169> |
| 作者 | Kaiyan Xiao、Zihan Xu、Cheng Zhe、Chengju Liu、Qijun Chen（同济大学等） |
| 发表 | 2025 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 结论

**这篇补的是工业 loco-manip 的「主动发力」短板：把力从被动结果变成可控量，靠的不是更大的网络，而是解耦、先验、课程这三件工程化的事。**

- 真正起作用的是两处结构性注入：上身用 **隐式嵌入 FK 先验的启发式奖励** 加速收敛并提升性能，下身用 **基于力的课程学习** 学会主动施加与调节交互力。
- 关键取舍是解耦三阶段（上身 / 下身 / delta 指令）：放弃端到端联合学习，换取可训练性——页面已点明直接端到端学「又灵巧又发力」收敛慢、性能差；代价是需要 delta 指令策略去缝合上下身。
- 适用边界是 **高负载工业搬运/推压** 类任务；只追求灵巧操作的场景用不上这套力课程，反而多付训练成本。
- 定位上与 HAFO、FALCON、CHIP、HMC 等力/柔顺路线同向，上下身分治则呼应 EGM 的专家分设，本文的差异点在于把运动学先验写进奖励而非留给黑箱探索。
- 本页为深读笔记的索引级摘要，量化 benchmark 与消融以笔记和论文 PDF 为准。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_kinematics-aware-multi-policy-rl-for-force-capab.md](../../sources/papers/humanoid_pnb_kinematics-aware-multi-policy-rl-for-force-capab.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation.html>
- 论文：<https://arxiv.org/abs/2511.21169>

## 推荐继续阅读

- [机器人论文阅读笔记：Kinematics-Aware Multi-Policy Reinforcement Learning for Force-Capable Humanoid Loco-Manipulation](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/04_Loco-Manipulation_and_WBC/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation/Kinematics-Aware_Multi-Policy_RL_for_Force-Capable_Humanoid_Loco-Manipulation.html)
