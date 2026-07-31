---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-07-28
venue: curated
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./paper-notebook-learning-to-ball.md
sources:
  - ../../sources/papers/humanoid_pnb_composite-motion-learning-with-task-control.md
summary: "Composite Motion Learning with Task Control：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后升格为完整索引实体。"
---

# Composite Motion Learning with Task Control

**Composite Motion Learning with Task Control** 已列入 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页为 **计划索引实体**，深读笔记尚未撰写；笔记完成后应链向笔记站并深化归纳。

## 一句话定义

Composite Motion Learning with Task Control 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks **progress 待深读** 清单，便于与全库 [人形论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 在深读笔记完成前，本页作为 **占位子节点**，避免知识图谱缺失该论文实体。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/13_Physics-Based_Animation/composite-motion-learning-with-task-control` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 Composite Motion Learning 的占位实体，但它在本库不是孤立节点：它被标注为[Learning to Ball](./paper-notebook-learning-to-ball.md) 官方实现的方法底座，这是目前页内最有价值的一条信息。**

- 可确认的只有索引层信息——分类 13_Physics-Based_Animation、策展来源与计划中的笔记文件夹；本条目无 arXiv 字段，标题指向「组合式动作学习 + 任务控制」，具体的组合与任务奖励机制页内未展开。
- 真正的定位线索来自下游：[Learning to Ball](./paper-notebook-learning-to-ball.md) 的官方实现建立在本方法与 ICCGAN 之上，说明它更像可复用的动作合成基座，而不是某个单点任务的解法。
- 适用边界：属于物理动画一侧（见[分类父节点](../overview/paper-notebook-category-13-physics-based-animation.md)），不要直接当作真机控制方案引用。
- 风险提示：量化 benchmark、消融与实机指标本页均缺失，深读状态以 PROGRESS.md 为准。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 后续篮球长程组合应用：[Learning to Ball](./paper-notebook-learning-to-ball.md) — 官方实现基于本方法 + ICCGAN

## 参考来源

- [humanoid_pnb_composite-motion-learning-with-task-control.md](../../sources/papers/humanoid_pnb_composite-motion-learning-with-task-control.md)
- [Humanoid Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
