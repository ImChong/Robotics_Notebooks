---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-08-15
arxiv: "2506.12769"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./paper-gentrack.md
sources:
  - ../../sources/papers/humanoid_pnb_rl-from-physical-feedback-aligning-large-motion.md
summary: "RL from Physical Feedback：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后升格为完整索引实体。"
---

# RL from Physical Feedback

**RL from Physical Feedback: Aligning Large Motion Models with Humanoid Control** 已列入 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页为 **计划索引实体**，深读笔记尚未撰写；笔记完成后应链向笔记站并深化归纳。

## 一句话定义

RL from Physical Feedback 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
| 计划文件夹 | `papers/13_Physics-Based_Animation/rl-from-physical-feedback-aligning-large-motion` |
| arXiv | <https://arxiv.org/abs/2506.12769> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是「把大运动模型对齐到人形控制」这一议题的占位子节点：归属已确定（13_Physics-Based_Animation），但对齐信号如何构造、代价多大，页面尚未给出答案。**

- 页面上唯一确定的信息是归类与计划路径：分类 13_Physics-Based_Animation、计划文件夹 `papers/13_Physics-Based_Animation/rl-from-physical-feedback-aligning-large-motion`，深读状态仍为「待撰写」。
- 适用边界：可用于分类检索与交叉链接，不可当作方法或指标依据——本页未给出物理反馈的具体形式与任何量化结果。
- 主要风险是被误当成已消化的笔记：本页「实验与评测」已明确 benchmark、消融与实机指标待补，引用前应回到 PROGRESS.md 与 <https://arxiv.org/abs/2506.12769>。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 双向在线对照：[GenTrack](./paper-gentrack.md) 把 RLPF 式物理反馈做成生成器–跟踪器共训，并拿冻结 tracker FlowGRPO 当单向对照

## 参考来源

- [humanoid_pnb_rl-from-physical-feedback-aligning-large-motion.md](../../sources/papers/humanoid_pnb_rl-from-physical-feedback-aligning-large-motion.md)
- [Humanoid Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2506.12769>

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
