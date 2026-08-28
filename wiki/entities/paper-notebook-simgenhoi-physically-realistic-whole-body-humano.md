---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-07-22
arxiv: "2508.14120"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_simgenhoi-physically-realistic-whole-body-humano.md
summary: "SimGenHOI：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后升格为完整索引实体。"
---

# SimGenHOI

**SimGenHOI: Physically Realistic Whole-Body Humanoid-Object Interaction via Generative Modeling and RL** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页为 **计划索引实体**，深读笔记尚未撰写；笔记完成后应链向笔记站并深化归纳。

## 一句话定义

SimGenHOI 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 列入 Paper Notebooks **progress 待深读** 清单，便于与全库 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 及分类父节点交叉检索。
- 在深读笔记完成前，本页作为 **占位子节点**，避免知识图谱缺失该论文实体。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/13_Physics-Based_Animation/simgenhoi-physically-realistic-whole-body-humano` |
| arXiv | <https://arxiv.org/abs/2508.14120> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**SimGenHOI 在本库尚处占位阶段：本页只固定了它「生成式建模 + RL 做全身人–物交互」的坐标，并未给出可引用的机制或指标。**

- 可确认的仅索引层信息——arXiv 2508.14120、分类 13_Physics-Based_Animation 与计划中的笔记文件夹；标题点出的取舍是「生成动作」与「物理可行」两侧要同时兼顾，但具体如何配合页内未展开。
- 归类落在物理动画而非真机 locomotion 一侧，说明它与本库[分类父节点](../overview/paper-notebook-category-13-physics-based-animation.md)下的仿真/角色动画工作同栈；把它直接当作 Sim2Real 落地方案是误用。
- 主要风险是把占位页当结论：量化 benchmark、消融与实机指标全部缺失，深读进度以 PROGRESS.md 为准，判断请回到 arXiv 原文。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_simgenhoi-physically-realistic-whole-body-humano.md](../../sources/papers/humanoid_pnb_simgenhoi-physically-realistic-whole-body-humano.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2508.14120>

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
