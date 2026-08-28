---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-06-26
arxiv: "2505.19214"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_omni-perception-omnidirectional-collision-avoida.md
summary: "Omni-Perception：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后升格为完整索引实体。"
---

# Omni-Perception

**Omni-Perception: Omnidirectional Collision Avoidance for Legged Locomotion in Dynamic Environments** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页为 **计划索引实体**，深读笔记尚未撰写；笔记完成后应链向笔记站并深化归纳。

## 一句话定义

Omni-Perception 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/omni-perception-omnidirectional-collision-avoida` |
| arXiv | <https://arxiv.org/abs/2505.19214> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**这条目的两个关键词是「全向」和「动态环境」：标题把腿足避障从前向视野推到 360°、从静态地形推到会移动的障碍；但本页尚未深读，代价一侧完全空白。**

- 可确认信息：分类 05_Locomotion、arXiv 2505.19214、深读状态「待撰写」；机制线索仅有「腿足运动的全向碰撞规避」。
- 本页无法回答的恰是最要紧的几个问题：用什么传感配置实现全向覆盖、动态障碍下的反应延迟与成功率如何、是否真机验证。
- 因此当前只能作为占位与检索入口，挂在 [paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md) 之下；量化 benchmark 与实机指标待深读笔记补充。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_omni-perception-omnidirectional-collision-avoida.md](../../sources/papers/humanoid_pnb_omni-perception-omnidirectional-collision-avoida.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2505.19214>

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
