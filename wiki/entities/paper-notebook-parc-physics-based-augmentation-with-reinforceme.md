---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-07-13
venue: "2025 SIGGRAPH"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../overview/jason-peng-flexible-motion-skill-learning.md
sources:
  - ../../sources/papers/humanoid_pnb_parc-physics-based-augmentation-with-reinforceme.md
  - ../../sources/courses/jason_peng_synthetic_motion_humanoid_youtube.md
  - ../../sources/blogs/wechat_human_five_jason_peng_flexible_motion_skills.md
summary: "PARC：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后升格为完整索引实体。"
---

# PARC

**[PARC: Physics-based Augmentation with Reinforcement Learning for Character Controllers](https://michaelx.io/parc/index.html)** 已列入 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页为 **计划索引实体**，深读笔记尚未撰写；笔记完成后应链向笔记站并深化归纳。

[human five 对 Jason Peng 分享的归纳](../../sources/blogs/wechat_human_five_jason_peng_flexible_motion_skills.md) 与 [NUS 研讨会讲者视频](../../sources/courses/jason_peng_synthetic_motion_humanoid_youtube.md) 均概括其核心为 **生成器—跟踪器迭代数据增强**：14 分钟初始移动数据经多轮仿真反馈扩至 900+ 分钟，并涌现原数据集中不存在的攀爬策略；机制总览见 [灵活运动技能学习技术地图](../overview/jason-peng-flexible-motion-skill-learning.md)。

## 一句话定义

PARC 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
| 计划文件夹 | `papers/13_Physics-Based_Animation/parc-physics-based-augmentation-with-reinforceme` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_parc-physics-based-augmentation-with-reinforceme.md](../../sources/papers/humanoid_pnb_parc-physics-based-augmentation-with-reinforceme.md)
- [jason_peng_synthetic_motion_humanoid_youtube.md](../../sources/courses/jason_peng_synthetic_motion_humanoid_youtube.md) — Peng 讲者原声演示 PARC 数字与 G1 案例（<https://www.youtube.com/watch?v=2looxieN53o>）
- [Humanoid Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
