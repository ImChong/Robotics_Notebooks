---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-08-05
arxiv: "2505.07294"
venue: "2025.05"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ./paper-fddc.md
sources:
  - ../../sources/papers/humanoid_pnb_hub.md
  - ../../sources/papers/fddc_arxiv_2608_00500.md
summary: "HuB：列入 Paper Notebooks progress 待深读清单；深读笔记完成后升格为完整索引实体。"
---

# HuB

**HuB: Learning Extreme Humanoid Balance** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页为 **计划索引实体**，深读笔记尚未撰写；笔记完成后应链向笔记站并深化归纳。

## 一句话定义

HuB 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/HuB__Learning_Extreme_Humanoid_Balance` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**HuB（Learning Extreme Humanoid Balance）在本库被挂在 loco-manipulation/WBC 分类下，作为「极限平衡」方向的占位节点存在，实质内容尚未写入。**

- 唯一的方向线索来自标题与分类：极限平衡问题，归入 **04_Loco-Manipulation_and_WBC**，与全身控制主线同族。
- 「极限平衡」如何定义、用什么任务与指标衡量，本页均无依据，不应据此下效果判断。
- 与同分类多数条目不同，本页的深读进度追踪指向 **progress.json** 而非 PROGRESS.md，后续 ingest 时注意来源差异。
- 计划文件夹路径已定，笔记完成后应把本页升格为完整索引实体并重写本节。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- [FDDC](./paper-fddc.md) — 可部署动态 CoM 单腿平衡；文中以 HuB 为「特权平衡→蒸馏上真机」对照（arXiv:2608.00500）

## 参考来源

- [humanoid_pnb_hub.md](../../sources/papers/humanoid_pnb_hub.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [FDDC（arXiv:2608.00500）](../../sources/papers/fddc_arxiv_2608_00500.md) — 极限/单腿平衡对照与基准

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- [FDDC 论文实体](./paper-fddc.md)
