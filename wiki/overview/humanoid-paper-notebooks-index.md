---
type: overview
tags: [humanoid-paper-notebooks, paper-index, overview]
status: complete
updated: 2026-06-07
related:
  - ./paper-notebook-category-01-foundational-rl.md
  - ./paper-notebook-category-02-motion-retargeting.md
  - ./paper-notebook-category-03-high-impact-selection.md
  - ./paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ./paper-notebook-category-05-locomotion.md
  - ./paper-notebook-category-06-manipulation.md
  - ./paper-notebook-category-07-teleoperation.md
  - ./paper-notebook-category-08-navigation.md
  - ./paper-notebook-category-09-state-estimation.md
  - ./paper-notebook-category-10-sim-to-real.md
  - ./paper-notebook-category-11-simulation-benchmark.md
  - ./paper-notebook-category-12-hardware-design.md
  - ./paper-notebook-category-13-physics-based-animation.md
  - ./paper-notebook-category-14-human-motion.md
summary: "Humanoid Paper Notebooks 137+ 篇深读笔记在本库的分类父节点与 wiki 子节点总索引（共 252 篇）。"
---

# Humanoid Paper Notebooks 知识库索引

本页把 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html) 的 **14 类主页分类** 映射为本仓库 `wiki/overview/paper-notebook-category-*` **父节点**；每篇论文对应 **子节点**（已有深度 wiki 或 `wiki/entities/paper-notebook-*` 索引实体）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 分类父节点（与笔记主页面一致）

- [Foundational RL（基础强化学习）](./paper-notebook-category-01-foundational-rl.md) — `01_Foundational_RL`，15 篇
- [Motion Retargeting（运动重定向）](./paper-notebook-category-02-motion-retargeting.md) — `02_Motion_Retargeting`，4 篇
- [High Impact Selection（高影响力精选）](./paper-notebook-category-03-high-impact-selection.md) — `03_High_Impact_Selection`，23 篇
- [Loco-Manipulation and WBC（运动操作与全身控制）](./paper-notebook-category-04-loco-manipulation-and-wbc.md) — `04_Loco-Manipulation_and_WBC`，147 篇
- [Locomotion（行走运动）](./paper-notebook-category-05-locomotion.md) — `05_Locomotion`，9 篇
- [Manipulation（灵巧操作）](./paper-notebook-category-06-manipulation.md) — `06_Manipulation`，6 篇
- [Teleoperation（遥操作）](./paper-notebook-category-07-teleoperation.md) — `07_Teleoperation`，6 篇
- [Navigation（导航）](./paper-notebook-category-08-navigation.md) — `08_Navigation`，6 篇
- [State Estimation（状态估计）](./paper-notebook-category-09-state-estimation.md) — `09_State_Estimation`，6 篇
- [Sim-to-Real（仿真到现实）](./paper-notebook-category-10-sim-to-real.md) — `10_Sim-to-Real`，6 篇
- [Simulation Benchmark（仿真与基准）](./paper-notebook-category-11-simulation-benchmark.md) — `11_Simulation_Benchmark`，6 篇
- [Hardware Design（硬件设计）](./paper-notebook-category-12-hardware-design.md) — `12_Hardware_Design`，6 篇
- [Physics-Based Animation（物理动画）](./paper-notebook-category-13-physics-based-animation.md) — `13_Physics-Based_Animation`，6 篇
- [Human Motion（人体动作分析与生成）](./paper-notebook-category-14-human-motion.md) — `14_Human_Motion`，6 篇

## 维护说明

- 笔记 URL 与分类元数据：`schema/paper-notebook-index.json`、`schema/paper-notebook-categories.json`
- 论文 → wiki 完整映射：`schema/paper-notebook-wiki-full-map.yml`
- 向已有 wiki 页注入深读链接：`make paper-notebook-links`
- 补齐未映射论文的 sources/实体与分类树：`make paper-notebook-bootstrap`（含 progress.json 待深读条目）

## 与其他页面的关系

- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)
- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)
- [Ego 9 篇技术地图](./ego-9-papers-technology-map.md)

## 参考来源

- [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)
- [sources/sites/rl-sim2sim-demo-website.md](../../sources/sites/rl-sim2sim-demo-website.md)（姊妹演示站，非本索引范围）

## 推荐继续阅读

- [机器人论文阅读笔记总站](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)
- [人形 RL 身体系统栈](./humanoid-rl-motion-control-body-system-stack.md)
- [BFM 41 篇技术地图](./bfm-41-papers-technology-map.md)
