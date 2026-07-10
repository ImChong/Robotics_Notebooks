---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2407.07788"
related:
  - ../overview/paper-notebook-category-11-simulation-benchmark.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_bigym.md
summary: "BiGym 是一个面向移动双手、演示驱动机器人操作的新基准与学习环境。它含 40 个家居任务，从简单到达到复杂厨房清洁。为准确反映真实表现，BiGym 为每个任务提供人类采集的演示，体现真实机器人轨迹的多样模态。它支持多种观测，包括本体感受与视觉输入（3 个相机视角的 RGB 与深度）。为验证可用性，作者在环境中充分评测了当前最佳的模仿学习与演示驱动强化学习算法，并讨论了未来机会。"
---

# BiGym

**BiGym: A Demo-Driven Mobile Bi-Manual Manipulation Benchmark** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：11_Simulation_Benchmark），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

BiGym 是一个面向移动双手、演示驱动机器人操作的新基准与学习环境。它含 40 个家居任务，从简单到达到复杂厨房清洁。为准确反映真实表现，BiGym 为每个任务提供人类采集的演示，体现真实机器人轨迹的多样模态。它支持多种观测，包括本体感受与视觉输入（3 个相机视角的 RGB 与深度）。为验证可用性，作者在环境中充分评测了当前最佳的模仿学习与演示驱动强化学习算法，并讨论了未来机会。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| Demo-Driven | 演示驱动，依赖人类演示 |
| Bi-Manual | 双手 |
| Mobile Manipulation | 移动操作 |
| Proprioception | 本体感受 |
| RGB-D | 彩色 + 深度（3 视角） |
| IL / Demo-Driven RL | 模仿学习 / 演示驱动强化学习 |

## 为什么重要

- **移动 + 双手**是人形家务的核心，BiGym 提供演示驱动评测床；
- **人类演示反映真实多样模态**对模仿学习评测很重要；
- **演示驱动 RL** 是 IL 与 RL 的折中，值得在人形任务上探索；
- 与 RoboCasa、ManiSkill-HAB 等家居基准互补。

## 解决什么问题

移动双手操作缺**演示驱动**的统一基准： - 现有基准少覆盖**移动 + 双手 + 家居多样任务**； - 缺**人类演示**反映真实轨迹多样性； - 缺统一环境评测 IL 与演示驱动 RL。

BiGym 要：一个**演示驱动、移动双手、家居多任务**的基准与环境。

## 核心机制

1. **演示驱动移动双手基准**：40 家居任务，覆盖难度谱；
2. **人类演示**：每任务配演示，反映真实轨迹多样性；
3. **多模态观测**：本体感受 + 3 视角 RGB/深度；
4. **系统评测**：IL 与演示驱动 RL 基线。

方法拆解（深读笔记小节）：40 家居任务 + 人类演示；多模态观测；系统评测 IL / 演示驱动 RL；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 11_Simulation_Benchmark |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/BiGym__A_Demo-Driven_Mobile_Bi-Manual_Manipulation_Benchmark/BiGym__A_Demo-Driven_Mobile_Bi-Manual_Manipulation_Benchmark.html> |
| arXiv | <https://arxiv.org/abs/2407.07788> |
| 作者 | Nikita Chernyadev、Nicholas Backshall、Xiao Ma、Yunfan Lu、Younggyo Seo、Stephen James（Dyson Robot Learning Lab） |
| 发表 | 2024 年 7 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../overview/paper-notebook-category-11-simulation-benchmark.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_bigym.md](../../sources/papers/humanoid_pnb_bigym.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/BiGym__A_Demo-Driven_Mobile_Bi-Manual_Manipulation_Benchmark/BiGym__A_Demo-Driven_Mobile_Bi-Manual_Manipulation_Benchmark.html>
- 论文：<https://arxiv.org/abs/2407.07788>

## 推荐继续阅读

- [机器人论文阅读笔记：BiGym](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/BiGym__A_Demo-Driven_Mobile_Bi-Manual_Manipulation_Benchmark/BiGym__A_Demo-Driven_Mobile_Bi-Manual_Manipulation_Benchmark.html)
