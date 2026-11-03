---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2505.22642"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_fasttd3-simple-fast-and-capable-reinforcement-le.md
summary: "FastTD3：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# FastTD3

**FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

FastTD3 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见「流程总览」（依据论文方法部分），不给实验结论。

## 流程总览

```mermaid
flowchart LR
    ENV["大规模并行仿真环境（IsaacLab / MuJoCo Playground / HumanoidBench）"]
    ACT["Actor MLP 512-256-128 + 混合高斯探索噪声（σ_max=0.4）"]
    BUF["回放缓冲区：N × num_envs，整体放在 GPU"]
    BATCH["大批量采样（batch 32768）"]
    CRITIC["分布式 Critic C51 + Clipped Double Q-learning；MLP 1024-512-256"]
    UPD["TD3 更新：低 UTD（每 128–4096 个并行环境步 2/4/8 次梯度更新）"]
    ACT -->|"动作"| ENV
    ENV -->|"转移数据"| BUF
    BUF --> BATCH --> CRITIC --> UPD
    UPD -->|"更新"| ACT
```

图依据：论文 §2 / §2.1 / §2.2（设计选择与实现细节）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/fasttd3-simple-fast-and-capable-reinforcement-le` |
| arXiv | <https://arxiv.org/abs/2505.22642> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页目前只能支撑一个结论：FastTD3 是一条已进入 05_Locomotion 深读队列、但尚未产出笔记的条目，它现在的价值只是可检索，而不是内容。**

- 页面能确证的信息仅限分类（05_Locomotion）、arXiv 编号与计划文件夹路径；算法改动、训练速度与人形控制上的实际收益，本页均无依据。
- 因此本页 **不可作为技术判断依据**：机制与指标应回到论文原文，或等 PROGRESS.md 追踪的深读笔记完成后再引用。
- 定位是 **待读条目**，方便你按分类找到它；笔记完成后应补成完整摘要，并重写本节为面向内容的结论。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_fasttd3-simple-fast-and-capable-reinforcement-le.md](../../sources/papers/humanoid_pnb_fasttd3-simple-fast-and-capable-reinforcement-le.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2505.22642>
- [arXiv PDF](https://arxiv.org/pdf/2505.22642)：已通读方法部分（实验与附录以略读为准）：§1 概述、§2.1 设计选择、§2.2 实现细节（实验与附录略读）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
