---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2409.15610"
related:
  - ../overview/paper-notebook-category-14-human-motion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_full-order-sampling-based-mpc-for-torque-level-l.md
summary: "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing

**Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：14_Human_Motion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法流程见 [流程总览](#流程总览)，数据与指标请直接看原文。

## 一句话定义

Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置、原文入口与依据论文方法部分绘制的 [流程总览](#流程总览)，不给性能结论。

## 流程总览

```mermaid
flowchart TD
    INIT["初始化控制序列 u_0:H = 0"] --> LOOP
    subgraph LOOP["每个控制步 t：对 u_t:t+H 做 N 轮迭代（i 从 N 到 1）"]
        K["按式 7 取噪声核 Σ^i_t+h：轨迹级退火（随 i 减小）× 动作级退火（随时域 h 增大）"] --> SMP["采样 N_W 条扰动序列 u + W"]
        SMP --> ROL["全阶动力学并行 rollout，评估代价 J"]
        ROL --> SC["估计 score ∇log p_i（MPPI 式加权平均）"]
        SC --> UPD["更新 u_t:t+H"]
        UPD -->|"下一轮：噪声核变小"| K
    end
    LOOP --> APPLY["执行 u_t：力矩指令，50 Hz"]
    APPLY --> SHIFT["滚动时域：u_t+1:t+H+1 ← shift(u_t:t+H)"]
    SHIFT --> LOOP
```

图依据：论文 §III-B、C（Algorithm 1、式 4 至 7、Fig. 1）与附录 A（50 Hz 控制）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 14_Human_Motion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/14_Human_Motion/full-order-sampling-based-mpc-for-torque-level-l` |
| arXiv | <https://arxiv.org/abs/2409.15610> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页目前只能支撑一个结论：这是一条已进入 14_Human_Motion 深读队列、但尚未产出笔记的条目，它现在的价值只是可检索，而不是内容。**

- 页面能确证的信息仅限分类（14_Human_Motion）、arXiv 编号、计划文件夹路径与依据论文方法部分绘制的 [流程总览](#流程总览)；全阶采样式 MPC、扩散式退火与力矩级控制的实际效果，本页均无依据。
- 因此本页 **不可作为技术判断依据**：方法流程见 [流程总览](#流程总览)，指标应回到论文原文，或等 PROGRESS.md 追踪的深读笔记完成后再引用。
- 定位是 **待读条目**，方便你按分类找到它；笔记完成后应补成完整摘要，并重写本节为面向内容的结论。

## 与其他页面的关系

- 分类页：[paper-notebook-category-14-human-motion](../overview/paper-notebook-category-14-human-motion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_full-order-sampling-based-mpc-for-torque-level-l.md](../../sources/papers/humanoid_pnb_full-order-sampling-based-mpc-for-torque-level-l.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2409.15610>
- [arXiv PDF](https://arxiv.org/pdf/2409.15610)：已通读方法部分（实验与附录以略读为准）：§I 引言、§III-B/C 方法（扩散式退火与 DIAL-MPC 算法）、附录 A/B（软硬件与任务设置）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
