---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2508.01247"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_coordinated-humanoid-robot-locomotion-with-symme.md
summary: "Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy

**Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见上方「流程总览」（依据论文方法部分，非深读笔记）。

## 流程总览

```mermaid
flowchart TD
    O["观测历史 o[t-h:t]<br/>角速度、投影重力、速度指令、关节位置 / 速度、上一动作、步态相位"] --> E["History Encoder f_en<br/>输出潜变量 z"]
    E --> D["Observation Decoder f_de<br/>预测下一时刻观测 o(t+1)"]
    D -->|"自编码损失 L_AE (MSE)"| E
    E --> P["Policy Network f_π（actor）<br/>输入当前观测 o_t 与 z"]
    O2["当前观测 o_t"] --> P
    P --> A["动作 a_t<br/>腿、臂、腰关节目标"]
    H["高度图 H（特权信息）+ o_t"] --> V["Value Network V（critic）<br/>对称变换下不变"]
    V -->|"L_V = MSE，目标为 reward-to-go"| T["PPO 训练<br/>GAE 优势 + clip 目标 L_PPO"]
    A --> T
    T --> P
    S["对称等变设计：actor / encoder / decoder 等变<br/>critic 不变，用 ESCNN 参数共享的 Linear+ReLU 构建<br/>左右反射变换见 Table 1"] -.-> P
    C["课程学习（地形、指令、噪声）<br/>+ 域随机化"] -.-> T
```

图依据：论文 Method / Fig.1、Eq.(8)–(10)、Table 1

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/coordinated-humanoid-robot-locomotion-with-symme` |
| arXiv | <https://arxiv.org/abs/2508.01247> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2508.01247、分类 05_Locomotion、计划中的笔记文件夹；方法流程见「流程总览」，评测口径与实机结果在本页一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-05-locomotion.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_coordinated-humanoid-robot-locomotion-with-symme.md](../../sources/papers/humanoid_pnb_coordinated-humanoid-robot-locomotion-with-symme.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2508.01247>
- [arXiv PDF](https://arxiv.org/pdf/2508.01247)：已通读方法部分（实验与附录以略读为准）：Method（Problem Formulation、Model Architecture、Symmetry Equivariant Actor / Invariant Critic、Training Framework）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
