---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2505.24198"
venue: "2025.05"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-gentle-humanoid-locomotion-and-end-effe.md
summary: "Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control

**Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Learning Gentle Humanoid Locomotion and End-Effector Stabilization Control 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    OBS["共享观测：5 步本体历史 + 目标（根速度、朝向、步态、末端执行器 EE 指令）"]
    UA["上身 Agent（高频，14 DoF 手臂）"]
    LA["下身 Agent（低频，13 DoF 腿 + 腰）"]
    UR["上身奖励组：EE 位置跟踪 / EE 稳定"]
    LR["下身奖励组：步态跟踪 / 根速度跟踪"]
    ROBOT["27 维关节位置目标 → 人形机器人"]
    OBS --> UA
    OBS --> LA
    UA -->|"高频动作"| ROBOT
    LA -->|"低频动作"| ROBOT
    UR -.->|"PPO，独立 actor / critic"| UA
    LR -.->|"PPO，独立 actor / critic"| LA
```

图依据：论文 §3.1 / §3.2、Fig.2（SoFTA：Slow-Fast Two-Agent）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/Learning_Gentle_Humanoid_Locomotion_and_End-Effector_Stabilization_Control` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前的定位是一条待读记录，而不是一份可引用的论文归纳：它的作用是让你知道有这么一篇待读，而不是已经读懂了它。**

- 该条目已进入 Paper Notebooks 的 progress 待深读清单，归在 04_Loco-Manipulation_and_WBC 分类下，可与总索引及分类页交叉检索。
- 深读笔记尚未撰写，本页因此除「流程总览」外没有机制描述，也没有量化 benchmark 或实机指标；引用时不要把它当作结论来源。
- 后续动作明确：笔记在计划文件夹落地后，本页应链向笔记站并补成完整摘要。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-gentle-humanoid-locomotion-and-end-effe.md](../../sources/papers/humanoid_pnb_learning-gentle-humanoid-locomotion-and-end-effe.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2505.24198)：已通读方法部分（实验与附录以略读为准）：§3（问题设定、SoFTA 框架与分组奖励）、§4（实验设定与结果，略读）


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
