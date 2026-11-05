---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2509.21690"
venue: "2025.09"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_towards-versatile-humanoid-table-tennis.md
summary: "Towards Versatile Humanoid Table Tennis：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# Towards Versatile Humanoid Table Tennis

**Towards Versatile Humanoid Table Tennis: Unified Reinforcement Learning with Prediction Augmentation** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Towards Versatile Humanoid Table Tennis 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」，不含实验结论。

## 流程总览

```mermaid
flowchart TD
    subgraph SIM["仿真训练（LeggedLab / IsaacLab，4096 并行环境）"]
        S1["发球：随机初速的球 + 空气阻力模型<br/>域随机化与感知噪声/延迟"] --> S2["球位置观测历史（5 帧）"]
        S2 --> P["Predictor：预测击球后落点最高点 p~_ball<br/>RMSE 回归损失，在线训练"]
        S1 --> PH["仿真物理预测 p^_ball（真值）"]
        PH -->|"监督"| P
        P --> SH["目标基座平移 Δp~_base,xy = p~_ball,xy − p_base,xy"]
        P --> A["Actor（MLP）"]
        SH --> A
        A --> ACT["动作 a_t ∈ R21<br/>q_target = q_nom + α·a_t"]
        PH --> RW["预测式稠密奖励<br/>Reaching 奖励 + Returning 奖励"]
        RW --> C["Critic（特权信息：物理预测值等）"]
        C --> PPO["PPO 更新 Actor / Critic"]
    end
    ACT --> PD["低层关节 PD 控制器"]
    PD --> HW["Booster T1 真机击球<br/>动捕提供球与躯干位姿"]
```

图依据：论文 §III（Fig.2 / Fig.3 / Table I）、§IV-A。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/Towards_Versatile_Humanoid_Table_Tennis__Unified_Reinforcement_Learning_with_Pre` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是待读条目，登记的是这篇 04_Loco-Manipulation_and_WBC 论文的阅读进度，而不是对其方法的判断。**

- 页面可确证的信息只有分类归属与「待深读」状态；标题层面的线索是「统一强化学习 + 预测增强」，具体机制页面未展开。
- 「实验与评测」明确留空，任何回球成功率、消融或实机指标都需回到原文。
- 深读笔记完成后本页应补成完整摘要并链向笔记站。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_towards-versatile-humanoid-table-tennis.md](../../sources/papers/humanoid_pnb_towards-versatile-humanoid-table-tennis.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2509.21690)：已通读方法部分（实验与附录以略读为准）：§I 引言、§II 问题建模、§III 方法、§IV-A/B 训练与硬件设置

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
