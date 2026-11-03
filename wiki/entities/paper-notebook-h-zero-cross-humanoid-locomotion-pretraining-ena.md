---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2512.00971"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_h-zero-cross-humanoid-locomotion-pretraining-ena.md
summary: "H-Zero：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# H-Zero

**H-Zero: Cross-Humanoid Locomotion Pretraining Enables Few-shot Novel Embodiment Transfer** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

H-Zero 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」，指标与结论以论文为准。

## 流程总览

```mermaid
flowchart TD
  subgraph PRE["Cross-Embodiment Pretraining"]
    ROB[多款人形机器人模型 H1 G1 N1 GR1 PM01 等 Isaac Gym 多机器人仿真] --> UCS[统一控制语义 关节映射 M 加 运动学对齐 方向 s 与零位 b]
    ROB --> DR[扩展域随机化 刚体与关节动力学扰动范围加倍]
    UCS --> OBS[观测 加 embodiment descriptors 关节 刚体 运动学 几何特征]
    OBS --> POL[统一 MLP 策略 五步观测历史 加 base 速度估计器]
    DR --> POL
    POL --> EXP[Embodiment-wise Exploration 每个机器人独立探索方差 σ]
    POL --> LOSS[Embodiment-wise Update 按平均回报动态重加权各机器人损失]
    EXP --> POL
    LOSS --> POL
  end
  POL --> PT[预训练基础策略]
  subgraph FEW["Few-shot Transfer"]
    PT --> NEW[新机器人 零样本推理]
    PT --> FT[少量轨迹微调 初始采样用低动作方差]
    FT --> GAIT[学得步态]
  end
  GAIT --> REAL[真机部署]
```

图依据：论文 §IV-A 至 §IV-E / Fig.2（EXP/LOSS 与策略的回路箭头为按 Fig.2 标注归纳）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/h-zero-cross-humanoid-locomotion-pretraining-ena` |
| arXiv | <https://arxiv.org/abs/2512.00971> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 H-Zero 的阅读进度待读，不是方法归纳：目前页面能确定的只有分类归属与论文入口，跨形态预训练的具体做法与迁移代价一律以待撰写的深读笔记为准。**

- 页面正文未给出任何方法、指标或消融，「实验与评测」已明确标注待笔记完成后补充。
- 适用边界：可用于按 05_Locomotion 分类检索与追踪阅读进度，不可当作 H-Zero 的结论来引用。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_h-zero-cross-humanoid-locomotion-pretraining-ena.md](../../sources/papers/humanoid_pnb_h-zero-cross-humanoid-locomotion-pretraining-ena.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2512.00971>
- [arXiv PDF](https://arxiv.org/pdf/2512.00971)：已通读方法部分（实验与附录以略读为准）：§III Preliminaries、§IV Cross-Embodiment Humanoid Control（A–E）与 Fig.2、§V 实验设定与 Table II

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
