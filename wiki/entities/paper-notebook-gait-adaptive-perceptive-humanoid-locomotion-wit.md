---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2512.07464"
related:
  - ../overview/paper-notebook-category-03-high-impact-selection.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_gait-adaptive-perceptive-humanoid-locomotion-wit.md
summary: "Gait-Adaptive Perceptive Humanoid Locomotion with Real-Time Under-Base Terrain Reconstruction：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Gait-Adaptive Perceptive Humanoid Locomotion with Real-Time Under-Base Terrain Reconstruction

**Gait-Adaptive Perceptive Humanoid Locomotion with Real-Time Under-Base Terrain Reconstruction** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：03_High_Impact_Selection）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Gait-Adaptive Perceptive Humanoid Locomotion with Real-Time Under-Base Terrain Reconstruction 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
  subgraph PERC[under-base 地形感知 §III-E]
    DI[机身下方朝下深度相机 单帧深度图] --> PC[变换为重力对齐点云 投影成 raw height map]
    PC --> UN[U-Net 编码器 解码器]
    UN --> HH[Height Head 输出 refined height map]
    UN --> EH[Edge Head 仅训练期 辅助 BCE + Dice]
  end
  subgraph TS[Successive Teacher-Student 单阶段训练 §III-C D]
    GATE[T-S Switch Gate 学生环境占比 λ 由 0 逐步增大] --> TE[教师 privileged encoder 加 无噪声 height map]
    GATE --> ST[学生 proprio 历史 encoder 加 带噪声 height map]
    ST -.->|MSE 对齐教师隐变量 z_T| TE
    TE --> POL[共享 policy head π_θ 与 critic V_φ PPO]
    ST --> POL
  end
  HH --> ST
  POL --> ACT[32 维动作 31 个关节目标 加 标量步频 f_t]
  ACT --> PH[步频后处理 缩放 裁剪 平均滤波 更新步态相位 φ_t]
  ACT --> PD[PD 控制 1 kHz]
  PH --> POL
```

图依据：论文 §III-A 至 §III-E / Fig.2–4 与 Algorithm 1（PH→POL 的相位回馈箭头依 §III-B 的 φ_t 作为观测信号归纳）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 03_High_Impact_Selection |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/03_High_Impact_Selection/gait-adaptive-perceptive-humanoid-locomotion-wit` |
| arXiv | <https://arxiv.org/abs/2512.07464> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Gait-Adaptive Perceptive Humanoid Locomotion with Real-Time Under-Base Terrain Reconstruction 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2512.07464、分类 03_High_Impact_Selection、计划中的笔记文件夹；论文的方法机制、评测口径与实机结果在本页一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-03-high-impact-selection.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-03-high-impact-selection](../overview/paper-notebook-category-03-high-impact-selection.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_gait-adaptive-perceptive-humanoid-locomotion-wit.md](../../sources/papers/humanoid_pnb_gait-adaptive-perceptive-humanoid-locomotion-wit.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2512.07464>
- [arXiv PDF](https://arxiv.org/pdf/2512.07464)：已通读方法部分（实验与附录以略读为准）：§I 贡献概述、§III Method（A–E）与 Fig.2–4、Algorithm 1

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
