---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2512.23649"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_robomirror-understand-before-you-imitate-for-vid.md
summary: "RoboMirror：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# RoboMirror

**RoboMirror: Understand Before You Imitate for Video to Humanoid Locomotion** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

RoboMirror 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」（依据论文方法部分），不给实验结论。

## 流程总览

```mermaid
flowchart LR
    subgraph S1[阶段 1: 视频 -> 动作 latent 重建]
        V[第一人称 / 第三人称视频 + 任务提示词] --> Q[Qwen3-VL 视频理解 -> 视频 latent l_VLM]
        MD[动作数据集] --> VAE[动作 VAE -> 真值动作 latent l_motion]
        Q --> DT[流匹配 DiT D_θ: cross-attention 条件于 l_VLM]
        VAE -->|加噪 l_motion| DT
        DT --> LV[重建动作 latent l_v2m]
    end
    subgraph S2[阶段 2: 策略学习, 仿真]
        T0[初始策略 π0 在第三人称动作数据上训练] --> FLT[按下肢跟踪误差 e s 过滤, e 大于 0.6 的序列剔除]
        FLT --> TCH[MoE 残差教师 PPO: 特权信息 + 参考动作<br/>输出修正量 δa, 动作 = 参考关节位置 + δa]
        TCH -->|蒸馏| STU[扩散学生策略: 动作 latent + 观测历史]
    end
    LV --> STU
    STU --> ACT[DDIM 采样 + MLP 扩散 -> 可部署动作]
    ACT --> ROBOT[人形机器人: 无需位姿估计与重定向]
```

图依据：论文 §3.1–3.3（Fig. 2）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/robomirror-understand-before-you-imitate-for-vid` |
| arXiv | <https://arxiv.org/abs/2512.23649> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 RoboMirror 的待读条目：标题里「先理解再模仿」的主张暗示了视频到人形运动这条路线的关键分歧点，机制见「流程总览」，但页面尚未提供可引用的实验证据。**

- 页面上唯一确定的信息是归类与计划路径：分类 05_Locomotion、计划文件夹 `papers/05_Locomotion/robomirror-understand-before-you-imitate-for-vid`，深读状态仍为「待撰写」。
- 适用边界：可用于分类检索与交叉链接，不可当作方法或指标依据——「理解」环节的具体实现见「流程总览」，本页未给出任何量化结果。
- 主要风险是被误当成已消化的笔记：本页「实验与评测」已明确 benchmark、消融与实机指标待补，引用前应回到 PROGRESS.md 与 <https://arxiv.org/abs/2512.23649>。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_robomirror-understand-before-you-imitate-for-vid.md](../../sources/papers/humanoid_pnb_robomirror-understand-before-you-imitate-for-vid.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2512.23649>
- [arXiv PDF](https://arxiv.org/pdf/2512.23649)：已通读方法部分（实验与附录以略读为准）：§3.1 概览、§3.2 由 VLM 重建动作 latent、§3.3 策略训练与推理流程

## 推荐继续阅读

- [机器人论文阅读笔记：RoboMirror](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/05_Locomotion/RoboMirror__Understand_Before_You_Imitate_for_Video_to_Humanoid_Locomotion/RoboMirror__Understand_Before_You_Imitate_for_Video_to_Humanoid_Locomotion.html)
- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
