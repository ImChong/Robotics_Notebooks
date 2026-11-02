---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "1811.09656"
related:
  - ../overview/paper-notebook-category-13-physics-based-animation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_hierarchical-visuomotor-control-of-humanoids.md
summary: "Hierarchical visuomotor control of humanoids：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Hierarchical visuomotor control of humanoids

**Hierarchical visuomotor control of humanoids** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：13_Physics-Based_Animation）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Hierarchical visuomotor control of humanoids 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置、原文入口与依据论文方法部分的「流程总览」，不给指标结论。

## 流程总览

```mermaid
flowchart TD
    M["动作捕捉片段 (2-6 秒)"] --> T["单片段跟踪策略 π(a|s,t)<br/>先监督预训练, 再 RL 模仿 (奖励 r=exp(-βE))"]
    T --> LL["低层运动控制器 (仅本体感觉输入)"]
    LL --> L1["Steerable 控制器<br/>高层输出 1 维转向角"]
    LL --> L2["Switching 控制器<br/>1-of-k 选行为, 转移图约束"]
    LL --> L3["Control fragments 冷切换<br/>0.1-0.3 秒微行为, 高层 1-of-K 选择"]
    CAM["第一人称相机 (躯干根部)"] --> RES["ResNet 编码"]
    PRO["本体感觉"] --> LIN["线性层编码"]
    RES --> CAT["拼接"]
    LIN --> CAT
    CAT --> LSTM["LSTM 记忆"]
    LSTM --> HL["高层控制器: 策略头 + 价值头<br/>(稀疏任务奖励训练)"]
    HL -->|"每 k 步 / 每个 gait cycle 选择"| L1
    HL --> L2
    HL --> L3
    L1 --> ACT["关节位置控制 -> 56 DoF 人形"]
    L2 --> ACT
    L3 --> ACT
```

图依据：论文 §2 Approach（§2.1 跟踪动捕片段 / §2.2 低层控制器类型 / §2.3 训练高层策略）/ Fig.1–4

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 13_Physics-Based_Animation |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/13_Physics-Based_Animation/hierarchical-visuomotor-control-of-humanoids` |
| arXiv | <https://arxiv.org/abs/1811.09656> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是分层视觉运动控制这条线的阅读进度待读，不是方法归纳：页面里可确定的只有它被归入物理动画分类与论文入口，层级划分与视觉接入的方法流程见「流程总览」，细节与指标待深读笔记补齐。**

- 页面正文仅在「流程总览」给出方法流程，未给出指标或消融，「实验与评测」已明确标注待笔记完成后补充。
- 适用边界：可用于按 13_Physics-Based_Animation 分类检索与追踪阅读进度，不可当作该工作的结论来引用。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-13-physics-based-animation](../overview/paper-notebook-category-13-physics-based-animation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_hierarchical-visuomotor-control-of-humanoids.md](../../sources/papers/humanoid_pnb_hierarchical-visuomotor-control-of-humanoids.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/1811.09656>
- [arXiv PDF](https://arxiv.org/pdf/1811.09656)：已通读方法部分（实验与附录以略读为准）：§1 引言、§2 Approach（2.1–2.3）、§3.1 核心任务设定；其余实验与附录略读

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
