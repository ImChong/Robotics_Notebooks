---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2512.10477"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_symphony-a-heuristic-normalized-calibrated-advan.md
summary: "Symphony：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Symphony

**Symphony: A Heuristic Normalized Calibrated Advantage Actor and Critic Algorithm in application for Humanoid Robots** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Symphony 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
  X[初始化 Actor-Critic 单一对象 A_phi Q_theta 与 Target Critic] --> EXP[探索 N_exp = 10240 步 估计 r̄_n 并归一化奖励]
  EXP --> FILL[重复探索转移 50 次 填满 Fading Replay Buffer 512000]
  FILL --> ACT[Actor 输出动作 加有限 Gaussian 噪声 与环境交互]
  ACT --> STORE[存入 Fading Replay Buffer]
  STORE --> SAMP[按 tanh 渐隐优先级采样 batch 384]
  SAMP --> UPD[每帧 G = 3 次更新]
  subgraph ONE[单次更新]
    NA[Online Actor 预测下一动作 a' 与 σ β] --> TC[Target Critic 3 个 Critic 共 384 节点 排序后固定权重加权求和 得 Q_target]
    TC --> TA[Temporal Advantage Q_target 对比其指数滑动平均 Q_T 再除以 abs Q_T]
    TA --> LA[Actor 损失 ReHAE 加 Swaddling 正则]
    TC --> LC[Critic 损失 ReHSE TD 误差]
  end
  UPD --> NA
  LA --> POLY[同时更新 Online Actor 与 Critic]
  LC --> POLY
  POLY --> SOFT[Polyak 平均更新 Target Critic]
  SOFT --> ACT
```

图依据：论文 §III Foundation 与 Symphony: Pseudo-code（箭头按伪代码与正文归纳）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/symphony-a-heuristic-normalized-calibrated-advan` |
| arXiv | <https://arxiv.org/abs/2512.10477> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Symphony 在本库还只是一个待读条目：条目已归到 05_Locomotion 分类下，但「启发式归一化校准优势」的 actor-critic 究竟改了什么、换来什么，本页给不出答案。**

- 本页可确认的只有条目级信息：分类 05_Locomotion、arXiv 2512.10477、深读状态「待撰写」。
- 算法改动点、与标准 actor-critic 的差异、在人形上的量化收益均未涉及，不应据此页做方法选型或效果判断。
- 当前唯一职责是方便你按分类找到它，并维持与 [分类页](../overview/paper-notebook-category-05-locomotion.md) / [总索引](../overview/humanoid-paper-notebooks-index.md) 的交叉检索；深读笔记完成后本节应重写。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_symphony-a-heuristic-normalized-calibrated-advan.md](../../sources/papers/humanoid_pnb_symphony-a-heuristic-normalized-calibrated-advan.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2512.10477>
- [arXiv PDF](https://arxiv.org/pdf/2512.10477)：已通读方法部分（实验与附录以略读为准）：Preliminary、§III Foundation（Target Critic、Temporal Advantage、Fading Replay Buffer、ReSine、ReHSE/ReHAE、Swaddling）与伪代码

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
