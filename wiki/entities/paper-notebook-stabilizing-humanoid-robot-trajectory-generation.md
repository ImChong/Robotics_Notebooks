---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2509.24697"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_stabilizing-humanoid-robot-trajectory-generation.md
summary: "Stabilizing Humanoid Robot Trajectory Generation via Physics-Informed Learning：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Stabilizing Humanoid Robot Trajectory Generation via Physics-Informed Learning

**Stabilizing Humanoid Robot Trajectory Generation via Physics-Informed Learning** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Stabilizing Humanoid Robot Trajectory Generation via Physics-Informed Learning 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口；方法流程见下方「流程总览」，效果结论仍待深读。

## 流程总览

```mermaid
flowchart TD
    subgraph TRAIN["训练"]
        DATA["人体动捕重定向数据（ergoCub，n=26 关节）"] --> MANN
        MANN["MANN：门控网络 + 运动预测网络 → 下一时刻完整运动学状态 (q, ν)"] --> LOSS["损失 L = L_D 数据拟合 + w·L_B 物理信息项（支撑脚速度为零）"]
        LOSS --> MANN
    end
    subgraph DEP["在线部署"]
        UI["用户期望运动方向 / 朝向"] --> GEN["轨迹生成：MANN 预测基座速度、关节位置与速度、基座位姿"]
        GEN --> COR["控制启发的纠偏：对预测基座线 / 角速度做反馈"]
        UI --> COR
        COR -->|"作为下一步网络输入"| GEN
        GEN --> ADJ["轨迹调整与轨迹控制"]
        COR --> ADJ
    end
    MANN -.-> GEN
    ADJ --> ROBOT["ergoCub 实机：离线控制（ZMP/CoM 稳定 + 全身 QP）或在线控制（MPC + QP）"]
```

图依据：论文 §III / Fig. 2、§II-D（控制器）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/stabilizing-humanoid-robot-trajectory-generation` |
| arXiv | <https://arxiv.org/abs/2509.24697> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是这篇「物理约束入学习」工作的待读条目：它在清单里登记了 physics-informed 轨迹生成这条线，但稳定性从何而来仍待笔记回答。**

- 页面上唯一确定的信息是归类与计划路径：分类 05_Locomotion、计划文件夹 `papers/05_Locomotion/stabilizing-humanoid-robot-trajectory-generation`，深读状态仍为「待撰写」。
- 适用边界：可用于分类检索与交叉链接，不可当作方法或指标依据——本页未说明物理先验以何种形式进入学习，也未给出任何量化结果。
- 主要风险是被误当成已消化的笔记：本页「实验与评测」已明确 benchmark、消融与实机指标待补，引用前应回到 PROGRESS.md 与 <https://arxiv.org/abs/2509.24697>。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_stabilizing-humanoid-robot-trajectory-generation.md](../../sources/papers/humanoid_pnb_stabilizing-humanoid-robot-trajectory-generation.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2509.24697>
- [arXiv PDF](https://arxiv.org/pdf/2509.24697)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§II 背景（PIML、MANN、轨迹控制器）、§III 方法（训练特征、物理信息损失、控制启发的纠偏）；§IV 实验仅略读。

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
