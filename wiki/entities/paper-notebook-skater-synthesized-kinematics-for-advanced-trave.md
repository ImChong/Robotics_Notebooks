---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2601.04948"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_skater-synthesized-kinematics-for-advanced-trave.md
summary: "SKATER：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# SKATER

**SKATER: Synthesized Kinematics for Advanced Traversing Efficiency on a Humanoid Robot via Roller Skate Swizzles** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

SKATER 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    subgraph SIM[Isaac Lab 4096 并行环境, PPO 非对称 Actor-Critic]
        DR[域随机化: 轮地接触 / 惯性 / 执行器增益 / 外部扰动<br/>指令范围与随机化幅度渐进扩大]
        RW[参考无关奖励, 共 22 项: 速度跟踪 + 四类物理偏置<br/>双足间距上下界 / 轮轴向滑移抑制 / ZMP 支撑稳定 / 左右对称]
    end
    OBS[Actor 观测: 5 步本体感知历史<br/>关节位置速度 + 角速度 + 投影重力 + 速度指令 + 上一动作] --> ACTOR[策略网络 π_θ]
    PRIV[Critic 特权观测: 无噪真值线速度 / ZMP 偏移 / 脚间距 / 踝角度] --> CRITIC[Critic]
    ACTOR --> ACT[25 维动作 q_default + 0.25 · a]
    ACT --> PD[关节 PD 控制器 -> 力矩]
    PD --> ROBOT[SKATER 33 DoF: 25 驱动 + 8 被动轮]
    DR --> ROBOT
    ROBOT -->|奖励| RW
    RW --> CRITIC
    ACTOR -.->|MuJoCo sim-to-sim| MJ[MuJoCo 验证]
    MJ --> HW[真机部署与实验]
```

图依据：论文 §III-B（Fig. 3、Table I/II）、§IV。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/skater-synthesized-kinematics-for-advanced-trave` |
| arXiv | <https://arxiv.org/abs/2601.04948> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 SKATER 的待读条目：轮滑 swizzle 这种「换移动方式提效率」的路线值得单列一个实体，但页面本身尚无可引用的机制与数据。**

- 页面上唯一确定的信息是归类与计划路径：分类 05_Locomotion、计划文件夹 `papers/05_Locomotion/skater-synthesized-kinematics-for-advanced-trave`，深读状态仍为「待撰写」。
- 适用边界：可用于分类检索与交叉链接，不可当作方法或指标依据——本页未给出运动学合成方式，也没有任何效率对比数据。
- 主要风险是被误当成已消化的笔记：本页「实验与评测」已明确 benchmark、消融与实机指标待补，引用前应回到 PROGRESS.md 与 <https://arxiv.org/abs/2601.04948>。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_skater-synthesized-kinematics-for-advanced-trave.md](../../sources/papers/humanoid_pnb_skater-synthesized-kinematics-for-advanced-trave.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2601.04948>
- [arXiv PDF](https://arxiv.org/pdf/2601.04948)：已通读方法部分（实验与附录以略读为准）：§III-A 硬件设计、§III-B 强化学习（观测/动作/奖励/sim-to-real）、§IV 实验设置

## 推荐继续阅读

- [机器人论文阅读笔记：SKATER](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/05_Locomotion/SKATER__Synthesized_Kinematics_for_Advanced_Traversing_Efficiency_via_Roller_Skate_Swizzles/SKATER__Synthesized_Kinematics_for_Advanced_Traversing_Efficiency_via_Roller_Skate_Swizzles.html)
- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
