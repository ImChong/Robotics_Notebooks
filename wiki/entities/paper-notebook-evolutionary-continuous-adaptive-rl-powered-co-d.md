---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2509.26082"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_evolutionary-continuous-adaptive-rl-powered-co-d.md
summary: "Evolutionary Continuous Adaptive RL-Powered Co-Design for Humanoid Chin-Up Performance：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Evolutionary Continuous Adaptive RL-Powered Co-Design for Humanoid Chin-Up Performance

**Evolutionary Continuous Adaptive RL-Powered Co-Design for Humanoid Chin-Up Performance** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：12_Hardware_Design）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Evolutionary Continuous Adaptive RL-Powered Co-Design for Humanoid Chin-Up Performance 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    B["基座策略 π0：在初始设计 D0 上预训练"] -->|"热启动"| FT
    subgraph OUT["外循环：设计进化（CMA-ES）"]
        P["从高斯分布采样设计种群 Di（齿轮比因子 d_leg, d_shoulder, d_elbow, d_wrist）"] --> EVAL["适应度 Jpop（由 RL 训练结果得到）"]
        EVAL --> SEL["选择高性能候选，进化到下一代"]
        SEL --> P
    end
    subgraph INN["内循环：持续自适应（Isaac Gym 并行环境）"]
        FT["微调策略 π*：对每个新设计种群做 PPO 微调"] --> TA["任务适配策略 πθ"]
        TA -->|"若 Jpop < J*：更新微调策略"| FT
    end
    P --> FT
    FT --> EVAL
    SEL --> BEST["最优设计 d* + 优化策略 π*（RH5 引体向上）"]
```

图依据：论文 §II / Fig. 2、§III-C 与 Algorithm 1 的描述

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 12_Hardware_Design |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/12_Hardware_Design/evolutionary-continuous-adaptive-rl-powered-co-d` |
| arXiv | <https://arxiv.org/abs/2509.26082> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Evolutionary Continuous Adaptive RL-Powered Co-Design for Humanoid Chin-Up Performance 目前还没读：本页只登记了它的分类位置与原文入口，给不出可直接引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2509.26082、分类 12_Hardware_Design、计划中的笔记文件夹；论文的方法机制、评测口径与实机结果在本页一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-12-hardware-design.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为技术选型或方法对比的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 progress.json 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_evolutionary-continuous-adaptive-rl-powered-co-d.md](../../sources/papers/humanoid_pnb_evolutionary-continuous-adaptive-rl-powered-co-d.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2509.26082>
- [arXiv PDF](https://arxiv.org/pdf/2509.26082)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§II 方法（外循环设计进化、内循环持续自适应、最优设计）、§III 实验设置（设计空间、观测/动作/奖励）；§IV 实验结果仅略读。

## 推荐继续阅读

- [机器人论文阅读笔记：Evolutionary Continuous Adaptive RL-Powered Co-Design for Humanoid Chin-Up Performance](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/EA-CoRL_Evolutionary_Continuous_Adaptive_RL_Co-Design_for_Humanoid_Chin-Up/EA-CoRL_Evolutionary_Continuous_Adaptive_RL_Co-Design_for_Humanoid_Chin-Up.html)
- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
