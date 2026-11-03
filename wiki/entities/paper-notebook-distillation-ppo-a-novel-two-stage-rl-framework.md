---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2503.08299"
related:
  - ../overview/paper-notebook-category-03-high-impact-selection.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_distillation-ppo-a-novel-two-stage-rl-framework.md
summary: "Distillation-PPO：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Distillation-PPO

**Distillation-PPO: A Novel Two-Stage RL Framework for Humanoid Robot Perceptive Locomotion** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：03_High_Impact_Selection）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Distillation-PPO 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 深读笔记尚未写出；方法流程见下方「流程总览」（依据论文方法部分），其余结论以论文为准。

## 流程总览

```mermaid
flowchart TD
    subgraph SENS["感知与高程图"]
        LIO["Livox MID-360 + IMU → FAST-LIO2 位姿"] --> EM["高程图（一维 Kalman 滤波逐格更新）"]
        DEP["Orbbec 深度图 → 点云；用关节角 + 正运动学剔除腿部遮挡"] --> EM
        EM --> SCAN["Scan dots 向量 m（441 维）"]
    end
    subgraph S1["阶段一：Teacher（完全可观测 MDP）"]
        SCAN1["仿真器直接给出的干净 scan dots"] --> C1["Conv1D → 32 维感知隐变量"]
        H1["50 帧历史状态 → 外部编码器 → 32 维隐变量"] --> M1["组合 MLP"]
        C1 --> M1
        M1 --> TP["Teacher 策略（PPO，周期步态 + 指令 + 正则奖励）"]
    end
    subgraph S2["阶段二：Student（部分可观测 MDP）"]
        INIT["继承 Teacher 网络结构与参数初始化"] --> SP["Student 策略<br/>输入：加噪 scan dots + 历史本体感觉（域随机化）"]
        SP --> LOSS["Ltotal = α·Ldistillation（与 Teacher 动作的 MSE）+ β·LPPO"]
        RL["RL 奖励 → PPO 损失"] --> LOSS
        LOSS -->|"梯度同时更新策略与历史状态编码器"| SP
    end
    TP -->|"监督信号 / 初始化"| INIT
    TP -->|"监督信号"| LOSS
    SCAN -.->|"加高斯噪声模拟真实传感器"| SP
    SP --> DEPLOY["天工（Tien Kung）真机复杂地形行走"]
```

图依据：论文 §III-B、§IV-A Two-Stage RL with Teacher Distillation、§IV-B Rewards / Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 03_High_Impact_Selection |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/03_High_Impact_Selection/distillation-ppo-a-novel-two-stage-rl-framework` |
| arXiv | <https://arxiv.org/abs/2503.08299> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**Distillation-PPO 在本库目前只是一条「已排期、未深读」的待读条目：它保证高影响力选读清单里这条"两阶段 RL + 感知运动"的线索不缺节点，但还给不出可引用的方法结论。**

- 本页能确认的只有索引层信息——arXiv 2503.08299、分类 03_High_Impact_Selection 与计划中的笔记文件夹；标题指向"两阶段 RL 框架 + 人形感知运动"这一命题，蒸馏与 PPO 如何分工、两阶段各自训练什么，页内一概未展开。
- 适用边界：适合经[分类页](../overview/paper-notebook-category-03-high-impact-selection.md)与[机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md)做检索与交叉定位，不适合作为方案对比或工程选型的依据。
- 主要风险是把待读条目误读成已消化的结论：量化 benchmark、消融与实机指标本页全部缺失，深读状态须回到 PROGRESS.md 跟踪，结论以论文原文为准。

## 与其他页面的关系

- 分类页：[paper-notebook-category-03-high-impact-selection](../overview/paper-notebook-category-03-high-impact-selection.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_distillation-ppo-a-novel-two-stage-rl-framework.md](../../sources/papers/humanoid_pnb_distillation-ppo-a-novel-two-stage-rl-framework.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2503.08299>
- [arXiv PDF](https://arxiv.org/pdf/2503.08299)：已通读方法部分（实验与附录以略读为准）：§III-B 高程图与 Scan Dots、§IV Method（A 两阶段蒸馏 RL、B 奖励与感知）、§V-A 训练与实现细节

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
