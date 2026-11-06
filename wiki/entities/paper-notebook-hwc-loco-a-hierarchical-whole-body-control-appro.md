---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2503.00923"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_hwc-loco-a-hierarchical-whole-body-control-appro.md
summary: "HWC-Loco：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# HWC-Loco

**HWC-Loco: A Hierarchical Whole-Body Control Approach to Robust Humanoid Locomotion** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

HWC-Loco 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
    subgraph ST1["阶段一：训练两个低层策略"]
        REF["参考动作（CMU MoCap 重定向：站立 / 行走 / 跑步）"] --> DISC["判别器 fd（Wasserstein-1 + 梯度惩罚）"]
        CMDU["用户速度指令"] --> GT["Goal-Tracking Policy π1（PPO）<br/>最大化任务奖励 rT 并模仿人类动作"]
        DISC --> GT
        EXT["极端初始化 + 极端情形不确定集<br/>（多尺度外力 / 本体噪声与 PD 增益 / 恶意指令重采样 / 域随机化）"] --> SR["Safety Recovery Policy π2<br/>奖励 + 模仿 + ZMP 可行性约束 ϕ"]
        DISC --> SR
    end
    subgraph ST2["阶段二：训练高层规划策略"]
        HIST["历史观测 + 机器人当前状态"] --> HP["High-level Policy π0（离散动作：选择 π1 或 π2）<br/>奖励 rT − 切换惩罚 − α·终止惩罚"]
    end
    GT --> ST2
    SR --> ST2
    HP -->|"选择激活"| LOW["被选中的低层策略输出动作"]
    LOW --> PD["PD 控制器（100 Hz）→ 力矩 → 人形机器人"]
    VAE["VAE 估计器：由观测历史推断特权信息的上下文嵌入 zt 与 et（体速、ZMP 特征）"] -.->|"部署时替代特权信息"| LOW
```

图依据：论文 §4 / §4.1 Goal-Tracking Policy、§4.2 Safety Recovery Policy、§4.3 High-Level Planning / Fig.2。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/hwc-loco-a-hierarchical-whole-body-control-appro` |
| arXiv | <https://arxiv.org/abs/2503.00923> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页是 HWC-Loco 的阅读进度待读，不是方法归纳：页面里可确定的只有它被归入 05_Locomotion 与论文入口，分层全身控制如何提升鲁棒行走待深读笔记补齐。**

- 页面正文未给出任何方法、指标或消融，「实验与评测」已明确标注待笔记完成后补充。
- 适用边界：可用于按分类检索与追踪阅读进度，不可当作 HWC-Loco 的结论来引用。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_hwc-loco-a-hierarchical-whole-body-control-appro.md](../../sources/papers/humanoid_pnb_hwc-loco-a-hierarchical-whole-body-control-appro.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2503.00923>
- [arXiv PDF](https://arxiv.org/pdf/2503.00923)：已通读方法部分（实验与附录以略读为准）：§4 Hierarchical Whole-Body Control（4.1 Goal-Tracking、4.2 Safety Recovery、4.3 High-Level Planning、Practical Deployment）、Fig.2、§5 实验设定（略读）

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
