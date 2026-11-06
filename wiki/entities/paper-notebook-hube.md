---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2508.19002"
venue: "2025.08"
related:
  - ../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_hube.md
summary: "HuBE：列入 Paper Notebooks progress 待深读清单；深读笔记完成后补成完整摘要。"
---

# HuBE

**HuBE: Cross-Embodiment Human-like Behavior Execution for Humanoid Robots** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **progress 待深读** 清单（分类：04_Loco-Manipulation_and_WBC）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

HuBE 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在笔记写出来之前，这里只保留分类位置与原文入口，不给方法结论。

## 流程总览

```mermaid
flowchart TD
    subgraph DATA["数据处理"]
        SRC["KIT / AMASS / Motion-X → HPose 数据集（11 个关键关节，四元数）"] --> LLMD["GPT-4o 生成情境描述 l"]
        SRC --> BONE["骨长缩放增广：按 9 款人形机器人身体参数得到 Pseudo GT"]
    end
    subgraph GEN["行为生成：模仿策略 π_θ（MLP 或 Transformer）"]
        ALIGN["输入对齐：MLP(s ⊕ g) ⊕ MLP(BERT(l))"] --> PI["π_θ 输出上半身关键点位姿动作 a"]
        HIND["Hindsight 训练：取窗口 H 内未来位姿作目标"] -.-> ALIGN
    end
    subgraph EXE["行为执行：两步 CLIK 逆运动学（Pinocchio）"]
        IK1["① 求躯干 / 颈 / 头关节角 J_torso（强调旋转位姿）"] --> IK2["② 锁定躯干，在缩减模型上求双臂关节角 J_arms"]
        IK2 --> FILT["SE(3) 群滤波"]
    end
    LLMD --> ALIGN
    BONE --> ALIGN
    IN["当前状态 s + 目标位姿 g（双手与头）+ 情境 l"] --> ALIGN
    PI --> IK1
    FILT --> ROBOT["机器人关节控制值"]
```

图依据：论文 §III-A 至 §III-C / Fig. 2、Fig. 3、Algorithm 1、式 (1)–(5)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 04_Loco-Manipulation_and_WBC |
| 深读状态 | 待撰写（[progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)） |
| 计划文件夹 | `papers/04_Loco-Manipulation_and_WBC/HuBE__Cross-Embodiment_Human-like_Behavior_Execution_for_Humanoid_Robots` |


## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**HuBE 在本库当前只是待深读的待读条目：它标记了「跨本体类人行为执行」这一方向，但页面本身不提供可引用的机制或结论。**

- 本页只是登记这篇论文属于 04_Loco-Manipulation_and_WBC 分类。
- 从标题可读出的定位是面向人形机器人的跨本体类人行为执行，属 loco-manipulation 与 WBC 脉络。
- 量化 benchmark、消融与实机指标本页一概没有，待深读笔记撰写后补充。
- 引用注意：深读完成前不宜把本页当作方法对比的依据。

## 与其他页面的关系

- 分类页：[paper-notebook-category-04-loco-manipulation-and-wbc](../overview/paper-notebook-category-04-loco-manipulation-and-wbc.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_hube.md](../../sources/papers/humanoid_pnb_hube.md)
- [Robot Learning Paper Notebooks · progress.json](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/progress.json)
- [arXiv PDF](https://arxiv.org/pdf/2508.19002)：已通读方法部分（实验与附录以略读为准）：§III Methodology（A–C）


## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
