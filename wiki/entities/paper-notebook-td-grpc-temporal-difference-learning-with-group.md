---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: stub
updated: 2026-09-15
arxiv: "2505.13549"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_td-grpc-temporal-difference-learning-with-group.md
summary: "TD-GRPC：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# TD-GRPC

**TD-GRPC: Temporal Difference Learning with Group Relative Policy Constraint for Humanoid Locomotion** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

TD-GRPC 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 在深读笔记写出来之前，本页只保留分类位置与原文入口；方法模块与数据流见下方「流程总览」（仅依据论文方法部分，不含实验结论）。

## 流程总览

```mermaid
flowchart TD
    A["初始状态 s_t"] --> B["编码器 h_θ → 潜状态 z_t"]
    B --> C["潜动力学 d_θ 滚动 H 步（MPPI 规划时域）"]
    C --> D["R_θ / Q_θ / π_θ（MLP）估计奖励、Q 值与动作"]
    D --> E["策略 π_θ(z) 采样 G 个动作作为一组"]
    E --> F["用 Q_θ 评估各动作，softmax 得组相对优势 A_g"]
    F --> G["改进的 GRPO 目标：Σ A_g · log π_θ"]
    E --> H["KL 散度信任域约束：当前策略 vs MPPI 导出的先验策略 π_old"]
    G --> I["策略损失 L_π = GRPO 项 + β · 策略约束项"]
    H --> I
    I --> J["更新策略参数 θ"]
    C --> K["模型损失：潜状态预测 + 奖励 + TD 价值目标（式 10a-10c）"]
    K --> J
    J --> L["推理：与 TD-MPC 相同，对 rollout 策略做 CEM 规划选取动作"]
    L --> M["环境交互，经验存入潜缓冲区 D"]
    M -.->|"采样 H 步轨迹"| A
```

图依据：论文 §III-A / §III-B、Fig.2、Algorithm 1

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/td-grpc-temporal-difference-learning-with-group` |
| arXiv | <https://arxiv.org/abs/2505.13549> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前的价值是待读条目与检索锚点，而不是对 TD-GRPC 的技术判断——真正的结论要等深读笔记落地后重写。**

- 页面能确定的只有元信息：分类 05_Locomotion、PROGRESS.md 标记为待深读、计划文件夹路径与 arXiv 入口；核心机制、量化 benchmark、消融与实机指标均尚未编译。
- 因此不应把本页当作引用该工作的技术依据；需要结论时请回到 arXiv 原文，或等待笔记站的深读页。
- 适用边界很窄：本页只服务于 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页之间的交叉检索，方便你按分类找到它。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_td-grpc-temporal-difference-learning-with-group.md](../../sources/papers/humanoid_pnb_td-grpc-temporal-difference-learning-with-group.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2505.13549>
- [arXiv PDF](https://arxiv.org/pdf/2505.13549)：已通读方法部分（实验与附录以略读为准）：§I 引言与贡献、§III-A 问题设定、§III-B TD-GRPC（策略约束、GRPO、模型目标）、Algorithm 1、Fig.2；§IV 仅读实验设定

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
