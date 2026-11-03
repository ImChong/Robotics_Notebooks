---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2402.18294"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_whole-body-humanoid-robot-locomotion-with-human.md
summary: "Whole-body Humanoid Robot Locomotion with Human Reference：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Whole-body Humanoid Robot Locomotion with Human Reference

**Whole-body Humanoid Robot Locomotion with Human Reference** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Whole-body Humanoid Robot Locomotion with Human Reference 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |

## 为什么重要

- 这篇已排进 Paper Notebooks 的待读清单，可以从 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页找到同一批工作。
- 方法流程见上方「流程总览」（依据论文方法部分）；深读笔记写出来之前，这里不给更多方法结论。

## 流程总览

```mermaid
flowchart TD
    M1["公开动捕库 SFU / CMU<br/>+ 自采高精度动捕"] --> RT["手动标定转换（re-targeting）<br/>得到 Adam 参考动作"]
    RT --> DR["判别器 D 的参考状态转移<br/>(关节速度/位置、双手双脚位置，58 维)"]
    POL["策略 π（PPO，Isaac Gym 4096 并行环境）"] --> ENV["仿真环境（含域随机化）"]
    ENV --> OBS["观测 91 维：IMU、关节编码器、速度命令、<br/>周期相位 sin/cos 与摆动占比 ρ、根高度、上一步动作"]
    OBS --> POL
    POL --> ACT["动作：25 维关节位置目标 → PD 控制器"]
    ACT --> ENV
    ENV --> D["判别器 D（AMP）"]
    DR --> D
    D --> RI["模仿奖励 rI"]
    ENV --> TR["任务奖励：命令奖励 + 周期奖励 + 正则奖励"]
    RI --> POL
    TR --> POL
    POL --> CV["交叉验证：Webots 与 Isaac Gym"]
    CV -->|"反馈微调"| POL
    CV --> RB["部署真机 Adam"]
```

图依据：论文 §III-B、§IV-A/B、§V-A/B / Fig.3。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/whole-body-humanoid-robot-locomotion-with-human` |
| arXiv | <https://arxiv.org/abs/2402.18294> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页当前的价值是待读条目与检索锚点，而不是对这篇「人类参考驱动全身运动」工作的技术判断——真正的结论要等深读笔记落地后重写。**

- 页面能确定的只有元信息：分类 05_Locomotion、PROGRESS.md 标记为待深读、计划文件夹路径与 arXiv 入口；人类参考的使用方式、量化 benchmark、消融与实机指标均尚未编译。
- 因此不应把本页当作引用该工作的技术依据；需要结论时请回到 arXiv 原文，或等待笔记站的深读页。
- 适用边界很窄：本页只服务于 [机器人学习论文笔记总索引](../overview/humanoid-paper-notebooks-index.md) 与分类页之间的交叉检索，方便你按分类找到它。
- 深读笔记完成后，本页会补上笔记链接与实质要点。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_whole-body-humanoid-robot-locomotion-with-human.md](../../sources/papers/humanoid_pnb_whole-body-humanoid-robot-locomotion-with-human.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2402.18294>
- [arXiv PDF](https://arxiv.org/pdf/2402.18294)：已通读方法部分（实验与附录以略读为准）：§III 预备（机器人结构、动捕与重定向）、§IV 方法（AMP 模仿、任务奖励）、§V-A 训练细节、§V-B 交叉验证与反馈微调

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
