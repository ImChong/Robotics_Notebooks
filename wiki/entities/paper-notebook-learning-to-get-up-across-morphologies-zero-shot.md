---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-planned]
status: planned
updated: 2026-09-15
arxiv: "2512.12230"
related:
  - ../overview/paper-notebook-category-05-locomotion.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_learning-to-get-up-across-morphologies-zero-shot.md
summary: "Learning to Get Up Across Morphologies：列入 Paper Notebooks PROGRESS.md 待深读清单；深读笔记完成后补成完整摘要。"
---

# Learning to Get Up Across Morphologies

**Learning to Get Up Across Morphologies: Zero-Shot Recovery with a Unified Humanoid Policy** 已列入 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html) 的 **PROGRESS.md 待深读** 清单（分类：05_Locomotion）。本页 **还没有深读笔记**：只给出这篇论文的分类位置与原文入口，方法细节与数据请直接看原文。

## 一句话定义

Learning to Get Up Across Morphologies 的人形机器人学习论文条目，当前处于 Paper Notebooks 阅读进度（待深读）阶段。

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
  M[7 个人形机器人 MJCF 模型 MuJoCo] --> E[每个 episode 随机选 1 个形态 16 个并行环境]
  E --> I[随机倒地初始化 躯干与关节偏移最多 ±90°]
  E --> DR[领域随机化 质量 摩擦 电压 IMU 偏置]
  I --> OBS[形态无关观测 关节位置速度 期望位置 躯干欧拉角与角速度 头高 上一动作 无形态 ID]
  DR --> OBS
  OBS --> POL[统一策略 SAC + CrossQ 512-512-256]
  POL --> ACT[动作 期望关节角速度 肩 肘 髋 膝 踝 pitch]
  ACT --> SIM[MuJoCo 每 50 ms 更新目标关节角]
  SIM --> R[形态无关奖励 R_Up + R_Pitch + R_vel + R_var + R_collision]
  SIM --> TERM[提前终止 躯干翻转超过 135° 或动作过猛]
  R --> POL
  POL --> EVAL[留一法零样本评测 未见形态]
```

图依据：论文 §3.1–3.7、§4.1（文中无整体流程图，箭头为按方法各小节顺序归纳）

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 05_Locomotion |
| 深读状态 | 待撰写（[PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)） |
| 计划文件夹 | `papers/05_Locomotion/learning-to-get-up-across-morphologies-zero-shot` |
| arXiv | <https://arxiv.org/abs/2512.12230> |

## 实验与评测

- 深读笔记尚未完成；量化 benchmark、消融与实机指标待笔记撰写后补充。

## 结论

**本页只是把这篇「跨形态起身」的工作归到 05_Locomotion 分类下的待读条目，尚不构成对其方法与结论的归纳。**

- 页面可确证的只有分类归属、PROGRESS.md 上的「待深读」状态与 arXiv 编号 2512.12230；统一策略如何做到零样本恢复、在哪些形态上验证，页面均未记录。
- 适用边界：可用于分类检索，不能作为技术依据或引用来源。
- 主要风险是把待读条目当成已完成摘要；深读状态以上游 PROGRESS.md 为准，笔记完成后本页会补上笔记链接并重写。

## 与其他页面的关系

- 分类页：[paper-notebook-category-05-locomotion](../overview/paper-notebook-category-05-locomotion.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)

## 参考来源

- [humanoid_pnb_learning-to-get-up-across-morphologies-zero-shot.md](../../sources/papers/humanoid_pnb_learning-to-get-up-across-morphologies-zero-shot.md)
- [Robot Learning Paper Notebooks · PROGRESS.md](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
- 论文：<https://arxiv.org/abs/2512.12230>
- [arXiv PDF](https://arxiv.org/pdf/2512.12230)：已通读方法部分（实验与附录以略读为准）：§3 Methods（3.1–3.7）、§4.1–4.3 实验设定

## 推荐继续阅读

- [Paper Notebooks 阅读进度（PROGRESS.md）](https://github.com/ImChong/Robot_Learning_Paper_Notebooks/blob/main/papers/PROGRESS.md)
