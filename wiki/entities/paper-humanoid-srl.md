---
type: entity
tags: [paper, humanoid, supernumerary-robotic-limbs, loco-manipulation, reinforcement-learning, trajectory-optimization, diffusion-policy, georgia-tech]
status: complete
updated: 2026-10-21
arxiv: "2610.25628"
venue: "2026 · arXiv"
related:
  - ../tasks/loco-manipulation.md
  - ../methods/reinforcement-learning.md
  - ../methods/diffusion-motion-generation.md
sources:
  - ../../sources/papers/humanoid_srl_arxiv_2610_25628.md
summary: "Humanoid SRL 研究在人形机器人上集成超冗余机械臂，以模型式轨迹优化生成动态可行运动、训练目标条件强化学习控制器，并将仿真策略零样本迁移至硬件；同时展示基于硬件示范训练的扩散策略用于视觉引导自主执行。"
---

# Humanoid SRL

**Humanoid SRL: Augmenting Humanoids with Supernumerary Limbs for Coordinated Loco-Manipulation**（arXiv:2610.25628，Georgia Institute of Technology）研究在人形机器人上增加超冗余机械臂，扩展双臂形态的移动操作能力。

## 一句话定义

通过 **模型式轨迹优化生成动态可行运动 → 目标条件强化学习控制 → 仿真到硬件零样本迁移**，让新增机械臂与人形机器人的全身运动和平衡协调；部分任务另使用从硬件示范训练的扩散策略进行视觉引导自主执行。

## 方法概览

| 组件 | 作用 |
|---|---|
| 超冗余机械臂（SRL） | 扩展人形机器人可实现的全身交互、操控和支撑能力 |
| 模型式轨迹优化（TO） | 生成可供控制器学习的动态可行运动 |
| 目标条件强化学习（RL） | 学习新增肢体与原有身体协调的控制策略 |
| 仿真到硬件迁移 | 论文报告控制器从仿真零样本迁移至硬件 |
| 示范学习扩散策略 | 从硬件示范训练，在选定任务上执行视觉引导自主操作 |

## 论文报告的验证

- 在两种配置下进行定量分析与硬件实验，评估形态增强如何改变可行交互空间。
- 展示新增机械臂与全身运动协同的接触丰富三臂移动操作技能。
- 在选定任务上展示基于硬件示范训练的扩散策略进行视觉引导自主执行。

## 复现信息

- **论文：** [arXiv:2610.25628](https://arxiv.org/abs/2610.25628)，v1 于 2026-10-19 提交。
- **代码 / 数据：** arXiv 摘要未列出官方仓库或数据集链接；本文暂不标注其开源状态为已发布。
- **实验细节边界：** 本页仅记录 arXiv 摘要确认的信息。具体硬件配置、任务指标与控制器细节应以论文全文为准。

## 参考来源

- [Humanoid SRL 来源摘录](../../sources/papers/humanoid_srl_arxiv_2610_25628.md)
- arXiv：<https://arxiv.org/abs/2610.25628>
