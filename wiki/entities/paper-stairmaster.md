---
type: entity
tags:
  - paper
  - quadruped
  - locomotion
  - stairs
  - depth-perception
  - reinforcement-learning
  - sjtu
status: complete
updated: 2026-10-22
project_id: stairmaster
arxiv: "2606.25765"
project: https://sivan666666.github.io/StairMaster/
related:
  - ../overview/hub-sim2real.md
  - ../concepts/sim2real.md
  - ../concepts/physics-fidelity-sim2real-gap.md
  - ../queries/robot-perception-stack-selection-loop.md
sources:
  - ../../sources/papers/stairmaster_arxiv_2606_25765.md
summary: "StairMaster（2026，上海交通大学）：三阶段 RL 框架，用 cross-attention 与空间感知循环单元处理噪声深度，使 Unitree Go2 零样本攀爬坡度达 55° 的镂空楼梯。"
---

# StairMaster（四足机器人攀爬镂空楼梯）

## 一句话定义

**StairMaster 用三阶段 RL 训练四足机器人攀爬镂空楼梯：在噪声稀疏的深度输入下避免腿被卡住，并通过落脚点相关奖励改善踩点。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习，三阶段训练框架 |
| sim-to-real | Simulation-to-Real | 仿真到真机迁移；本文含深度传感器模型 |
| SRU | Spatial-aware Recurrent Unit | 空间感知循环单元，维持时空记忆（展开式据摘要） |
| Go2 | Unitree Go2 | 真机部署的四足平台 |

## 为什么重要

- 镂空楼梯的风险在于腿部被卡住，而深度感知稀疏且噪声大；两者叠加使 RL 策略难以学会踩点。
- 作者声称这是首个在真实世界实现此类陡峭镂空楼梯攀爬的 RL 策略（作者声称，未独立核实）。

## 方法与机制

- **Cross-Attention：** 从噪声深度输入中提取结构特征。
- **Spatial-aware Recurrent Unit：** 维持时空记忆，补偿感知盲区。
- **深度传感器仿真模型：** 缩小 sim-to-real 的感知差距。
- **奖励：** 3D waypoint 引导的主动感知奖励，并惩罚镂空缝隙与楼梯边缘。

## 实验与评测

- 真机：Unitree Go2，零样本从仿真迁移。
- 结果：可攀爬坡度最高 55° 的镂空楼梯（据摘要）。

## 与其他工作对比

- 摘要未列出与其他楼梯攀爬方法的数值对比；「首个」为作者声称。

## 结论

本文的核心是用感知结构与记忆来处理镂空楼梯的深度缺陷，而不是只靠更大的策略网络。

- 在稀疏噪声深度下，结构化注意力与空间记忆是关键设计（据摘要）。
- 深度传感器模型是零样本迁移的一部分，不能省略。
- 证据来自摘要与单平台（Go2）真机部署，其他机器人尚未验证。

## 工程实践

- 项目页：[sivan666666.github.io/StairMaster](https://sivan666666.github.io/StairMaster/)。
- 项目页 GitHub 仓库 `Sivan666666/StairMaster` 是学术项目页模板，**未包含训练或部署代码、权重或数据**（已核实：仓库仅含网页文件）。
- 因此复现需依赖论文描述，代码公开状态为「未开源」。

## 局限与风险

- 代码与权重未公开，无法直接复现。
- 摘要中的「55°」来自单一平台（Go2）；推测在更高重心或不同足型上需重新评估。
- 推测：依赖的深度传感器模型需要与真机传感器参数匹配，参数偏差会影响迁移。

## 关联页面

- [Sim2Real 知识链](../overview/hub-sim2real.md) — 仿真到真机迁移的知识背景
- [Sim2Real 概念](../concepts/sim2real.md) — 深度传感器仿真与零样本迁移
- [机器人视觉感知栈选型闭环](../queries/robot-perception-stack-selection-loop.md) — ①层「有深度图 ≠ 深度处处可信」：本文用深度传感器仿真与空间记忆应对镂空楼梯处稀疏、噪声深度；④层策略直接消费深度

## 参考来源

- [StairMaster 论文来源归档](../../sources/papers/stairmaster_arxiv_2606_25765.md)

## 推荐继续阅读

- [arXiv 2606.25765](https://arxiv.org/abs/2606.25765)
- [项目页](https://sivan666666.github.io/StairMaster/)
