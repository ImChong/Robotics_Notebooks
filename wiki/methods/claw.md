---
type: method
tags: [data-generation, data-engine, simulation, mujoco, humanoid, unitree-g1, language-alignment]
status: complete
updated: 2026-04-23
related:
  - ./imitation-learning.md
  - ../concepts/motion-retargeting.md
  - ../concepts/foundation-policy.md
  - ../tasks/loco-manipulation.md
  - ./auto-labeling-pipelines.md
sources:
  - ../../sources/blogs/claw_unitree_g1_language_annotated_motion_data.md
summary: "CLAW (Composable Language-Annotated Whole-Body Motion Data Generation) 是一种面向人形机器人的网页交互式数据生成流水线，利用 MuJoCo 仿真和底层规划器自动生成带语言标签且符合物理约束的全身运动数据。"
---

# CLAW (宇树 G1 全身动作数据生成管线)

**CLAW** (Composable Language-Annotated Whole-Body Motion Data Generation) 是一种面向人形机器人的模块化数据生成方案。它通过将复杂的全身运动拆解为可组合的原子动作，并利用底层物理引擎 (MuJoCo) 进行轨迹生成，从而绕过了传统动捕影棚的高昂成本和重定向带来的滑步/穿模问题。

## 主要技术路线

CLAW 的工作流程可以概括为以下三个核心阶段：

1. **动作基元定义（Atomic Motion Primitives）**：通过分析机器人（如 G1）的物理极限，预定义一组基础运动单元。
2. **交互式合成（Interactive Synthesis）**：用户通过网页界面实时调度基元，并在 MuJoCo 仿真中生成符合物理规律的全身轨迹。
3. **自动标注引擎（Auto-annotation Engine）**：基于动作状态的确定性，自动生成与轨迹对齐的语言描述，产出高质量的 (Instruction, Image, Action) 数据对。

## 为什么这一技术重要

在具身基础策略（Foundation Policies）的训练中，**“动作轨迹 + 精准语言标签”** 的配对数据是极度稀缺的。CLAW 提供了以下价值：
- **消除重定向误差**：由于直接在机器人原生资产上生成动作，完全避免了从人类骨骼到机器人骨骼重定向带来的误差。
- **高质量标注**：自动生成的语言标签比人工后期标注更准确、粒度更细，特别适合训练视觉-语言-动作 (VLA) 模型。
- **极佳的可扩展性**：通过增加原子动作库，可以无限扩充数据的多样性（Long-tail Distribution）。

## 在本项目中的角色

CLAW 代表了“数据引擎（Data Engine）”的最新趋势，即利用**合成数据**来解决真实世界数据获取困难的问题。它特别适合为 [Unitree G1](../entities/unitree-g1.md) 平台提供大规模的高质量预训练数据。

## 关联页面
- [模仿学习 (Imitation Learning)](./imitation-learning.md)
- [动作重定向 (Motion Retargeting)](../concepts/motion-retargeting.md)
- [基础策略模型 (Foundation Policy)](../concepts/foundation-policy.md)
- [自动化标注流水线 (Auto-labeling Pipelines)](./auto-labeling-pipelines.md)
- [Unitree G1](../entities/unitree-g1.md)

## 参考来源
- [sources/blogs/claw_unitree_g1_language_annotated_motion_data.md](../../sources/blogs/claw_unitree_g1_language_annotated_motion_data.md)
- [CLAW 项目主页（GitHub / Paper）](https://mp.weixin.qq.com/s/MNwq3k8MiNHMLuleDyFiHw)
