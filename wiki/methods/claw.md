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

## 核心设计思路

CLAW 的核心理念是：**将数据采集问题转化为算力密集的数据生成问题。**

1. **组合式动作基元**：将行走、下蹲、匍匐、拳击等基础运动模式定义为“原子”，操作者通过简单的网页端键盘交互或时间轴编辑器即可组合出复杂的全身轨迹。
2. **物理一致性校验**：所有生成的关节轨迹都在 MuJoCo 中经过底层规划器校验。这保证了导出的 50Hz 关节序列不仅视觉流畅，且符合机器人的动力学约束。
3. **自动语言标注**：系统内置了一套基于模板的语言注释引擎。它根据当前执行的原子动作模式、运动速度和时间戳，自动生成精准的文本描述（如“加速向左侧走三步并出右拳”）。

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
