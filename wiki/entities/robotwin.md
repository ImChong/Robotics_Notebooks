---
type: entity
title: RoboTwin 2.0
tags: [simulation, data-generation, dual-arm, dataset]
---

# RoboTwin 2.0

**RoboTwin 2.0** 是一个专为双臂机器人操作设计的**自动数据生成与仿真平台**。它建立在 [SAPIEN (仿真引擎)](./sapien.md) 仿真引擎之上，旨在解决具身智能（Embodied AI）中高质量专家数据获取昂贵且难以规模化的问题。

## 为什么重要？

在具身智能的训练中，**数据规模化（Scaling）** 是核心瓶颈。RoboTwin 通过以下方式提供解决方案：
- **自动数据合成**：不再依赖人类遥操作，通过脚本或预定义策略生成海量专家轨迹。
- **双臂任务聚焦**：针对当前最热门的双臂操作（如 [ALOHA](./aloha.md) 任务）提供深度支持。
- **低门槛上手**：作为 Lumina 社区《具身智能百科全书》推荐的实践平台，提供了从数据采集到模型训练的全链路工具。

## 核心特性

- **基于 SAPIEN 引擎**：利用其优秀的物理仿真能力和对 PartNet-Mobility 数据集的支持。
- **任务库**：内置了 50+ 个双臂自动化任务，覆盖了常见的家庭和工业操作场景。
- **真机对齐**：强调 Sim-to-Real 的一致性，生成的轨迹可以直接用于训练并在真机上验证。

## 与其他系统的关系

- **底层驱动**：依赖 [SAPIEN (仿真引擎)](./sapien.md) 进行物理模拟。
- **任务目标**：通常用于生成 [[behavior-cloning]] 或 [[action-chunking]] (ACT) 所需的训练数据。
- **硬件对应**：其仿真场景常模拟 [ALOHA](./aloha.md) 或类似的双臂遥操作设备。

## 参考来源
- [Embodied-AI-Guide](../../sources/repos/embodied-ai-guide.md)
- [RoboTwin 官方仓库](https://github.com/msc-robotwin/robotwin)
