---
type: entity
title: SAPIEN (仿真引擎)
tags: [simulation, physics-engine, manipulation, sapien]
---

# SAPIEN (仿真引擎)

**SAPIEN** (A Scannable Articulated Part Engine) 是一个专门针对**关节体（Articulated Objects）**交互和机器人操作设计的高性能物理仿真引擎。它由 UCSD 的 Su Lab 开发，是目前具身智能研究中处理物体细粒度交互（如开门、拉抽屉）的主流选择。

## 为什么重要？

在具身智能中，机器人需要学习如何与人类环境中的复杂物体交互。SAPIEN 的优势在于：
- **PartNet-Mobility 支持**：它是第一个大规模支持 PartNet-Mobility 数据集的引擎，该数据集包含了数千个具有真实物理关节的 3D 模型。
- **物理真实感**：相比传统的机器人仿真器，SAPIEN 在处理多体动力学和接触力学方面做了深度优化，特别适合操作（Manipulation）任务。
- **渲染与感知**：支持多机渲染，能够为训练提供高质量的视觉输入。

## 核心特性

- **ROS 集成**：原生支持 ROS 接口，方便控制算法的快速迁移。
- **并行仿真**：支持在大规模 GPU 集群上并行运行，加速强化学习训练。
- **灵活的 API**：提供 Python 和 C++ 接口，易于扩展。

## 与其他系统的关系

- **上层应用**：[[robotwin]] 2.0 建立在 SAPIEN 之上，用于自动化数据生成。
- **同类对比**：相比 [[mujoco]]，SAPIEN 更侧重于物体交互和视觉感知；相比 [[isaac-gym-isaac-lab]]，它在处理部件级关节体模型方面具有独特的生态优势；相比 [[genesis-sim]]，后者在多物理场（流体、柔性体）耦合方面更为先进。

## 参考来源
- [Embodied-AI-Guide](../../sources/repos/embodied-ai-guide.md)
- [SAPIEN Project Page](https://sapien.ucsd.edu/)
