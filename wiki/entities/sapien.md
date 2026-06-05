---
type: entity
title: SAPIEN (仿真引擎)
tags: [simulation, physics-engine, manipulation, sapien]
summary: "SAPIEN 是针对关节体交互与机器人操作设计的高性能仿真引擎，支持大规模 PartNet-Mobility 数据集，适用于细粒度操作任务。"
updated: 2026-05-30
related:
  - ./physx-omni.md
  - ./paper-physforge-physics-grounded-3d-assets.md
  - ./robotwin.md
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

- **上层应用**：[robotwin](robotwin.md) 2.0 建立在 SAPIEN 之上，用于自动化数据生成。
- **资产生成生态**：[PhysX-Omni](physx-omni.md)、[PhysForge](paper-physforge-physics-grounded-3d-assets.md) 等路线产出 **sim-ready 关节/可变形资产** 后，常需与 PartNet-Mobility 系引擎（含 SAPIEN）核对 **关节轴、碰撞与 URDF** 一致性。
- **同类对比**：相比 [mujoco](mujoco.md)，SAPIEN 更侧重于物体交互和视觉感知；相比 [isaac-gym-isaac-lab](isaac-gym-isaac-lab.md)，它在处理部件级关节体模型方面具有独特的生态优势；相比 [genesis-sim](genesis-sim.md)，后者在多物理场（流体、柔性体）耦合方面更为先进。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |
| GPU | Graphics Processing Unit | 图形处理器，大规模并行仿真训练的算力基础 |
| API | Application Programming Interface | 应用程序编程接口 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| AI | Artificial Intelligence | 人工智能 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |

## 参考来源
- [Embodied-AI-Guide](../../sources/repos/embodied-ai-guide.md)
- [SAPIEN Project Page](https://sapien.ucsd.edu/)
- [sources/papers/physforge_arxiv_2605_05163.md](../../sources/papers/physforge_arxiv_2605_05163.md) — PhysForge：VLM 物理蓝图 + KineVoxel 扩散合成关节资产，PhysDB 物理标注与 SAPIEN 等仿真平台互链
- [sources/papers/physx_omni_arxiv_2605_21572.md](../../sources/papers/physx_omni_arxiv_2605_21572.md) — PhysX-Omni：统一 sim-ready 物理 3D 生成；代码致谢 PartNet-mobility / SAPIEN 生态
