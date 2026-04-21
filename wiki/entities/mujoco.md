---
type: entity
tags: [software, simulation, physics-engine, reinforcement-learning, deepmind]
status: complete
updated: 2026-04-21
related:
  - ../comparisons/mujoco-vs-isaac-sim.md
  - ../methods/reinforcement-learning.md
  - ../concepts/sim2real.md
sources:
  - ../../sources/papers/simulation.md
summary: "MuJoCo 是专为生物力学、机器人学开发的高精度物理引擎。开源后成为机器人强化学习的基石，以极佳的接触稳定性和解析优化支持著称。"
---

# MuJoCo (物理引擎)

**MuJoCo (Multi-Joint dynamics with Contact)** 是一款专为机器人、生物力学和控制研究开发的高性能物理引擎。自被 DeepMind 收购并完全开源（Apache 2.0）以来，它已成为机器人强化学习（RL）和控制社区无可争议的基石工具。

## 核心设计理念

不同于面向游戏或视觉特效的引擎（如 PhysX, Havok, Bullet），MuJoCo 是为**严格的控制理论**而生的：

1. **连续时间动力学的精确离散化**：
   MuJoCo 并不使用传统的基于“惩罚冲量”的方法解决碰撞。相反，它将接触和摩擦建模为一个平滑的凸优化问题。这使得即使在极大的时间步长下，系统的能量守恒和接触稳定性依然极佳。
2. **极易求导**：
   其内部状态可以极其方便地进行有限差分求导甚至解析求导，这对于基于梯度的轨迹优化（Trajectory Optimization）和 iLQR 等算法极度友好。

## 对机器人研究的统治力

- **RL 领域的基准测试**：OpenAI Gym 中的连续控制任务（如 HalfCheetah, Ant, Humanoid）几乎全部由 MuJoCo 驱动。它是评价 PPO, SAC 等深度强化学习算法的绝对标准。
- **Sim2Real 的证明**：诸多成功的 Sim2Real 论文（尤其是四足机器人和灵巧手操作领域）都证明了：只要系统辨识和域随机化做得好，在 MuJoCo 中训练的策略可以直接无缝迁移到物理硬件上。

## 优势与局限

- **优势**：
  - 单线程计算性能极高（千赫兹级的闭环仿真毫无压力）。
  - 接触模型非常稳定，很少发生“穿模”或无理的反弹（Explosion）。
  - `mjcf` (XML) 模型描述文件格式严谨且专为机器人设计。
- **局限**：
  - 原生 MuJoCo 在单机多 GPU 大规模并行能力上，逊色于专为此设计的 Isaac Gym（不过 DeepMind 推出的 MuJoCo XLA 正在弥补这一短板）。
  - 对流体、软体（Soft body）和极其复杂的传感器渲染（如高保真相机）支持较弱。

## 关联页面
- [对比：MuJoCo vs Isaac Sim](../comparisons/mujoco-vs-isaac-sim.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [Sim2Real 概念](../concepts/sim2real.md)

## 参考来源
- Todorov, E., Erez, T., & Tassa, Y. (2012). *MuJoCo: A physics engine for model-based control*.
