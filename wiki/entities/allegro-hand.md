---
type: entity
tags: [hardware, dexterity, manipulation, robot-hand, research]
status: complete
updated: 2026-04-21
related:
  - ../tasks/manipulation.md
  - ../concepts/tactile-sensing.md
  - ../methods/in-hand-reorientation.md
  - ../queries/dexterous-data-collection-guide.md
sources:
  - ../../sources/papers/humanoid_hardware.md
summary: "Allegro Hand 是一款轻量化的四指灵巧手平台，拥有 16 个自由度，因其结构简单、API 友好且性价比高而成为机器人学习与灵巧操作研究的主流科研平台。"
---

# Allegro Hand (灵巧手)

**Allegro Hand** 是由 Wonik Robotics 开发的一款高性能四指灵巧手（Dexterous Hand）。它在机器人科研界（特别是强化学习和模仿学习领域）享有极高的普及率，被视为研究灵巧操作、手内重定向和多指触觉反馈的标准硬件平台。

## 硬件规格

- **手指数量**：4 指（3 个主要手指 + 1 个对向拇指）。
- **自由度 (DOFs)**：16 个（每根手指 4 个主动驱动关节）。
- **负载能力**：约 5kg。
- **通讯方式**：CAN 总线，支持高频（> 300Hz）控制。
- **兼容性**：提供官方 ROS/ROS2 驱动及 Python API，易于与主流仿真器（MuJoCo, Isaac Gym）集成。

## 为什么在科研中流行

1. **结构精简**：相比于拥有 20+ 自由度的 Shadow Hand，Allegro Hand 放弃了无名指和小指，将复杂性降到了“足以模拟人类大部分抓取动作”的临界点。
2. **易于建模**：它的运动学结构规整，DH 参数稳定，非常适合进行精确的运动学求逆和动力学仿真。
3. **开源友好**：社区贡献了大量的 URDF 模型和强化学习环境（如 DexDex, HandLib），研究者可以实现“算法跨实验室迁移”。
4. **易于集成触觉**：其指尖和指节位置预留了空间，方便安装 GelSight 或 Xela 等第三方触觉传感器。

## 代表性研究工作

- **Dexterous Manipulation from Vision**：利用 Allegro Hand 在 MuJoCo 中通过强化学习训练大规模灵巧操作技能。
- **Haptics-based Grasping**：在 Allegro Hand 上安装触觉阵列，实现对形状不规则物体的盲抓（Blind Grasping）。
- **In-hand Reorientation**：展示如何通过协调 16 个关节，让手中的魔方或笔在不掉落的前提下实现 360 度翻转。

## 关联页面
- [Manipulation 任务](../tasks/manipulation.md)
- [Dexterous Kinematics (灵巧手运动学)](../concepts/dexterous-kinematics.md) — 灵巧操作的数学基础
- [Tactile Sensing (触觉感知)](../concepts/tactile-sensing.md)
- [手内重定向 (In-hand Reorientation)](../methods/in-hand-reorientation.md)
- [Query: 灵巧手数据采集指南](../queries/dexterous-data-collection-guide.md)

## 参考来源
- Wonik Robotics Official Documentation.
- [sources/papers/humanoid_hardware.md](../../sources/papers/humanoid_hardware.md)
