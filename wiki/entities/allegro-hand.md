---

type: entity
tags: [hardware, dexterity, manipulation, robot-hand, research, wonik-robotics]
status: complete
updated: 2026-11-15
related:
  - ../tasks/manipulation.md
  - ../concepts/tactile-sensing.md
  - ../methods/in-hand-reorientation.md
  - ../queries/dexterous-data-collection-guide.md
  - ./gel-slim.md
  - ./wuji-robotics.md
  - ./kyber-labs.md
  - ./all-hands-up.md
  - ./paper-adept-dexterity.md
  - ./paper-karma-hand-metric.md
sources:
  - ../../sources/papers/humanoid_hardware.md
  - ../../sources/sites/allhandsup-org.md
summary: "Allegro Hand 是一款轻量化的四指灵巧手平台，拥有 16 个自由度，因其结构简单、API 友好且性价比高而成为机器人学习与灵巧操作研究的主流科研平台。"
---

# Allegro Hand (灵巧手)

**Allegro Hand** 是由 Wonik Robotics 开发的一款高性能四指灵巧手（Dexterous Hand）。它在机器人科研界（特别是强化学习和模仿学习领域）享有极高的普及率，被视为研究灵巧操作、手内重定向和多指触觉反馈的标准硬件平台。

## 一句话定义

Allegro Hand 是 Wonik Robotics 的 16 自由度四指灵巧手，因结构规整、API 友好、仿真生态成熟，成为 RL / IL 灵巧操作研究的常用科研平台。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| API | Application Programming Interface | 应用程序编程接口 |
| CAN | Controller Area Network | 电机/关节常用的现场总线通信协议 |
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| Isaac Gym | NVIDIA Isaac Gym | GPU 并行刚体仿真训练环境 |
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 为什么重要

它是灵巧操作论文中最常见的"标准硬件"之一：同一套 URDF 与环境可在多个实验室复现，使算法结果可比；同时是手内重定向、触觉抓取等方向的默认载体。

## 核心原理（硬件规格）

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
- **ADEPT（NVIDIA/UMich）**：[ADEPT](./paper-adept-dexterity.md) 在 Kuka iiwa7 + Allegro（23 DoF）上 zero-shot 部署 FMB peg insertion（star **5/10**）与 dish placement（**6/10**），pre-train 16 primitive reposing 覆盖 reach/grasp/lift/reorient（arXiv:2608.19182）。

## 工程实践

- 通过 CAN 总线 + 官方 ROS / ROS2 驱动或 Python API 控制，仿真侧常用 MuJoCo、Isaac Gym 的社区 URDF 模型。
- 指尖 / 指节预留空间，可加装 GelSight、Xela 等触觉传感器（见 [Tactile Sensing](../concepts/tactile-sensing.md)）。
- 数据采集常配合视觉遥操作与重定向，选型见 [数据手套 vs 视觉遥操作](../comparisons/data-gloves-vs-vision-teleop.md)。

## 局限与风险

- 仅 4 指（无无名指、小指），无法覆盖需要五指的人手类任务，仿生度低于 [Shadow Hand](./shadow-hand.md)。
- 本页规格（如负载、控制频率）来自厂商文档摘要，具体版本（V4 / V5 Plus 等）参数可能不同，使用前需核对官方手册。
- 本身不带触觉，需第三方加装并自行标定。

## 关联页面
- [Shadow Hand (灵巧手)](./shadow-hand.md) — 仿生程度更高的五指方案
- [Yale OpenHand](./yale-openhand.md) — 低执行器数欠驱动开源手族对照
- [Manipulation 任务](../tasks/manipulation.md)
- [Dexterous Kinematics (灵巧手运动学)](../concepts/dexterous-kinematics.md) — 灵巧操作的数学基础
- [Tactile Sensing (触觉感知)](../concepts/tactile-sensing.md)
- [手内重定向 (In-hand Reorientation)](../methods/in-hand-reorientation.md)
- [GelSlim 实体（薄片化视觉触觉传感器）](./gel-slim.md) — 可装到指尖的视觉触觉传感器
- [Query: 灵巧手数据采集指南](../queries/dexterous-data-collection-guide.md)
- [Kyber Labs](./kyber-labs.md) — 产业侧背驱动五指手与 skills-based AI 路线对照
- [All Hands Up](./all-hands-up.md) — RLWRLD 画廊含 Allegro V5 Plus（16 DoF、背驱、仿真 Kapandji 9/11）
- [ADEPT（论文实体）](./paper-adept-dexterity.md) — Kuka–Allegro 灵巧 RL 预训练 + sim2real FMB
- [KaRMA（运动学指标）](./paper-karma-hand-metric.md) — 16 手榜 **KaRMA-R 最高**（0.335），T 第二

## 参考来源
- Wonik Robotics Official Documentation.
- [sources/papers/humanoid_hardware.md](../../sources/papers/humanoid_hardware.md)
- [All Hands Up 站点归档](../../sources/sites/allhandsup-org.md)

## 推荐继续阅读

- [手内重定向](../methods/in-hand-reorientation.md)
- [Dexterous Kinematics](../concepts/dexterous-kinematics.md)
- [ADEPT](./paper-adept-dexterity.md)
