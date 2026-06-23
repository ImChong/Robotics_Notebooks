---
type: concept
tags: [robotics, urdf, kinematics, dynamics, modeling, ros]
status: complete
updated: 2026-06-23
related:
  - ../entities/pinocchio.md
  - ../entities/mujoco.md
  - ../formalizations/articulated-body-algorithms.md
  - ./floating-base-dynamics.md
  - ./robot-link-and-rotor-inertia.md
  - ../entities/urdf-studio.md
  - ../entities/quadruped-control-curriculum.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
summary: "URDF 是 ROS 生态统一的机器人连杆-关节-惯量描述格式；四足课程从 17-link 树解析入手，理解 n_q 与 n_v 差异是动力学编程的前提。"
---

# URDF（统一机器人描述格式）

**URDF（Unified Robot Description Format）** 是用 XML 描述机器人 **连杆几何、关节类型、惯量与碰撞体** 的标准格式，是 ROS、MuJoCo、Pinocchio 等栈的 **共同建模入口**。

## 一句话定义

> 一份 XML 把「机器人长什么样、关节怎么连、质量惯量多少」说清楚，供仿真、控制和可视化共用。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| URDF | Unified Robot Description Format | 统一机器人描述格式 |
| MJCF | MuJoCo XML Format | MuJoCo 原生模型格式，常与 URDF 互转 |
| SDF | Simulation Description Format | Gazebo 等使用的仿真描述格式 |
| FK | Forward Kinematics | 由关节角求连杆位姿 |
| CoM | Center of Mass | 质心，`<inertial>` 标签核心量 |
| ROS | Robot Operating System | URDF 最常随 ROS 包分发 |
| SDK | Software Development Kit | 实机厂商常提供 URDF + SDK 联调 |

## 为什么重要

四足课程 Ch2 从 **17 个 link 的运动学树** 出发：不懂 URDF，就无法理解为何四足浮动基有 **$n_q=19$（含四元数姿态）而 $n_v=18$（角速度 3 维）**，也无法正确调用 ABA/RNEA。

## 核心结构

| 元素 | 含义 |
|------|------|
| `<link>` | 刚体：visual / collision / inertial |
| `<joint>` | 连接两 link：revolute、fixed、continuous 等 |
| `<inertial>` | 质量、质心、惯性张量 |
| 运动学树 | 单父节点树；四足通常 12 驱动关节 + 浮动基 |

### 四足典型参数规模

课程 Ch3 强调：一条四足 URDF 可有 **~200+ 标量参数**（几何、惯量、关节限位、摩擦等），其中大量 **不可辨识或弱可观**，SysID 需筛选。

## 工程工作流

1. 厂商提供 URDF（如 Unitree / Zsibot）
2. 导入 [Pinocchio](../entities/pinocchio.md) / MuJoCo / MATRiX
3. 核对惯量、关节轴向与实机一致
4. SysID 修正关键参数（摩擦、转子惯量）

工具：[URDF Studio](../entities/urdf-studio.md)、[step2urdf](../entities/step2urdf.md)

## 常见误区

- **误区：「URDF 够准就能 Sim2Zero」。** 执行器动力学、间隙、柔性不在 URDF 标准字段里。
- **误区：「四元数姿态占 4 维所以 $n_q=n_v$」。** 姿态用四元数存储但速度在切空间，维数不同。

## 关联页面

- [Articulated Body Algorithms](../formalizations/articulated-body-algorithms.md)
- [Floating Base Dynamics](./floating-base-dynamics.md)
- [System Identification](./system-identification.md)
- [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)

## 推荐继续阅读

- ROS 文档：[URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [Pinocchio Quick Start](../queries/pinocchio-quick-start.md)

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch2–Ch3 URDF 与参数评估
