---
type: entity
tags: [software, simulation, optimization, trajectory-optimization, c++]
status: complete
updated: 2026-04-21
related:
  - ../methods/trajectory-optimization.md
  - ../concepts/whole-body-control.md
  - ../tasks/locomotion.md
  - ./mujoco.md
sources:
  - ../../sources/papers/simulation.md
summary: "Drake 是由丰田研究院（TRI）主导开发的开源 C++ 机器人工具箱，以其在轨迹优化（直接配点法）和严谨动力学建模方面的统治力而闻名。"
---

# Drake (机器人工具箱)

**Drake** 是由丰田研究院（Toyota Research Institute, TRI）主导开发，由 Russ Tedrake（MIT 教授）团队深度参与的核心开源机器人软件库。它并非单纯的物理引擎，而是一个包含了动力学计算、系统仿真、控制设计、尤其是**轨迹优化（Trajectory Optimization）**的庞大 C++ 工具箱。

## 核心特性

1. **为优化而生 (Optimization-first)**：
   Drake 最为学界推崇的功能是它的非线性规划（NLP）构建能力。它提供了极其优雅的 C++（和 Python）API 来将动力学方程自动转换为优化约束。它是执行**直接配点法 (Direct Collocation)** 和**接触隐式优化 (Contact-Implicit Optimization)** 的首选框架。
2. **严谨的多体动力学**：
   相比于许多为了游戏或渲染而生的引擎，Drake 的多体动力学建模具有极高的学术严谨性，严格区分广义坐标、速度和加速度，非常适合推导解析雅可比和海森矩阵。
3. **系统架构 (Systems Framework)**：
   它受到 Simulink 的启发，提供了一套积木式的 System API。你可以将控制器、传感器、环境动力学封装为一个个独立的 System 模块，连线闭环后再统一进行仿真或优化。

## 适用场景

- 双足/四足机器人翻跟头、跑酷等极限动作的离线轨迹优化。
- 机械臂运动规划与闭环控制算法原型的开发。
- 对物理接触精度和优化可导性有极高要求的学术研究。

## 关联页面
- [Trajectory Optimization](../methods/trajectory-optimization.md)
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [MuJoCo 物理引擎](./mujoco.md)
- [Locomotion](../tasks/locomotion.md)

## 参考来源
- Drake 官方文档 (drake.mit.edu).
- Tedrake, R. *Underactuated Robotics*.
