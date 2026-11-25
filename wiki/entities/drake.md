---

type: entity
tags: [software, simulation, optimization, trajectory-optimization, c++, toyota-research, mit]
status: complete
updated: 2026-11-21
original: ../../sources/sites/drake_official_overview.md
related:
  - ../methods/trajectory-optimization.md
  - ../concepts/whole-body-control.md
  - ../tasks/locomotion.md
  - ./painode-074-mitunderactuatedrobotics.md
  - ./mujoco.md
  - ./adams.md
sources:
  - ../../sources/sites/drake_official_overview.md
  - ../../sources/courses/mit_underactuated_robotics.md
  - ../../sources/papers/simulation.md
summary: "Drake 是由丰田研究院（TRI）主导开发的开源 C++ 机器人工具箱，以其在轨迹优化（直接配点法）和严谨动力学建模方面的统治力而闻名。"
---

# Drake (机器人工具箱)

**Drake** 是由丰田研究院（Toyota Research Institute, TRI）主导开发，由 Russ Tedrake（MIT 教授）团队深度参与的核心开源机器人软件库。它并非单纯的物理引擎，而是一个包含了动力学计算、系统仿真、控制设计、尤其是**轨迹优化（Trajectory Optimization）**的庞大 C++ 工具箱。

## 一句话定义

**Drake（Model-Based Design and Verification for Robotics）** 是暴露解析梯度、稀疏控制结构与多项式方程的系统级动力学建模与数值优化工具箱，与黑箱物理仿真器相对，专为规划、最优控制与鲁棒性验证打造。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| API | Application Programming Interface | 应用程序编程接口 |
| WBC | Whole-Body Control | 协调全身关节满足多任务/约束的控制基础设施 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理仿真引擎 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 为什么重要

- **结构透明而非黑箱仿真**：官网明确区别于常见“输指令出观测”的黑箱引擎，Drake 在仿真摩擦、接触、空气动力学时始终保留解析梯度、稀疏性与多体力学结构，使直接配点、SOS Lyapunov 验证与凸松弛成为一等公民。
- **工程验证与学术桥梁**：作为 MIT *Underactuated Robotics* 与 *Robotic Manipulation* 课程主框架，用统一 `MathematicalProgram` 衔接开源（OSQP、IPOPT）与商业求解器（SNOPT、MOSEK、Gurobi）。

## 核心原理与方法栈（核心特性）

1. **为优化而生 (Optimization-first)**：
   Drake 最为学界推崇的功能是它的非线性规划（NLP）构建能力。它提供了极其优雅的 C++（和 Python）API 来将动力学方程自动转换为优化约束。它是执行**直接配点法 (Direct Collocation)** 和**接触隐式优化 (Contact-Implicit Optimization)** 的首选框架。
2. **严谨的多体动力学**：
   相比于许多为了游戏或渲染而生的引擎，Drake 的多体动力学建模具有极高的学术严谨性，严格区分广义坐标、速度和加速度，非常适合推导解析雅可比和海森矩阵。
3. **系统架构 (Systems Framework)**：
   它受到 Simulink 的启发，提供了一套积木式的 System API。你可以将控制器、传感器、环境动力学封装为一个个独立的 System 模块，连线闭环后再统一进行仿真或优化。

## 工程实践

- **C++ 文档查签名 + `pydrake` 快速原型**：MIT *Underactuated Robotics* App. A（`drake.html`）指出 `pydrake` 自动生成文档较 C++ 滞后，建议先看 C++ API 文档定位接口与约束签名，再回到 Python 核对绑定名称。
- **规划求解器匹配**：运动学/凸轨迹优化用 `MathematicalProgram` 默认开源路径（OSQP / Clarabel / IPOPT）；大规模直接配点与光滑接触约束推荐搭配 SNOPT 或学术免费授权的 MOSEK / Gurobi。

## 常见误区或局限

- **不是万向并行 GPU RL 引擎**：Drake 定位于精准白箱多体、几何碰撞与数学规划，在大规模并行 RL 采样吞吐上不如 Isaac Lab / MuJoCo MJX。
- **依赖特定商业求解器释放全部实力**：复杂直接配点与半定规划/SOS 验证高度依赖 SNOPT 或 MOSEK 学术授权。

## 适用场景

- 双足/四足机器人翻跟头、跑酷等极限动作的离线轨迹优化。
- 机械臂运动规划与闭环控制算法原型的开发。
- 对物理接触精度和优化可导性有极高要求的学术研究。

## 关联页面
- [Trajectory Optimization](../methods/trajectory-optimization.md)
- [Whole-Body Control (WBC)](../concepts/whole-body-control.md)
- [MuJoCo 物理引擎](./mujoco.md)
- [ADAMS（工业多体动力学谱系）](./adams.md) — 稀疏约束 DAE / 工业虚拟样机对照；非 RL 训练栈
- [Locomotion](../tasks/locomotion.md)

## 参考来源
- [drake_official_overview.md](../../sources/sites/drake_official_overview.md) — Drake 官网（drake.mit.edu）与 MIT Underactuated App. A 原文
- [mit_underactuated_robotics.md](../../sources/courses/mit_underactuated_robotics.md) — Tedrake, R. *Underactuated Robotics*（Ch. 8/10/17/22）
- [simulation.md](../../sources/papers/simulation.md) — 仿真引擎总览占位索引

## 推荐继续阅读

- [Trajectory Optimization](../methods/trajectory-optimization.md) — Drake `DirectCollocation` 与 `MathematicalProgram` 理论背景
- [MIT Underactuated Robotics 知识节点](./painode-074-mitunderactuatedrobotics.md) — 配套课程全脉络
- [MuJoCo 物理引擎](./mujoco.md) — 接触仿真速度与强化学习生态对照
