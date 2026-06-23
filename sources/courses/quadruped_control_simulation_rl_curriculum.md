# 四足机器人：从动力学建模到强化学习 — 课程大纲

- **类型：** course
- **来源：** 具身智能研究室（微信公众号）课程大纲整理；课程由深蓝学院 × 纽卡斯尔大学潘为教授 × 智身科技联合推出，全程基于 MATRiX 平台
- **收录日期：** 2026-06-23
- **一句话说明：** 以 URDF/浮动基动力学、系统辨识、PPO 并行训练、域随机化、Sim2Real 与分层导航为主线，完整拆解四足从仿真到实机的闭环。

## 为什么值得保留

- 把 **PID → MPC → RL** 控制演进、**可微仿真**、**SysID**、**PPO + DR + 课程学习**、**RMA 蒸馏**、**VLN + 导航栈** 串成一条可执行工程路线。
- 四个 Project（动力学验证、越障步态、摩擦补偿部署、自主导航闭环）可直接映射本库任务页与方法页。
- 与 [IROS 2025 四足挑战赛](https://ai.zhiding.cn/2025/1023/3173033.shtml) 冠军实战叙事对齐，强调 **15 分钟现场重训** 与 **MATRiX → 实机** 工具链。

## 章节大纲（8 章 + 4 Project）

### 第 1 章 引言：四足控制演进与可微仿真

| 节 | 主题 |
|----|------|
| 1.1 | 控制技术演进：PID → MPC → RL |
| 1.2 | MPC 在碎石等复杂地形的局限 |
| 1.3 | 可微仿真作为 MPC 与 RL 之间的技术桥梁 |
| 1.4 | MATRiX 平台：可微、GPU 并行、跨平台 |

### 第 2 章 机器人建模：URDF 与刚体动力学

| 节 | 主题 |
|----|------|
| 2.1 | URDF 解析与 17-link 运动学树 |
| 2.2 | 四元数与位形流形（$n_q=19$ vs $n_v=18$） |
| 2.3 | 浮动基座运动方程与欠驱动特性 |
| 2.4 | ABA / RNEA 与李群积分 |
| **Project 1** | 仿真环境搭建与动力学验证：加载 ZSL-1、SDK 遥控、计算 $M$ 与 $g$、关节传感器可视化 |

### 第 3 章 模型精度与系统辨识

| 节 | 主题 |
|----|------|
| 3.1 | 206 个 URDF 参数分类与评估 |
| 3.2 | 关节摩擦：Coulomb / Viscous / Stribeck |
| 3.3 | 转子惯量（膝部转子惯量可达 link 百倍） |
| 3.4 | 经典回归 vs 可微辨识（`jax.grad`） |

### 第 4 章 强化学习与运动控制（上）

| 节 | 主题 |
|----|------|
| 4.1 | PPO 与 clipped ratio 直觉 |
| 4.2 | MATRiX 4096 并行环境 vs Isaac Gym |
| 4.3 | 48 维观测向量设计 |
| 4.4 | 奖励：速度跟踪、Trot 步态约束、能耗 |

### 第 5 章 强化学习与运动控制（下）

| 节 | 主题 |
|----|------|
| 5.1 | 域随机化：对参数分布优化期望 |
| 5.2 | 五阶段课程学习 |
| 5.3 | 程序化地形：坡、碎石、台阶、随机凸起 |
| **Project 2** | 越障行走策略：奖励设计、PPO 并行训练、≥3 地形泛化、有/无 DR 对比 |

### 第 6 章 Sim-to-Real 迁移

| 节 | 主题 |
|----|------|
| 6.1 | Sim2Real gap 系统分解（来源、严重度、对策） |
| 6.2 | 关节摩擦前馈补偿 |
| 6.3 | RMA 范式下的 Teacher–Student 蒸馏 |
| 6.4 | PD 增益整定与实机安全协议 |
| **Project 3** | 策略部署与摩擦补偿：SDK 底层接口、无补偿 / 补偿 / 补偿+DR 三组对比 |

### 第 7 章 感知引导导航与系统集成

| 节 | 主题 |
|----|------|
| 7.1 | 六层栈：VLN → Navigation → RL → PD → Hardware |
| 7.2 | RoamerX：LiDAR SLAM 与路径规划 |
| 7.3 | VLN：自然语言 → 运动指令 |
| 7.4 | 跳跃力估计与实现 |

### 第 8 章 实机部署与案例分析

| 节 | 主题 |
|----|------|
| 8.1 | IROS 2025 四足挑战赛冠军案例 |
| 8.2 | 现场「15 分钟重训」应急工程 |
| 8.3 | 四足 → 人形控制技术路径展望 |
| **Project 4** | 自主系统闭环：RoamerX + RL 运动策略，目标点 → 导航 → 到达 + 消融 |

## 对 wiki 的映射

| 课程主题 | wiki 页面 |
|---------|-----------|
| 课程总览 | [quadruped-control-curriculum](../../wiki/entities/quadruped-control-curriculum.md) |
| MATRiX 平台 | [matrix-simulation-platform](../../wiki/entities/matrix-simulation-platform.md) |
| RoamerX 导航 | [roamerx-navigation](../../wiki/entities/roamerx-navigation.md) |
| 可微仿真 | [differentiable-simulation](../../wiki/concepts/differentiable-simulation.md) |
| URDF | [urdf-robot-description](../../wiki/concepts/urdf-robot-description.md) |
| ABA / RNEA | [articulated-body-algorithms](../../wiki/formalizations/articulated-body-algorithms.md) |
| 关节摩擦模型 | [joint-friction-models](../../wiki/concepts/joint-friction-models.md) |
| 摩擦补偿 | [friction-compensation](../../wiki/concepts/friction-compensation.md) |
| PID | [pid-control](../../wiki/methods/pid-control.md) |
| MPC | [model-predictive-control](../../wiki/methods/model-predictive-control.md) |
| PPO | [ppo](../../wiki/methods/ppo.md) |
| 域随机化 | [domain-randomization](../../wiki/concepts/domain-randomization.md) |
| 课程学习 | [curriculum-learning](../../wiki/concepts/curriculum-learning.md) |
| 程序化地形 | [procedural-terrain-generation](../../wiki/concepts/procedural-terrain-generation.md) |
| Sim2Real | [sim2real](../../wiki/concepts/sim2real.md) |
| RMA / 蒸馏 | [privileged-training](../../wiki/concepts/privileged-training.md)、[paper-rma](../../wiki/entities/paper-rma-rapid-motor-adaptation.md) |
| 分层导航栈 | [hierarchical-quadruped-navigation-stack](../../wiki/concepts/hierarchical-quadruped-navigation-stack.md) |
| VLN | [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md) |
| 步态 / Trot | [gait-generation](../../wiki/concepts/gait-generation.md) |
| 系统辨识 | [system-identification](../../wiki/concepts/system-identification.md) |
| 浮动基动力学 | [floating-base-dynamics](../../wiki/concepts/floating-base-dynamics.md) |
| 李群 / 四元数 | [lie-group-rigid-body-motions](../../wiki/formalizations/lie-group-rigid-body-motions.md) |
