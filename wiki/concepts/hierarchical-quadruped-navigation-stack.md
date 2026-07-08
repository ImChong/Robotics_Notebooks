---
type: concept
tags: [quadruped, navigation, hierarchical-control, vln, reinforcement-learning, system-integration]
status: complete
updated: 2026-06-23
related:
  - ../entities/roamerx-navigation.md
  - ../entities/matrix-simulation-platform.md
  - ../tasks/vision-language-navigation.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ../methods/hipan.md
  - ../methods/ppo.md
  - ../entities/quadruped-control-curriculum.md
sources:
  - ../../sources/courses/quadruped_control_simulation_rl_curriculum.md
summary: "四足自主系统常采用 VLN→导航→RL loco→PD→硬件 分层栈：高层语义与路径规划与底层步态策略解耦，RoamerX + MATRiX 是课程集成范例。"
---

# Hierarchical Quadruped Navigation Stack（四足分层导航栈）

**四足分层导航栈** 将 **语义/语言目标、全局路径、局部运动、关节力矩** 拆成多层模块，避免「一个端到端网络从像素直接到电机」的工程不可控性。

## 一句话定义

> **上层决定去哪、中层决定往哪走、下层决定怎么迈步** —— 层间用速度指令、航向或步态参数接口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLN | Vision-Language Navigation | 视觉-语言导航，自然语言目标 |
| SLAM | Simultaneous Localization and Mapping | 定位与建图 |
| RL | Reinforcement Learning | 中层以下常负责崎岖地形步态 |
| PD | Proportional–Derivative | 最底层关节跟踪 |
| BT | Behavior Tree | 导航任务编排 |
| MPPI | Model Predictive Path Integral | 局部路径跟踪控制器 |
| cmd_vel | Command Velocity | 导航层典型输出（$v_x, v_y, \omega$） |

## 课程六层栈（Ch7）

```
┌─────────────────────────────────────┐
│  L1  VLN — 自然语言 → 目标/航点      │
├─────────────────────────────────────┤
│  L2  Navigation — SLAM + 全局规划    │  ← RoamerX
├─────────────────────────────────────┤
│  L3  RL Locomotion — 步态/姿态跟踪   │  ← PPO 策略
├─────────────────────────────────────┤
│  L4  PD — 关节伺服                   │
├─────────────────────────────────────┤
│  L5  Hardware / SDK — 电机驱动       │
└─────────────────────────────────────┘
```

**数据流**：语言指令 → 语义目标点 → 全局路径 → **平面速度 + 体姿态** → RL 策略调制步态 → PD 力矩 → 电机。

## 为什么分层

| 若合并 | 问题 |
|--------|------|
| VLN 直接输出关节角 | 样本极少、安全难保证、难复用 loco 策略 |
| 导航不管足式动力学 | 轮式假设在台阶/碎石失效 |
| RL 做全局规划 | 信用分配极长，训练不稳定 |

分层后：**loco 策略可在仿真大规模预训练**，导航栈可独立迭代 SLAM/规划。

## Final Project 4 闭环

课程要求：在 [MATRiX](../entities/matrix-simulation-platform.md) 中整合 [RoamerX](../entities/roamerx-navigation.md) 与 RL 运动策略，实现 **目标点 → 自主导航 → 到达**，含消融与失败分析。

## 与 HiPAN 等工作的关系

[HiPAN](../methods/hipan.md) 同样采用 **深度高层 + 足式低层 + teacher–student**，侧重 3D 非结构化场景；本栈偏 **工程 ROS2 导航 + 智身生态**。

## 常见误区

- **导航 cmd_vel 频率过低**：足式跟踪需 RL 层 **高频** 适应局部地形。
- **忽略体姿态维度**：四足除平面速度外常需 **roll/pitch 或 body height** 指令给 RL 层。

## 关联页面

- [Vision-Language Navigation](../tasks/vision-language-navigation.md)
- [Navigation SLAM Autonomy Stack](../overview/navigation-slam-autonomy-stack.md)
- [Quadruped Control Curriculum](../entities/quadruped-control-curriculum.md)
- [Gait Generation](./gait-generation.md)
- [具身大模型分类学选型闭环（专题枢纽）](../overview/topic-embodied-foundation-model.md) — 分层导航栈对应五层闭环的 VLN 空间导航层

## 推荐继续阅读

- [RoamerX GitHub](https://github.com/zsibot/genisom_roamerx_open)
- [HiPAN](../methods/hipan.md) — 分层导航学术对照

## 参考来源

- [sources/courses/quadruped_control_simulation_rl_curriculum.md](../../sources/courses/quadruped_control_simulation_rl_curriculum.md) — 课程 Ch7–Ch8 与 Project 4
