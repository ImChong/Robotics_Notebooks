---
type: entity
tags: [simulation, physics-engine, reinforcement-learning, pybullet, python]
status: complete
updated: 2026-06-16
related:
  - ./gym-pybullet-drones.md
  - ./motion-imitation-quadruped.md
  - ./mujoco.md
  - ./isaac-gym-isaac-lab.md
  - ../concepts/embodied-rl-minimal-closed-loop.md
  - ../methods/reinforcement-learning.md
  - ../comparisons/mujoco-vs-isaac-sim.md
sources:
  - ../../sources/blogs/wechat_shenlan_rl_embodied_minimal_closed_loop.md
summary: "PyBullet 是基于 Bullet 物理引擎的 Python 开源仿真器：URDF 加载、关节驱动、碰撞与传感器接口轻量，常用于 RL 入门闭环、四足模仿与课程实验；精细接触与人形大规模并行不如 MuJoCo / Isaac Lab。"
---

# PyBullet

**PyBullet**（[bulletphysics/pybullet](https://github.com/bulletphysics/pybullet)）把 **Bullet** 刚体物理引擎封装为 **Python API**，是机器人 RL 领域最常见的 **轻量级入门仿真器** 之一：几分钟内即可加载 URDF、设置重力、在循环里读状态、写电机指令。

## 一句话定义

用 Python 直接 `loadURDF` + `stepSimulation` 跑通 **状态–动作–奖励–物理转移** 闭环，而不必先搭 ROS 或 GPU 仿真农场。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| URDF | Unified Robot Description Format | PyBullet 默认机器人模型格式 |
| RL | Reinforcement Learning | 常与 PyBullet 联用做策略学习 |
| PD | Proportional–Derivative | `POSITION_CONTROL` / `VELOCITY_CONTROL` 底层跟踪 |
| GUI | Graphical User Interface | `p.connect(p.GUI)` 可视化调试 |
| DDS | Direct Drive Simulation | PyBullet 直连电机模式之一 |

## 为什么重要

- **教学与最小闭环**：在进 Isaac Lab / MuJoCo 大规模训练前，用 KUKA 臂定点、倒立摆等 **几十行脚本** 把 MDP 五元组与仿真步进对齐（见 [具身 RL 最小闭环](../concepts/embodied-rl-minimal-closed-loop.md)）。
- **历史生态**：[motion_imitation](./motion-imitation-quadruped.md)（四足模仿动物）、早期 DeepMimic 复现、[gym-pybullet-drones](./gym-pybullet-drones.md)（四旋翼 RL）均建立于 PyBullet。
- **依赖轻**：`pip install pybullet` 即可，适合笔记本与课程环境。

## 核心能力

| 能力 | API 直觉 |
|------|----------|
| 加载场景 | `loadURDF("plane.urdf")`、机器人 `loadURDF(...)` |
| 物理步进 | `stepSimulation()` 推进一个仿真 tick |
| 读状态 | `getLinkState` / `getJointState` → 末端位姿、关节角 |
| 写动作 | `setJointMotorControl2`（位置 / 速度 / 力矩模式） |
| 可视化 | `p.connect(p.GUI)` 或 `DIRECT` 无头 |

## 局限与选型

| 场景 | PyBullet | 更常选 |
|------|----------|--------|
| RL 入门、机械臂 reach 教学 | ✅ 足够 | — |
| 四旋翼 RL 基准 | ✅ [gym-pybullet-drones](./gym-pybullet-drones.md) | Flightmare / Isaac |
| 人形万环境并行 PPO | ⚠️ 慢 | [Isaac Lab](./isaac-gym-isaac-lab.md) |
| 精细接触 / 足端力可信 | ⚠️ 简化 | [MuJoCo](./mujoco.md) |
| sim2real 精细接触操作 | ❌ 不宜唯一真理源 | MuJoCo + 系统辨识 |

## 关联页面

- [具身 RL 最小闭环](../concepts/embodied-rl-minimal-closed-loop.md) — KUKA 定点任务与 MDP 要素对照
- [motion_imitation](./motion-imitation-quadruped.md) — 四足 PyBullet 模仿标杆
- [gym-pybullet-drones](./gym-pybullet-drones.md) — 四旋翼 Gymnasium 环境
- [Reinforcement Learning](../methods/reinforcement-learning.md)
- [MuJoCo vs Isaac Sim](../comparisons/mujoco-vs-isaac-sim.md)

## 参考来源

- [深蓝具身智能：跑通具身控制最小闭环](../../sources/blogs/wechat_shenlan_rl_embodied_minimal_closed_loop.md) — PyBullet KUKA 教学案例
- [PyBullet Quickstart](https://github.com/bulletphysics/pybullet/blob/master/examples/pybullet/examples/minitaur.py) — 官方示例（以仓库为准）
