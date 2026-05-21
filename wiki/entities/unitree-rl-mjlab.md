---
type: entity
tags: [repo, framework, unitree, mjlab, mujoco, reinforcement-learning, sim2real, humanoid, locomotion]
status: complete
updated: 2026-05-17
related:
  - ./unitree-ros.md
  - ./mjlab.md
  - ./mjlab-playground.md
  - ./unitree.md
  - ./unitree-g1.md
  - ./amp-mjlab.md
  - ./legged-gym.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/repos/unitree_rl_mjlab.md
summary: "unitree_rl_mjlab 是 Unitree 官方基于 mjlab 的 RL 训练框架，支持 Go2/G1/H1_2 等 7 款机型，覆盖速度跟踪与动作模仿，内建 ONNX 导出到 C++ 真机部署的完整链路。"
---

# unitree_rl_mjlab (Unitree 官方 RL 框架)

**unitree_rl_mjlab** 是由 Unitree Robotics 官方维护的强化学习训练框架，以 **mjlab**（Isaac Lab API + MuJoCo Warp）为底层，覆盖旗下 7 款机器人型号，提供从仿真训练到真机部署的完整 pipeline。

## 为什么重要？

这是 Unitree 的**官方推荐**训练路线之一，与社区框架（legged_gym、AMP_mjlab）的关键区别：

- **官方维护**：与 Unitree SDK、ONNX 部署链路深度整合
- **多机型覆盖**：单框架支持四足（Go2）和人形（G1、H1_2、H2 等）
- **完整 Sim2Real 链路**：训练 → ONNX 导出 → C++ 编译 → 真机部署，有明确的操作步骤

## 支持机器人

Go2、A2、As2、**G1**、R1、**H1_2**、**H2**（共 7 款，含主流人形）

## 训练任务

| 任务类型 | 示例环境 |
|---------|---------|
| 速度跟踪（Velocity Tracking） | `Unitree-G1-Flat` |
| 动作模仿（Motion Imitation） | `Unitree-G1-Tracking-No-State-Estimation` |

训练规模：支持 4096 个并行环境（GPU 加速）。

## Sim2Real 部署链路

```
MuJoCo 并行训练
    ↓ ONNX 导出
C++ 控制程序编译
（cyclonedds + unitree_sdk2）
    ↓
unitree_mujoco 仿真验证
    ↓
真机部署
（以太网 192.168.123.222，调试模式）
```

关键依赖：**cyclonedds**（DDS 通信）、**unitree_sdk2**（机器人接口）。

## 与相关框架的对比

| 维度 | unitree_rl_mjlab | AMP_mjlab | legged_gym | robot_lab |
|------|-----------------|-----------|------------|-----------|
| 维护方 | Unitree 官方 | 社区 | ETH RSL | 个人 |
| 底层仿真 | mjlab（MuJoCo Warp） | mjlab | IsaacGym | IsaacLab |
| 支持机型 | Unitree 7 款 | G1 | 通用 | 26+ |
| 核心任务 | 速度跟踪 + 动作模仿 | AMP + recovery | 速度跟踪 | 速度跟踪 + AMP Dance |
| 部署链路 | ONNX → C++（官方） | ONNX | 各异 | 各异 |

**与 `unitree_ros` 的分工**：`unitree_ros` 提供 ROS1 + Gazebo8 的 URDF 与关节级仿真/控制示例（README 明确 Gazebo 不做高层行走）；`unitree_rl_mjlab` 面向当前研究常用的 **MuJoCo 并行训练 + SDK2** 部署闭环。二者同属官方开源，但中间件与仿真器代际不同，见 [unitree_ros](./unitree-ros.md)。

## 关联页面

- [mjlab](./mjlab.md) — 底层框架（Isaac Lab API + MuJoCo Warp）
- [Unitree](./unitree.md) — 品牌主页
- [unitree_ros](./unitree-ros.md) — 官方 ROS1 / Gazebo URDF 与关节仿真栈（与本书 MuJoCo 主线对照）
- [Unitree G1](./unitree-g1.md) — 主要人形目标机型
- [AMP_mjlab](./amp-mjlab.md) — 同基于 mjlab 的社区 AMP 实现
- [legged_gym](./legged-gym.md) — 同类框架，基于 IsaacGym
- [Locomotion](../tasks/locomotion.md) — 任务方向

## 参考来源

- [sources/repos/unitree_rl_mjlab.md](../../sources/repos/unitree_rl_mjlab.md)
- [unitreerobotics/unitree_rl_mjlab GitHub Repo](https://github.com/unitreerobotics/unitree_rl_mjlab)
