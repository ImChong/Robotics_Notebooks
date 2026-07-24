---
type: entity
tags: [repo, unitree, unitreerobotics, rl]
status: draft
updated: 2026-07-24
related:
  - ./unitree.md
  - ./unitree-rl-gym.md
  - ./unitree-rl-mjlab.md
  - ./unitree-mujoco.md
  - ../tasks/locomotion.md
  - ./legged-gym.md
sources:
  - ../../sources/repos/unitree_rl_lab.md
  - ../../sources/repos/unitree.md
summary: "官方 Isaac Lab 2.x RL 环境，面向 Go2/H1/G1 等机型的并行训练入口。"
---

# unitree_rl_lab

**unitree_rl_lab** 是 [unitreerobotics](https://github.com/unitreerobotics) 组织下的官方仓库，归属 **强化学习训练** 主线。

## 一句话定义

官方 Isaac Lab 2.x RL 环境，面向 Go2/H1/G1 等机型的并行训练入口。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| Sim2Real | Simulation to Real | 仿真到真机迁移 |
| ONNX | Open Neural Network Exchange | 跨框架神经网络交换格式 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习框架 |
| MuJoCo | Multi-Joint dynamics with Contact | 接触丰富的刚体物理引擎 |

## 为什么重要

官方三条 RL 仓分别绑定不同仿真栈；独立节点便于选型与避免混用观测/导出约定。

在宇树官方开源地图中，本仓是 **强化学习训练** 的独立节点；与其它仓的选型关系见 [Unitree](./unitree.md)。

## 核心信息

| 字段 | 内容 |
|------|------|
| 仓库 | [`unitreerobotics/unitree_rl_lab`](https://github.com/unitreerobotics/unitree_rl_lab) |
| 组织分类 | 强化学习训练 |
| 星标（2026-07-24） | ~1223 |
| 最近推送 | 2026-05-25 |
| 主要语言 | Python |

## 工程实践

- Install Isaac Lab by following the installation guide.
- Install the Unitree RL IsaacLab standalone environments.
- Clone or copy this repository separately from the Isaac Lab installation (i.e. outside the IsaacLab directory):
- Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:

- 组织级导航与五条研发主线见 [Unitree 软件生态](./unitree.md)；原始归档见 [sources/repos/unitree_rl_lab.md](../../sources/repos/unitree_rl_lab.md)。

## 局限与风险

- **不要脱离代际混用**：SDK1/ROS1 遗产仓与 SDK2/ROS2/DDS 新栈的消息与启动方式不兼容。
- **README 边界优先**：功能承诺以官方 README 为准；星标高不等于适合你的机型/仿真器。
- **外设/第三方手部仓**：驱动与固件版本需与具体灵巧手/雷达硬件对齐，否则 Serial↔DDS 桥会静默失败。

## 关联页面

- [Unitree 组织枢纽](./unitree.md)
- [unitree_rl_gym](./unitree-rl-gym.md)
- [unitree_rl_mjlab](./unitree-rl-mjlab.md)
- [unitree_mujoco](./unitree-mujoco.md)
- [Locomotion](../tasks/locomotion.md)
- [legged_gym](./legged-gym.md)

## 参考来源

- [sources/repos/unitree_rl_lab.md](../../sources/repos/unitree_rl_lab.md)
- [sources/repos/unitree.md](../../sources/repos/unitree.md) — 组织级仓库地图
- 上游仓库：<https://github.com/unitreerobotics/unitree_rl_lab>

## 推荐继续阅读

- 官方开发者文档：<https://support.unitree.com/home/zh/developer>
- 组织总览：<https://github.com/unitreerobotics>
