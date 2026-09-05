# Getting Started With Isaac Sim — Software-in-the-Loop (SIL)

> 来源归档

- **标题：** Software-in-the-Loop (SIL) — Getting Started With Isaac Sim
- **类型：** course / 教程模块
- **URL：** <https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-sim/latest/developing-robots-with-sil-in-isaac-sim/02-software-in-the-loop-sil.html>
- **课程：** Getting Started With Isaac Sim（NVIDIA Physical AI Learning）
- **门户：** <https://docs.nvidia.com/learning/physical-ai/>
- **入库日期：** 2026-09-05
- **一句话说明：** 官方入门课模块：在 **仿真虚拟机器人 + 虚拟环境** 中验证软件（**Software-in-the-Loop**），用 **Isaac Sim + ROS 2 bridge** 测分割、驾驶等场景，再谈与 **HIL** 的分工。

## 模块要点（2026-09-05 抓取）

### SIL 定义

- 在 **无需早期硬件** 的情况下，于仿真中 **测试与验证机器人软件**。
- 虚拟机器人运行在虚拟环境中，与真机解耦。

### 为何需要仿真

| 物理测试痛点 | 仿真收益 |
|--------------|----------|
| 背景/光照/物体布局/动作空间组合难穷举 | AI 可在极端场景训练与测试 |
| 费时、费资源、安全风险 | 降本、控风险、可迭代优化 |

### Isaac Sim 中的 SIL 能力

| 能力 | 说明 |
|------|------|
| **ROS 2 集成** | ROS 2 bridge 与 ROS 包在 Sim 内联调 |
| **OmniGraph** | 无代码可视化脚本搭仿真 |
| **Python API** | 高级定制与自动化 |
| **互补栈** | Isaac ROS / Perceptor / Manipulator 等 AI 应用 |

### SIL vs HIL（课程预告）

| 缩写 | 全称 | 焦点 |
|------|------|------|
| **SIL** | Software-in-the-Loop | 软件在 **纯仿真** 环境验证 |
| **HIL** | Hardware-in-the-Loop | 软件跑在 **真实计算硬件** 上，环境仍可仿真（后续模块 *Leveraging ROS 2 and Hardware-in-the-Loop in Isaac Sim*） |

### 测验要点（官方）

- SIL 主要目的：**在仿真环境测试验证软件**（非替代全部硬件测试）。
- 收益：降本省时、安全受控环境、迭代优化性能。
- 与 HIL 关系：SIL 测软件；HIL 测 **软硬件集成**。

## 命名消歧

本模块 **SIL = Software-in-the-Loop**，与 NVIDIA Research **[Spatial Intelligence Lab](../../wiki/entities/nvidia-spatial-intelligence-lab.md)** 缩写无关。

## 对 wiki 的映射

- 概念页：**`wiki/concepts/software-in-the-loop.md`**
- [Isaac Sim](../../wiki/entities/isaac-sim.md)
- [NVIDIA Physical AI Learning](../../wiki/entities/nvidia-physical-ai-learning.md)
- [ROS 2 基础](../../wiki/concepts/ros2-basics.md)
