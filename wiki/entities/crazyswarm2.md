---
type: entity
tags: [repo, swarm, crazyflie, ros2, motion-capture, multirotor]
status: complete
updated: 2026-05-27
related:
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ./crazyflie-firmware.md
  - ./gym-pybullet-drones.md
  - ./quad-swarm-rl.md
  - ./ego-planner-swarm.md
sources:
  - ../../sources/repos/crazyswarm2.md
summary: "Crazyswarm2 是 IMRCLab 的 ROS2 大规模 Crazyflie 群体框架：动捕/UWB 定位、轨迹上传与碰撞避免，用于室内多机编队与灯光秀实验。"
---

# Crazyswarm2

**Crazyswarm2**（[IMRCLab/crazyswarm2](https://github.com/IMRCLab/crazyswarm2)）在 **[Crazyflie 固件](./crazyflie-firmware.md)** 之上提供 **数十至上百架微四轴** 的同步起飞、轨迹执行与编队能力（ROS2 重写版 Crazyswarm）。

## 为什么重要

- **真机 swarm 标杆**（微四轴尺度）：与 [EGO-Planner Swarm](./ego-planner-swarm.md)（标准多旋翼 + PX4）形成 **平台尺度对照**。
- 依赖 **动作捕捉 / Lighthouse / UWB** 等 **室内全局定位**，是理解「swarm 先解决定位再谈编队」的典型案例。
- 仿真侧可先用 [gym-pybullet-drones](./gym-pybullet-drones.md) 或 [quad-swarm-rl](./quad-swarm-rl.md)，再迁移到本框架。

## 核心结构/机制

- **ROS2** 节点：广播、轨迹服务、状态监控  
- **Python 脚本**：编队几何、起飞序列、灯光模式  
- **机载**：Crazyflie + 定位甲板（Lighthouse 等）  
- **安全**：简化碰撞模型 + 操作规范（仍以现场规程为准）  

## 常见误区或局限

- **误区：可户外 GPS 大编队** — 设计目标是 **室内定位** 场景。
- **局限：载荷与续航** — 微四轴不适合重载任务。
- **局限：与 PX4 栈不互通** — 控制链为 CRTP + Bitcraze 生态，非 MAVLink 多旋翼。

## 参考来源

- [sources/repos/crazyswarm2.md](../../sources/repos/crazyswarm2.md)
- [IMRCLab/crazyswarm2](https://github.com/IMRCLab/crazyswarm2)

## 关联页面

- [多旋翼栈总览](../overview/multirotor-simulation-planning-control-stack.md)
- [Crazyflie Firmware](./crazyflie-firmware.md)
- [gym-pybullet-drones](./gym-pybullet-drones.md)

## 推荐继续阅读

- [Crazyswarm 文档](https://crazyswarm.readthedocs.io/) — 概念与硬件清单（v2 见仓库 README）
