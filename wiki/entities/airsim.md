---
type: entity
tags: [repo, simulation, uav, computer-vision, unreal-engine, microsoft]
status: complete
updated: 2026-05-27
related:
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ./px4-autopilot.md
  - ../concepts/sim2real.md
  - ../queries/simulator-selection-guide.md
sources:
  - ../../sources/repos/airsim.md
summary: "Microsoft AirSim：基于 Unreal/Unity 的无人机与自动驾驶视觉仿真，提供 RGB/深度/分割与 PX4 耦合，常用于 SLAM、避障与视觉 Sim2Real（项目已进入维护模式）。"
---

# AirSim

**AirSim**（[microsoft/AirSim](https://github.com/microsoft/AirSim)）是微软开源的 **高保真视觉仿真平台**，基于 **Unreal Engine**（及 Unity 分支），面向无人机与地面自动驾驶的 **感知、控制与 Sim2Real** 研究。

## 为什么重要

- **传感器丰富**：同步 RGB、深度、语义分割、IMU、GPS 等，适合深度学习 pipeline。
- **与 PX4 集成**：可用 **SimpleFlight** 快速实验，或接 **PX4 SITL/HITL** 跑真实飞控栈。
- 文献与课程引用量极大；即使进入维护期，仍是理解「**视觉仿真 + 飞控**」组合的标杆。

## 核心结构/机制

- **车辆 API**（Python/C++）：重置、轨迹跟踪、图像抓取、状态查询。
- **环境**：内置地图、天气与时间变化；可导入自定义 UE 资产。
- **ROS**：社区 wrapper 将图像与位姿接入 ROS 节点。
- **局限域**：桨叶/接触物理简化；**不宜**作为唯一的高保真飞行动力学来源。

与 [Flightmare](https://github.com/uzh-rpg/flightmare)、[gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) 的对照见 [多旋翼栈总览](../overview/multirotor-simulation-planning-control-stack.md)。

## 常见误区或局限

- **误区：AirSim 物理 = 真机** — 视觉域随机化有效，但推力、延迟、风扰需单独建模。
- **局限：维护状态** — 新长期项目应评估 Colosseum 等 fork 与替代仿真器。
- **局限：资源** — UE 构建与 GPU 要求高于 PyBullet Gym。

## 参考来源

- [sources/repos/airsim.md](../../sources/repos/airsim.md)
- [microsoft/AirSim](https://github.com/microsoft/AirSim)

## 关联页面

- [多旋翼仿真—规划—飞控栈总览](../overview/multirotor-simulation-planning-control-stack.md)
- [PX4 Autopilot](./px4-autopilot.md)
- [Sim2Real](../concepts/sim2real.md)

## 推荐继续阅读

- [AirSim 文档](https://microsoft.github.io/AirSim/)
- [PX4 + AirSim 集成说明](https://microsoft.github.io/AirSim/px4_setup/)
