---
type: entity
tags: [repo, autopilot, px4, mavlink, uav, multirotor, open-source]
status: complete
updated: 2026-05-27
related:
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ../concepts/sim2real.md
  - ../concepts/can-bus-protocol.md
  - ./airsim.md
  - ./mavsdk.md
  - ./ego-planner-swarm.md
  - ./xtdrone.md
  - ./crazyflie-firmware.md
sources:
  - ../../sources/repos/px4_autopilot.md
  - ../../sources/sites/cia_dronecan_uavcan.md
summary: "PX4 是 Dronecode 生态的开源自动驾驶仪：多机型飞控固件、EKF 估计、任务/Offboard 模式、SITL 与 MAVLink，常与 QGroundControl、MAVSDK、Gazebo/AirSim 仿真组合。"
---

# PX4 Autopilot

**PX4**（[PX4/PX4-Autopilot](https://github.com/PX4/PX4-Autopilot)）是面向多旋翼、固定翼、VTOL 等的 **开源飞控固件**，通过 **MAVLink** 与地面站、伴机、仿真器通信，是民用与研究无人机领域最常见的基础栈之一。

## 为什么重要

- **分层清晰**：机载闭环（估计 + 控制）与伴机规划（ROS / MAVSDK）解耦，便于 Sim2Real 与算法迭代。
- **生态完整**：QGroundControl 校准、**SITL** 软件在环、ROS2 `px4_ros_com`、大量硬件支持包。
- 与仓库 [多旋翼栈总览](../overview/multirotor-simulation-planning-control-stack.md) 中 XTDrone、EGO-Planner、AirSim 等均为 **执行层锚点**。

## 核心结构/机制

| 组件 | 说明 |
|------|------|
| **uORB** | 机载发布/订阅总线，模块间解耦 |
| **EKF2** | 多传感器融合（IMU、GPS、视觉、光流等） |
| **MC/FW 控制器** | 多旋翼姿态/速率/位置；固定翼横向/纵向 |
| **Navigator** | 任务、返航、降落状态机 |
| **MAVLink** | 地面站、Offboard 设定点、参数读写 |
| **SITL** | 无硬件在 PC 上跑完整飞控栈 |

**Offboard 模式**：伴机以一定频率发送位置/速度/姿态设定点，PX4 内环跟踪——与 [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner-swarm) 等规划器对接的标准方式。

外设 **ESC、GPS** 等可走 [DroneCAN](../../sources/sites/cia_dronecan_uavcan.md)（见 [电机协议总览](../overview/motor-drive-firmware-bus-protocols.md) 中 DroneCAN 行）。

## 常见误区或局限

- **误区：装好 PX4 就等于完成自主导航** — 仍需定位、规划、避障与安全监管；飞控只保证 **可飞行的低层跟踪**。
- **误区：SITL 与真机参数一致** — 需重新校准传感器、推力模型与延迟；Sim2Real 见 [Sim2Real](../concepts/sim2real.md)。
- **局限：微四轴** — 27g 级 [Crazyflie Firmware](./crazyflie-firmware.md) 不走 PX4，选型勿混淆。

## 参考来源

- [sources/repos/px4_autopilot.md](../../sources/repos/px4_autopilot.md)
- [sources/sites/cia_dronecan_uavcan.md](../../sources/sites/cia_dronecan_uavcan.md)
- [PX4/PX4-Autopilot](https://github.com/PX4/PX4-Autopilot)

## 关联页面

- [多旋翼仿真—规划—飞控栈总览](../overview/multirotor-simulation-planning-control-stack.md)
- [AirSim](./airsim.md)
- [MAVSDK](./mavsdk.md)
- [XTDrone](./xtdrone.md)
- [EGO-Planner Swarm](./ego-planner-swarm.md)
- [Sim2Real](../concepts/sim2real.md)

## 推荐继续阅读

- [PX4 User Guide — Flying](https://docs.px4.io/main/en/flying/)
- [MAVSDK Offboard](https://mavsdk.mavlink.io/main/en/cpp/guide/offboard.html)
