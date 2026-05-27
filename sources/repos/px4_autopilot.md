# PX4-Autopilot

> 来源归档

- **标题：** PX4 Autopilot
- **类型：** repo
- **链接：** https://github.com/PX4/PX4-Autopilot
- **Stars / Forks：** ~11.8k / —（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** 开源多旋翼/固定翼/VTOL 自动驾驶仪：模块化飞行栈、MAVLink 生态、SITL/HITL 与 QGroundControl 配套。
- **沉淀到 wiki：** [px4-autopilot](../../wiki/entities/px4-autopilot.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**PX4** 是 Dronecode 基金会维护的 **C/C++ 飞控固件**，面向研究、工业与爱好者多机型。上层通过 **MAVLink** 与 QGC、ROS/ROS2、MAVSDK 等通信；底层以 uORB 话题总线连接估计器、控制器与驱动。

典型能力：姿态/位置控制、任务模式（Mission / Offboard / Hold）、避障模块接口、**SITL**（软件在环）与多种 RTOS/NuttX 板卡支持。

---

## 架构要点

| 模块 | 职责 |
|------|------|
| **Estimator** | EKF2 等融合 IMU、GPS、视觉、光流 |
| **Controller** | 多旋翼 MC 姿态/速率/位置 PID；固定翼 FW 控制 |
| **Navigator** | 航点、返航、降落状态机 |
| **Drivers** | 传感器、PWM/DroneCAN ESC、仿真 UDP 注入 |
| **MAVLink** | 地面站、伴机、Offboard 设定点 |

---

## 与本批其它资料的关系

| 资料 | 关系 |
|------|------|
| [mavsdk.md](mavsdk.md) | 应用层 C++/Python API，经 MAVLink 驱动 PX4 |
| [xtdrone.md](xtdrone.md) | Gazebo + ROS 仿真常对接 PX4 SITL |
| [cia_dronecan_uavcan.md](../sites/cia_dronecan_uavcan.md) | ESC/外设 DroneCAN 与 PX4 驱动栈 |
| [airsim.md](airsim.md) | 可用 AirSim 作为视觉/物理前端，PX4 作飞控后端 |

---

## 对 wiki 的映射

- 飞控栈实体：[px4-autopilot](../../wiki/entities/px4-autopilot.md)
- 多旋翼全栈总览：[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)
