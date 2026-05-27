# 多旋翼 UAV 仿真—规划—飞控栈：10 仓 source 索引

> 来源归档（catalog）

- **入库日期：** 2026-05-27
- **一句话说明：** 将 PX4、XTDrone、EGO-Planner、PyBullet 无人机 Gym、AirSim、群体 RL/仿真、Crazyflie 固件与 MAVSDK 等 **10 个开源仓库** 分别落成 `sources/repos/*` 归档，并沉淀至 [multirotor-simulation-planning-control-stack.md](../../wiki/overview/multirotor-simulation-planning-control-stack.md)。

## 分层对照

| 层级 | 角色 | 本批仓库 |
|------|------|----------|
| **飞控 / 协议** | 姿态环、模式机、MAVLink | [px4_autopilot.md](px4_autopilot.md)、[mavsdk.md](mavsdk.md)、[crazyflie_firmware.md](crazyflie_firmware.md) |
| **运动规划** | 局部/全局轨迹、避障、编队 | [ego_planner_swarm.md](ego_planner_swarm.md) |
| **仿真平台** | 传感器、场景、SITL/HITL | [airsim.md](airsim.md)、[xtdrone.md](xtdrone.md)、[flightmare.md](flightmare.md) |
| **RL 训练环境** | Gym API、多机交互 | [gym_pybullet_drones.md](gym_pybullet_drones.md)、[quad_swarm_rl.md](quad_swarm_rl.md) |
| **真机群体** | 多机起飞、灯光秀、定位 | [crazyswarm2.md](crazyswarm2.md) |

## 仓库列表

| Source | GitHub | Stars（约） | wiki 实体 |
|--------|--------|-------------|-----------|
| [px4_autopilot.md](px4_autopilot.md) | [PX4/PX4-Autopilot](https://github.com/PX4/PX4-Autopilot) | 11.8k | [px4-autopilot](../../wiki/entities/px4-autopilot.md) |
| [mavsdk.md](mavsdk.md) | [mavlink/MAVSDK](https://github.com/mavlink/MAVSDK) | 0.9k | [mavsdk](../../wiki/entities/mavsdk.md) |
| [ego_planner_swarm.md](ego_planner_swarm.md) | [ZJU-FAST-Lab/ego-planner-swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm) | 2.0k | [ego-planner-swarm](../../wiki/entities/ego-planner-swarm.md) |
| [airsim.md](airsim.md) | [microsoft/AirSim](https://github.com/microsoft/AirSim) | 18.2k | [airsim](../../wiki/entities/airsim.md) |
| [xtdrone.md](xtdrone.md) | [robin-shaun/XTDrone](https://github.com/robin-shaun/XTDrone) | 1.6k | [xtdrone](../../wiki/entities/xtdrone.md) |
| [flightmare.md](flightmare.md) | [uzh-rpg/flightmare](https://github.com/uzh-rpg/flightmare) | 1.4k | [flightmare](../../wiki/entities/flightmare.md) |
| [gym_pybullet_drones.md](gym_pybullet_drones.md) | [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) | 2.0k | [gym-pybullet-drones](../../wiki/entities/gym-pybullet-drones.md) |
| [quad_swarm_rl.md](quad_swarm_rl.md) | [Zhehui-Huang/quad-swarm-rl](https://github.com/Zhehui-Huang/quad-swarm-rl) | 0.2k | [quad-swarm-rl](../../wiki/entities/quad-swarm-rl.md) |
| [crazyswarm2.md](crazyswarm2.md) | [IMRCLab/crazyswarm2](https://github.com/IMRCLab/crazyswarm2) | 0.2k | [crazyswarm2](../../wiki/entities/crazyswarm2.md) |
| [crazyflie_firmware.md](crazyflie_firmware.md) | [bitcraze/crazyflie-firmware](https://github.com/bitcraze/crazyflie-firmware) | 1.5k | [crazyflie-firmware](../../wiki/entities/crazyflie-firmware.md) |

## 关联原始资料（既有）

- [cia_dronecan_uavcan.md](../sites/cia_dronecan_uavcan.md) — PX4/ArduPilot 外设 CAN 协议
- [sim2real.md](../papers/sim2real.md) — 仿真迁移通用框架（腿式为主，概念可类比）
