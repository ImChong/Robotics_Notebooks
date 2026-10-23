---

type: entity
project_id: ardupilot
institutions: [ardupilot-foundation]
project: https://ardupilot.org/
code: https://github.com/ArduPilot/ardupilot
tags: [repo, autopilot, ardupilot, mavlink, uav, multirotor, open-source, sitl]
status: complete
updated: 2026-10-23
related:
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ./px4-autopilot.md
  - ./mavsdk.md
  - ./project-quiver.md
  - ./xtdrone.md
sources:
  - ../../sources/repos/ardupilot.md
summary: "ArduPilot 是覆盖多旋翼、固定翼、地面与水下车辆的开源自动驾驶仪软件栈，支持多种飞控板卡，并通过 MAVLink 接入地面站、伴机与仿真工具。"
---

# ArduPilot

**ArduPilot**（[官方文档](https://ardupilot.org/) · [ArduPilot/ardupilot](https://github.com/ArduPilot/ardupilot)）是一个开源自动驾驶仪软件项目。它为不同载具提供对应固件，并运行在兼容的自动驾驶仪硬件上；项目采用 **GPL-3.0** 许可证。

ArduPilot 是社区开发的项目；官方捐赠说明将 ArduPilot Foundation 列为优先捐赠对象，该澳大利亚非营利协会承担项目资金支持。所属机构记录用于标识这一项目支持关系。

## 覆盖的载具

| 固件 | 载具类型 |
|------|----------|
| **ArduCopter** | 多旋翼与传统直升机 |
| **ArduPlane** | 固定翼飞机 |
| **ArduRover** | 地面车辆 |
| **ArduSub** | 水下车辆 |

官方文档还列出 Antenna Tracker 等配套类型。不同固件、板卡和传感器的支持范围有差异，实施时应按具体机型查官方文档。

## 主要接口与开发方式

- **MAVLink**：连接地面站、伴机和上层控制应用，支持遥测、参数配置与任务控制。
- **SITL**：软件在环仿真，可在无实体飞控硬件时运行固件进行开发和测试。
- **多种飞控板卡**：通过硬件抽象与板卡支持层适配不同自动驾驶仪；选型需核对处理器、传感器、接口和目标固件要求。
- **地面站与工具生态**：可按任务配合 Mission Planner、MAVProxy 等工具；更高层应用也可经 MAVLink/MAVSDK 接入。

## 与 PX4 的关系

ArduPilot 与 [PX4 Autopilot](./px4-autopilot.md) 是两套独立的开源飞控软件栈，均可通过 MAVLink 与地面站和伴机系统集成。它们在固件架构、参数体系、板卡支持、仿真工作流和工具生态上各有实现，工程选型应按目标载具、硬件与已有流程验证，不能把两者的参数或固件直接互换。

[Project Quiver](./project-quiver.md) 是一个采用 ArduPilot 的开源机架项目；机架设计与飞控固件是不同层级的资源。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MAVLink | Micro Air Vehicle Link | 飞行器与地面站、伴机之间常用的消息协议 |
| SITL | Software in the Loop | 在软件仿真环境中运行固件的测试方式 |
| GPL | GNU General Public License | ArduPilot 主仓库采用的开源许可证系列 |

## 参考来源

- [ArduPilot 官方文档](https://ardupilot.org/)
- [ArduPilot Foundation 捐赠说明](https://ardupilot.org/dev/docs/common-donation.html)
- [ArduPilot 官方代码仓库](https://github.com/ArduPilot/ardupilot)
- [支持的载具类型](https://ardupilot.org/ardupilot/docs/common-all-vehicle-types.html)
- [ArduPilot Copter 介绍](https://ardupilot.org/copter/docs/introduction.html)
- [向 ArduPilot 项目捐赠](https://ardupilot.org/dev/docs/common-donation.html)

## 关联页面

- [PX4 Autopilot](./px4-autopilot.md) — 独立的开源飞控栈
- [MAVSDK](./mavsdk.md) — 通过 MAVLink 调用飞控能力的伴机 API
- [Project Quiver](./project-quiver.md) — 使用 ArduPilot 的开源多旋翼机架
- [多旋翼仿真—规划—飞控栈总览](../overview/multirotor-simulation-planning-control-stack.md)
