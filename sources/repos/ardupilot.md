# ArduPilot

> 来源归档

- **标题：** ArduPilot
- **类型：** repo
- **链接：** https://github.com/ArduPilot/ardupilot
- **官方文档：** https://ardupilot.org/
- **许可证：** GPL-3.0
- **入库日期：** 2026-10-23
- **一句话说明：** 开源自动驾驶仪软件栈，覆盖多旋翼与直升机（Copter）、固定翼（Plane）、地面车辆（Rover）、水下车辆（Sub）等，并提供仿真、地面站和伴机 API 生态。
- **沉淀到 wiki：** [ardupilot](../../wiki/entities/ardupilot.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

ArduPilot 是开源自动驾驶仪软件项目。主仓库包含 ArduCopter、ArduPlane、ArduRover、ArduSub 等固件代码，可运行在多种兼容自动驾驶仪板卡上。官方文档按机型提供安装、配置、调参与任务操作指南。

## 主要接口与生态

- 使用 MAVLink 与地面站、伴机及任务控制应用通信；Mission Planner、MAVProxy、MAVSDK 等工具可按场景配套使用。
- 提供 SITL（Software in the Loop）仿真路径，可在无实体飞控硬件时开发和测试。
- 具体支持能力、传感器与模式取决于机型、固件版本和飞控硬件；部署前应查阅对应机型的官方文档与板卡支持列表。

## 与 wiki 的映射

- 飞控实体：[ardupilot](../../wiki/entities/ardupilot.md)
- 多旋翼全栈总览：[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

## 一手来源

- [ArduPilot 官方文档](https://ardupilot.org/)
- [ArduPilot 官方 GitHub 仓库](https://github.com/ArduPilot/ardupilot)
- [Vehicle Types Supported by ArduPilot](https://ardupilot.org/ardupilot/docs/common-all-vehicle-types.html)
