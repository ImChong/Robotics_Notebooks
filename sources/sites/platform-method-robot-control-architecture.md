# ros2_control Humble 控制管理器与硬件接口

> 一手资料归档 · 2026-11-26

- [Controller Manager](https://control.ros.org/humble/doc/ros2_control/controller_manager/doc/userdoc.html)
- [硬件接口类型](https://control.ros.org/humble/doc/ros2_control/hardware_interface/doc/hardware_interface_types_userdoc.html)
- **核对范围：** Humble 官方说明中的 Determinism、Parameters、Helper scripts，以及硬件接口类型的 Joints 和 Sensors。
- **通读：** 已完成

## 核对摘记

关节硬件按状态与命令分接口。控制器管理器的主循环依次执行硬件读、控制更新和硬件写，并管理控制器生命周期。

## 对 wiki 的映射

- [机器人控制系统架构](../../wiki/methods/robot-control-architecture.md)
