# ros2_control 硬件抽象层框架官方文档与源码入口

> 来源归档（official docs / repo；2026-11-22 核对并通读）

- **标题：** ros2_control — Hardware Abstraction Framework for ROS 2
- **类型：** site / open-source framework
- **源码仓库：** <https://github.com/ros-controls/ros2_control>（[README.md](https://raw.githubusercontent.com/ros-controls/ros2_control/master/README.md)）
- **官方文档：** <https://control.ros.org/>
- **许可：** Apache-2.0
- **入库日期：** 2026-11-22
- **通读：** 已完成
- **核对范围：** `ros-controls/ros2_control` 官方仓库 `README.md` 与文档站索引入口（含 Humble / Jazzy / Rolling LTS 发行矩阵、API 参考与硬件/控制器分离架构定位）。
- **一句话说明：** `ros2_control` 是 ROS 2 生态标准实时控制硬件抽象层（HAL），通过统一的硬件资源接口将上层控制器生命周期与底层总线驱动/仿真插件解耦。

## 核心摘录（面向 wiki 编译）

1. **硬件与控制解耦：** 机器人控制器（如关节轨迹/力矩/导纳控制器）仅面向统一的状态与指令接口，不直接依赖 CAN、EtherCAT 或 Gazebo/Isaac/MuJoCo 等具体后端。
2. **生命周期与多发行版：** 按 ROS 2 Humble/Jazzy/Rolling LTS 长期维护，提供标准硬件插件与控制器管理器（ControllerManager / ResourceManager）范式，支持配置文件一键切换仿真与真机硬件后端。

## 对 wiki 的映射

- [`wiki/queries/hardware-abstraction-layer.md`](../../wiki/queries/hardware-abstraction-layer.md)
