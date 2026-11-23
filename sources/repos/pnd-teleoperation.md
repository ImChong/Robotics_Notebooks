# PNDbotics 遥操作仓：Adam-U 的 Noitom 动捕分支

- **类型：** repo / ROS 2 / teleoperation
- **仓库：** <https://github.com/pndbotics/pnd_teleoperation>
- **项目页：** [Adam-U 官方发布](https://noitomrobotics.com/purpose-built-humanoid-data-collection-platform-for-embodied-ai/)（[归档](../sites/noitom-robotics.md)）
- **核查版本：** [5b95de7，v0.8.0](https://github.com/pndbotics/pnd_teleoperation/tree/5b95de781aab6dcd3c2b1e4a874843dd96136715)
- **核查日期：** 2026-11-21
- **核对范围：** README、Python CLI 分发、Adam-U Noitom 启动文件、Noitom 驱动、重定向与关节状态订阅的运行入口。
- **通读：** 已完成

## 入口与源码

README 的二进制快启示例使用 WebVR 与 Mink。核对 Noitom 分支时，在仓库所述的 Ubuntu 22.04 / ROS 2 Humble 环境，使用 `pteleop teleop adam_u noitom pinocchio` 对应的 `ros2 launch bringup pinocchio-adam_u-noitom.launch.py`。

| 文件 | 运行职责 |
|---|---|
| `pyproject.toml`、`scripts/cli/teleop_cli.py`、`scripts/cli/teleop_cmd.py` | 将 `pteleop teleop` 的参数组合解析到 bringup 启动文件。 |
| `src/bringup/launch/pinocchio-adam_u-noitom.launch.py` | 启动 Noitom 驱动、静态坐标变换、重定向与机器人状态发布；将动捕 TF 重映射到 `/mocap/tf`。 |
| `src/driver/noitom_mocap/src/noitom_data_handler.cpp`、`src/driver/noitom_mocap/src/main.cpp` | 调用 MocapApi 读取人体骨架更新，`RobotTfBroadcaster` 将关节平移和旋转广播为 ROS TF。 |
| `src/algorithm/adam_retarget/src/main.cpp`、`Retarget.cpp` | 定时从 TF 读取骨架，计算重定向参数并求解 Adam-U 关节状态，发布 `/joint_states`。 |
| `src/visualization/adam_state_publisher/src/adam_state_publisher.cpp` | 订阅 `joint_states`，更新机器人状态。 |

该路径核查止于关节状态发布。仓库根目录当时未见统一 LICENSE；第三方组件另带许可证。本次为源码核对，尚无动捕设备连线或真机运行结果。

## 对 wiki 的映射

- [诺亦腾机器人 · 代码路径](../../wiki/entities/noitom-robotics.md#源码运行时序图)
