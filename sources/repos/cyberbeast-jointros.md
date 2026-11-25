# CyberBeast JointROS：ROS 2 关节驱动与 ros2_control / ROS 2 Joint Integration

> 来源归档（CyberBeast 官方 GitHub）

- **标题：** JointROS
- **类型：** repo / firmware / joint-control
- **官方仓库：** https://github.com/cyberbeast0829/JointROS
- **核对分支：** `main`
- **入库日期：** 2026-11-22
- **核对范围：** README 环境准备、虚拟总线快速启动、真机生命周期、YAML tick_groups 与 ros2_control 章节；jr_system_interface.cpp 初始化与控制模式接口。一手入口：`README.md`、`jr_ros2_control/src/jr_system_interface.cpp`、`jr_ros2/CMakeLists.txt`；审查重点为本次软件栈定位，不声称覆盖仓库全部源码、历史提交或硬件实测。
- **通读：** 已完成
- **一句话说明：** 在 JointSDK 上封装 ROS 2 lifecycle、/joint_states、ros2_control SystemInterface 与 JointTrajectoryController 的关节通信层。
- **开源状态：** 已公开源码；对应构建与真机兼容性以仓库 README 和具体硬件版本为准。
- **沉淀到 wiki：** 是 → [cyberbeast-jointros](../../wiki/entities/cyberbeast-jointros.md)

## 一手事实与代码定位

`jr_bus` lifecycle 的 configure 阶段加载 YAML、打开总线、设备发现，activate 后才建话题/服务并运行 tick；`jr_ros2_control` 从同一 YAML 导出 position/velocity/effort 接口供 JTC 使用。默认示例 tick_groups.rate_hz=1000。

## 官方构建 / 运行入口

- `colcon build --base-paths jr_interfaces jr_ros2 jr_ros2_control jr_bringup --cmake-args -DJRSDK_SOURCE_DIR=$HOME/JointSDK`
- `ros2 launch jr_bringup vbus_demo.launch.py`
- `ros2 lifecycle set /axis1 configure`
- `ros2 lifecycle set /axis1 activate`

## 工程边界

裸 `ros2 run jr_bus` 不会自动 activate；需要 lifecycle configure/activate；SLCAN 仅配置与监控，生产实时闭环优先 SocketCAN 并测量抖动；不保证任意设备 1 kHz。

## 参考原文

- [README.md](https://github.com/cyberbeast0829/JointROS/blob/main/README.md)
- [jr_ros2_control/src/jr_system_interface.cpp](https://github.com/cyberbeast0829/JointROS/blob/main/jr_ros2_control/src/jr_system_interface.cpp)
- [jr_ros2/CMakeLists.txt](https://github.com/cyberbeast0829/JointROS/blob/main/jr_ros2/CMakeLists.txt)
