# YufengJin/role-ros2 — Nautilus 硬件接口与数据采集仓（nautilus-collect）档案

- **GitHub 地址：** https://github.com/YufengJin/role-ros2
- **默认分支：** `main`
- **开源协议：** Apache-2.0（README 声明 Apache-2.0；仓库元数据未单独回填 license 字段）
- **关联论文：** *Nautilus: From One Prompt to Plug-and-Play Robot Learning*（[arXiv:2605.11665](https://arxiv.org/abs/2605.11665)，[项目页](https://yufengjin.github.io/nautilus/)）
- **开源范围定位：** 本仓库对应论文中的 **`nautilus-collect`**（数据采集与真机控制室），实现 Nautilus 的 `Robot` 类型契约与 ROS 2 双容器硬件底座；项目页显示 Nautilus 主体 Claude Code 插件仓仍为 `Nautilus (Coming Soon)`，因此论文整体开源状态为**部分开源**
- **核对边界：** 静态代码核对，未本地运行（已核查 `README.md` 与 `files_sample` 中的包清单、ROS 2 消息/动作/节点/Launch 文件、Docker Compose 配置与 `role_ros2/` Python 模块结构；未连接 Franka 实体臂、Oculus Quest 或在本地构建 ROS 2 容器执行）

## 核心入口与目录结构（已核实）

- **构建与容器编排（`docker/`、`CMakeLists.txt`、`package.xml`）：**
  - `docker/ros2_cu118/`、`docker/ros2_franka_libfranka_0.14.x/`、`docker/ros2_franka_libfranka_0.18.x/`、`docker/ros2_cu118_franka_0.14.x/`、`docker/ros2_cu118_franka_0.18.x/`：实现相机/GPU 容器（`ros2_cu118_container`，ROS 2 Humble + CUDA 11.8 + ZED SDK）与 Franka 实时控制容器（`ros2_polymetis_container`，ROS 2 Foxy + Polymetis + libfranka 0.14.x/0.18.x）的依赖隔离，二者通过共享 `ROS_DOMAIN_ID` 与 `network_mode: host` 通信。
  - `CMakeLists.txt`、`package.xml`、`quick_rebuild.sh`、`role_ros2/pyproject.toml`、`role_ros2/setup.py`：ROS 2 colcon 工作区与 Python 包构建入口。
- **ROS 2 节点、消息与启动入口（`nodes/`、`msg/`、`action/`、`launch/`、`config/`、`hardware/`）：**
  - `nodes/`：`franka_robot_interface_node.py`、`franka_gripper_interface_node.py`、`robot_state_aggregator_node.py`、`oculus_reader_node.py`、`camera_calibration_tf_publisher_node.py` 五类设备级独立节点，支持按命名空间前缀组合为单臂或双臂系统。
  - `msg/` & `action/`：定义统一向上接口消息（`RobotState.msg`、`AggregatedRobotState.msg`、`ArmState.msg`、`GripperState.msg`、`GripperCommand.msg`、`CartesianPositionCommand.msg`、`CartesianVelocityCommand.msg`、`JointPositionCommand.msg`、`JointVelocityCommand.msg`、`ControllerStatus.msg`、`OculusButtons.msg`、`VRPolicyAction.msg` 及 Polymetis 适配消息）与长时程动作 `action/MoveToPose.action`。
  - `launch/` & `hardware/`：`franka_robot.launch.py`、`bimanual_franka_robot.launch.py`、`oculus_controller.launch.py`、`realsense_camera.launch.py`、`zed_camera.launch.py`，以及 Franka FR3/Panda 硬件启动脚本（`hardware/franka/launch/launch_robot.sh`、`launch_gripper.sh`、`kill_all.sh`）。
- **Python 窄腰抽象与工具链（`role_ros2/` 及 README 记录的 `scripts/`）：**
  - `role_ros2/robot/base_robot.py`、`role_ros2/robot/franka/robot.py`（`FrankaRobot`）、`role_ros2/robot/franka/bimanual_robot.py`（`BimanualFrankaRobot`）与 `role_ros2/robot_env.py`（Gym 兼容 `RobotEnv`）：向上暴露 Nautilus 的 `Robot` 四方法契约（`reset`、`get_observation`、`apply_action`、`safe_stop`），向下屏蔽 Polymetis/ROS 2 细节。
  - `role_ros2/controllers/base_controller.py` 与 `oculus_controller.py`：抽象遥操作控制器（Oculus Quest VR 与 README/论文中的 GELLO 均输出统一 `Action`）。
  - `role_ros2/camera/`（`multi_camera_wrapper.py`、`ros2_camera_reader.py`）、`role_ros2/calibration/`（`calibration_utils.py`、`config.py` Charuco 手眼标定）与 `role_ros2/robot_ik/`（`robot_ik_solver.py`、`arm.py` 与 FR3/Panda URDF/XML 模型）。
  - README 另有记载但超出 `files_sample` 截断列表的 `scripts/` 工具入口：`scripts/bringup.py`（PyQt5 Docker-ROS 控制中心）、`collect_trajectory_franka.py`、`collect_trajectory_bimanual_franka.py`、`postprocess/label_data.py` 与 `postprocess/to_tfrecord.py`（HDF5 转 DROID 兼容 TFRecord/TFDS）。

## 复现与运行依赖边界

- **硬件与系统依赖：** 完整真机运行需 Docker、Docker Compose、`nvidia-container-toolkit`（ZED 相机 GPU 加速）、Franka Panda/FR3 控制机（实时内核 + libfranka 0.14.x 或 0.18.x + 作为 git submodule 引入的已归档 Polymetis），以及可选 Oculus Quest 2/3 或 GELLO 主手与 ZED/RealSense 相机。
- **与论文范围差异：** 本仓库仅覆盖 Nautilus 架构中的 `nautilus-collect` 硬件通信、遥操作数据采集、TFDS 导出与通过 `policy_websocket` 对接外部策略推理服务的部分；论文 §3.3 所述的 L2–L6 Claude Code 插件、MCP 注册表与子智能体（`env-generator`、`benchmark-generator`、`policy-generator` 等）位于项目页标为 `Coming Soon` 的主仓，不在本仓库内。

## 与 wiki 的映射

- 对应实体页：[Nautilus: From One Prompt to Plug-and-Play Robot Learning](../../wiki/entities/paper-agentic-2605-11665.md)
