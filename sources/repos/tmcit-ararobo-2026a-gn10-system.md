# TMCIT 荒川高专 2026 A 队（GN10）｜机器人系统开源归档

> 官方多仓库开发工程的一手资料索引；归档范围是 **荒川校区 2026 A 队 GN10**，不是全国赛全部队伍，也不是机械整机完全可复刻的保证。

- **标题：** GN10 Robot System / 都立产技高专荒川 2026 A 队机器人开源系统
- **类型：** repo
- **来源：** Tokyo Metropolitan College of Industrial Technology, Arakawa Campus, Robot Research Club — TMCIT Ararobo 2026A
- **链接：** https://github.com/tmcit-ararobo-2026a
- **原文：** https://github.com/tmcit-ararobo-2026a/gn10-navigation/blob/main/README.md
- **代码：** https://github.com/tmcit-ararobo-2026a/ros2_ws
- **项目：** https://github.com/tmcit-ararobo-2026a
- **赛事官方：** https://official-robocon.com/kosen/
- **开源状态：** 部分系统工程已开源（多仓库源码及部分电路设计）；未证明全体参赛队/完整机械 BOM、CAD、线束与制造资料全部开源。
- **核对版本：** 2026-11-18 访问 GitHub 公开仓库当前 README 与对应公开源码路径；该多仓系统持续更新，不作固定 release 保证。
- **入库日期：** 2026-11-18
- **核对范围：** 官方组织公开项目列表；gn10-navigation README 全文及 launch/navigation.launch.py、package.xml；gn10-mainboard README 与 Core/Src/main.c；ros2_ws README；power-management-board 与 esc-control-hub README；robot-sim-unity README 中物理建模、启动与 ROS 话题；已有 [定位源码精读归档](gn10-pointcloud-localization.md)。核对的是各仓主流程和公开运行入口，不代表已逐行审计所有仓库、进行真机复现或测量性能。
- **通读：** 已完成
- **沉淀到 wiki：** [GN10 系统级实体](../../wiki/entities/tmcit-ararobo-2026a-gn10-system.md)
- **局部专项：** [GN10 PointCloud Localization](gn10-pointcloud-localization.md)
- **路线：** [导航纵深 Stage 2](../../roadmap/depth-navigation.md)、[PCB 设计纵深](../../roadmap/depth-pcb-design.md)

## 核心公开仓库及作用

| 模块 | 官方仓库 | 已核实的资料 / 入口 |
|---|---|---|
| ROS 2 工作区 | [ros2_ws](https://github.com/tmcit-ararobo-2026a/ros2_ws) | `setup.sh` 拉取依赖，`update.sh` 更新构建；README |
| 地图定位 | [gn10-pointcloud-localization](https://github.com/tmcit-ararobo-2026a/gn10-pointcloud-localization) | 静态 ESDF + CUDA 3D 纹理；`localization.launch.py`；详见既有归档 |
| 目标/规划/跟踪 | [gn10-navigation](https://github.com/tmcit-ararobo-2026a/gn10-navigation) | `navigation.launch.py` 同时启动目标管理、costmap、规划、跟踪、可选移动桶射速节点 |
| 主控制器固件 | [gn10-mainboard](https://github.com/tmcit-ararobo-2026a/gn10-mainboard) | STM32G474 + W5500；`Core/Src/main.c` 的 `setup()/loop()`；CMake Presets |
| 电源保护板 | [power-management-board](https://github.com/tmcit-ararobo-2026a/power-management-board) | STM32G431、预充、ADC/DMA 电压电流、物理/CAN 双急停 |
| 电机控制隔离板 | [esc-control-hub](https://github.com/tmcit-ararobo-2026a/esc-control-hub) | 隔离 CAN、DJI C610/C620 转发、1 kHz PID、限位与超时安全 |
| Unity 仿真 | [robot-sim-unity](https://github.com/tmcit-ararobo-2026a/robot-sim-unity) | Unity 6 场景 `NHK2026_SimScene.unity`，ROS 2 Humble、LiDAR/IMU、RealSense 和布投掷 |
| Isaac Sim 仿真 | [robot-sim](https://github.com/tmcit-ararobo-2026a/robot-sim) | 官方组织发布的 Isaac Sim 仿真仓库；本次未审读可运行入口 |

## 可核实的运行链

1. 在 ROS 2 Humble 工作区按 `ros2_ws/README.md` 的 `./setup.sh` 安装关联源码并按需编译。
2. 地图定位器消费 LiDAR、IMU 与静态场地图，输出 `map -> base_link` TF；这是预建地图定位而非完整自主建图。
3. `ros2 launch gn10_navigation navigation.launch.py`：从 `launch/navigation.launch.py` 确认启动目标管理、costmap、路径规划、路径跟踪及移动桶射速计算（按参数可关），RViz 默认不启动。
4. `target_manager_node` 从目标 ID 产生 `/goal_pose`；`costmap_generator_node` 从 `nhk2026_map.json` 产生 `/costmap`；`path_planner_node` 用 8 邻接 A* 与路径平滑输出 `/planned_path`；`path_follower_node` 输出 `/robot/operation/cmd_vel`。
5. 主板 README 定义 Jetson 端 ROS 2 运行指令到 UDP `operation_t`，STM32G474 执行遥控/自动仲裁与超时保护，再通过 3 路 FDCAN 与外设交互；实际 `udp_node` 源文件与真机链路尚未作端到端运行验证。
6. Unity 6 打开场景并 Play；官方 README 列举 `ros2 launch robot_bringup unity_sim.launch.py use_sim_time:=true`，与真实 ROS 节点复用话题接口。仿真是刚体和传感器的工程近似，不等于实机动力学标定。

## 安全与接口参数（均为仓库文档，不是独立测试值）

- `gn10-mainboard`：`teleop_t` 超时 **100 ms** 关闭执行器；`operation_t` 超时 **200 ms** 回退到手动控制；`feedback_t` 每 **20 ms** 发送。
- `power-management-board`：预充 **500 ms** 后主继电器闭合，**700 ms** 后断开预充继电器；具物理/远程双急停。
- `esc-control-hub`：文档定义 **1 kHz** 反馈控制、命令超时关闭输出；均需在具体电路/负载上重新验证。
- `robot-sim-unity`：使用刚体近似软布、全向轮系，没有逐轮打滑/扭矩精细模型，物理参数含开发者估计值，不能把仿真结果等同真机性能。

## 开放程度与许可边界

GN10 的多个 ROS 2、STM32 固件与板卡仓库公开并有各自许可证（如 navigation/mainboard/power/ESC 为 MIT；GN10 定位仓库为 Apache-2.0）。**不要把组织页面的“公开 45 仓库”理解为 45 个完整可运行的独立整机或所有资料具有同一许可证**。Unity 仿真仓库需按实际文件与 LFS 逐项确认可取得性；机械结构完整 CAD、BOM、装配与布线仍需要专门核对，本站不声明完全开源。

## 原始链接

- [官方代码组织](https://github.com/tmcit-ararobo-2026a)
- [导航 README 与运行入口](https://github.com/tmcit-ararobo-2026a/gn10-navigation/blob/main/README.md)
- [导航 launch 源码](https://github.com/tmcit-ararobo-2026a/gn10-navigation/blob/main/launch/navigation.launch.py)
- [主板 README 与固件入口](https://github.com/tmcit-ararobo-2026a/gn10-mainboard/blob/main/README.md)
- [Unity 仿真 README](https://github.com/tmcit-ararobo-2026a/robot-sim-unity/blob/main/README.md)
- [官方竞赛](https://official-robocon.com/kosen/)
