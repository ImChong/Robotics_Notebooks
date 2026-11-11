# ExoStack 官方 ROS 2 实现核验
- **仓库：** https://github.com/debadrata/ExoStack
- **许可：** `LICENSE` = PolyForm Noncommercial 1.0.0；`LICENSE-DOCS.md` = CC BY-NC 4.0
- **核对范围：** 官方 README/安装/Quickstart、LICENSE，`PASE_GUI/main.py`、`exo_stack/setup.py`、`exo_stack/exo_stack/{sync_with_trigger,gait_event_detector,mplanner,profile,cost_updated,hil}.py`、节点目录及 launch 文件。
- **通读：** 已完成
- **日期：** 2026-11-11

## 已核实的实际入口
- GUI `python3 PASE_GUI/main.py` -> `MainWindow` -> `ConfigTab`、`LaunchTab`、`VisualizationTab`，负责配置、启动与显示（不直接替代底层 node）。
- 宿主 `ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888`，接 ESP32 ESA hub。
- `ros2 run exo_stack sync` -> `sync_with_trigger.GenericSyncNode`，由 sensors.yaml 配置，使用 `ApproximateTimeSynchronizer` 对 hub 合并。
- `gait_event_detector_launch.py` -> `gait_event_detector`，步态事件通道。
- `mplanner_launch.py` -> `mplanner.GenericPlannerNode`，由 `mplanner.yaml` 加载算法插件、profile 与 gait topic；定时发布运动规划。
- `profile_launch.py` -> `profile.GenericProfileBuilder`，根据 speed/intent 或 `hil_callback` 产生辅助 profile。
- `cost_launch.py` -> `cost_updated.GenericCostNode`，按 candidate/profile/gait 评估成本。
- `hil_launch.py` -> `hil.GenericHILNode`，从 `profile.yaml` 动态加载 CMA-ES/Bayesian 类，`publish_next_candidate()` 后订阅 `cost_callback` 并更新下一候选。
- `setup.py` 的 `console_scripts` 含一些旧的 emulator/gui 路径，目前 `exo_stack/exo_stack/` 并无所有同名模块；部署需逐条核验实际可用入口与 launch 安装，不应声称所有命令都已实测。

## 边界
没有在本次任务中完成 ROS 2 + ESP32/外骨骼实时硬件复现；所列为代码静态核对。
- [论文](../papers/exostack_arxiv_2611_12306.md) · [官网](../sites/exostack-documentation.md) · [wiki](../../wiki/entities/paper-exostack.md)
