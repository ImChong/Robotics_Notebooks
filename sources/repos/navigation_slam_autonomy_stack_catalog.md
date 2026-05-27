# 导航·SLAM·自动驾驶·具身栈：21 仓 source 索引

> 来源归档（catalog）

- **入库日期：** 2026-05-27
- **一句话说明：** 将 Nav2、2D/3D SLAM、LiDAR/VIO、Autoware、Isaac ROS、OpenLoong 动力学控制、LeRobot/OpenVLA、MuSHR 等 **21 个开源仓库** 落成 `sources/repos/*` 归档，并沉淀至 [navigation-slam-autonomy-stack.md](../../wiki/overview/navigation-slam-autonomy-stack.md)。

## 分层对照

| 层级 | 角色 | 本批仓库 |
|------|------|----------|
| **全局导航（ROS 2）** | 行为树、全局/局部规划、恢复 | [navigation2.md](navigation2.md)、[mushr.md](mushr.md) |
| **2D 激光 SLAM** | 建图、定位、 lifelong | [slam_toolbox.md](slam_toolbox.md)、[cartographer.md](cartographer.md) |
| **3D LiDAR 里程计/SLAM** | LIO、地面优化、图优化 | [fast_lio.md](fast_lio.md)、[lio_sam.md](lio_sam.md)、[lego_loam.md](lego_loam.md)、[hdl_graph_slam.md](hdl_graph_slam.md) |
| **视觉 / VIO** | 单目/双目/IMU、语义 SLAM | [orb_slam3.md](orb_slam3.md)、[vins_fusion.md](vins_fusion.md)、[openvslam.md](openvslam.md)、[open_vins.md](open_vins.md)、[kimera.md](kimera.md) |
| **多模态建图** | RGB-D、记忆管理 | [rtabmap.md](rtabmap.md)、[voxgraph.md](voxgraph.md) |
| **自动驾驶全栈** | 感知-规划-控制 | [autoware.md](autoware.md) |
| **NVIDIA 加速栈** | cuVSLAM、TSDF/ESDF + Nav2 | [isaac_ros_visual_slam.md](isaac_ros_visual_slam.md)、[isaac_ros_nvblox.md](isaac_ros_nvblox.md) |
| **人形动力学（非导航）** | MuJoCo MPC+WBC | [openloong_dyn_control.md](openloong_dyn_control.md) |
| **具身学习** | 数据+策略+VLA | [lerobot.md](lerobot.md)、[openvla.md](openvla.md) |

## 仓库列表

| Source | GitHub | Stars（约） | wiki 映射 |
|--------|--------|-------------|-----------|
| [navigation2.md](navigation2.md) | [ros-navigation/navigation2](https://github.com/ros-navigation/navigation2) | 4.3k | [navigation2](../../wiki/entities/navigation2.md) |
| [slam_toolbox.md](slam_toolbox.md) | [SteveMacenski/slam_toolbox](https://github.com/SteveMacenski/slam_toolbox) | 2.5k | 总览栈 |
| [cartographer.md](cartographer.md) | [cartographer-project/cartographer](https://github.com/cartographer-project/cartographer) | 7.9k | 总览栈 |
| [orb_slam3.md](orb_slam3.md) | [UZ-SLAMLab/ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | 8.7k | SLAM 选型 |
| [vins_fusion.md](vins_fusion.md) | [HKUST-Aerial-Robotics/VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) | 4.5k | SLAM 选型 |
| [openvslam.md](openvslam.md) | [xdspacelab/openvslam](https://github.com/xdspacelab/openvslam) | 3.0k | SLAM 选型 |
| [fast_lio.md](fast_lio.md) | [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO) | 4.7k | 总览 + 选型 |
| [lio_sam.md](lio_sam.md) | [TixiaoShan/LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) | 4.8k | SLAM 选型 |
| [autoware.md](autoware.md) | [autowarefoundation/autoware](https://github.com/autowarefoundation/autoware) | 11.7k | [autoware](../../wiki/entities/autoware.md) |
| [openloong_dyn_control.md](openloong_dyn_control.md) | [loongOpen/OpenLoong-Dyn-Control](https://github.com/loongOpen/OpenLoong-Dyn-Control) | 0.3k | [openloong](../../wiki/entities/openloong.md) |
| [lego_loam.md](lego_loam.md) | [RobustFieldAutonomyLab/LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM) | 2.7k | SLAM 选型 |
| [rtabmap.md](rtabmap.md) | [introlab/rtabmap](https://github.com/introlab/rtabmap) | 3.8k | 总览 + 选型 |
| [kimera.md](kimera.md) | [MIT-SPARK/Kimera](https://github.com/MIT-SPARK/Kimera) | 2.1k | SLAM 选型 |
| [open_vins.md](open_vins.md) | [rpng/open_vins](https://github.com/rpng/open_vins) | 2.9k | SLAM 选型 |
| [hdl_graph_slam.md](hdl_graph_slam.md) | [koide3/hdl_graph_slam](https://github.com/koide3/hdl_graph_slam) | 2.3k | SLAM 选型 |
| [voxgraph.md](voxgraph.md) | [ethz-asl/voxgraph](https://github.com/ethz-asl/voxgraph) | 0.5k | SLAM 选型 |
| [lerobot.md](lerobot.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) | 24.4k | [lerobot](../../wiki/entities/lerobot.md) |
| [openvla.md](openvla.md) | [openvla/openvla](https://github.com/openvla/openvla) | 6.3k | [openvla](../../wiki/entities/openvla.md) |
| [mushr.md](mushr.md) | [prl-mushr/mushr](https://github.com/prl-mushr/mushr) | 0.2k | 总览栈 |
| [isaac_ros_visual_slam.md](isaac_ros_visual_slam.md) | [NVIDIA-ISAAC-ROS/isaac_ros_visual_slam](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam) | 1.4k | [isaac-ros-visual-slam](../../wiki/entities/isaac-ros-visual-slam.md) |
| [isaac_ros_nvblox.md](isaac_ros_nvblox.md) | [NVIDIA-ISAAC-ROS/isaac_ros_nvblox](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox) | 0.7k | [isaac-ros-nvblox](../../wiki/entities/isaac-ros-nvblox.md) |

## 关联原始资料（既有）

- [openloong.md](openloong.md) — 青龙全栈；Dyn-Control 为并行研究栈
- [ros2-basics](../../wiki/concepts/ros2-basics.md) — Nav2 / slam_toolbox 的 ROS 2 前提
- [lerobot 实体](../../wiki/entities/lerobot.md) — 本批补充官方 GitHub source 归档
