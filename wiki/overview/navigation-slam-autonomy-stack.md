---

type: overview
tags: [navigation, slam, ros2, autoware, lidar, vio, mobile-robot, autonomous-driving, openloong]
status: complete
updated: 2026-07-21
related:
  - ../entities/python-robotics.md
  - ../entities/navigation2.md
  - ../entities/slam-toolbox.md
  - ../entities/cartographer.md
  - ../entities/fast-lio.md
  - ../entities/lio-sam.md
  - ../entities/orb-slam3.md
  - ../entities/paper-vs-graphs-visual-slam-scene-graph.md
  - ../entities/paper-glob3r.md
  - ../methods/lingbot-map.md
  - ../entities/vins-fusion.md
  - ../entities/open-vins.md
  - ../entities/rtabmap.md
  - ../entities/autoware.md
  - ../entities/isaac-ros-visual-slam.md
  - ../entities/isaac-ros-nvblox.md
  - ../entities/openloong-dyn-control.md
  - ../entities/lerobot.md
  - ../entities/openvla.md
  - ../entities/mushr.md
  - ../entities/aeris-10-plfm-radar.md
  - ../entities/rhino-auto.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
  - ../concepts/ros2-basics.md
  - ../methods/vla.md
  - ../tasks/vision-language-navigation.md
sources:
  - ../../sources/repos/python_robotics.md
  - ../../sources/repos/navigation_slam_autonomy_stack_catalog.md
  - ../../sources/repos/navigation2.md
  - ../../sources/repos/slam_toolbox.md
  - ../../sources/repos/cartographer.md
  - ../../sources/repos/fast_lio.md
  - ../../sources/repos/autoware.md
  - ../../sources/repos/isaac_ros_visual_slam.md
  - ../../sources/repos/isaac_ros_nvblox.md
  - ../../sources/repos/plfm_radar.md
summary: "移动机器人导航与 SLAM 开源栈总览：Nav2 + 2D SLAM、LiDAR/VIO 里程计、Autoware 自动驾驶、Isaac ROS 加速感知建图，并与 OpenLoong 动力学、LeRobot/OpenVLA 具身学习对照分层选型。"
---

# 导航·SLAM·自动驾驶开源栈总览

> **本页回答：** 做地面移动机器人或低速自动驾驶时，**Nav2、slam_toolbox、Cartographer、FAST-LIO、VINS、Autoware、Isaac ROS** 各在什么层？如何与仓库主线的 **腿式/人形控制**、**VLA 操作** 区分而不混栈？

## 一句话总结

**ROS 2 导航闭环** 以 [Navigation2](../entities/navigation2.md) 为中枢，上游接 **2D SLAM**（[slam_toolbox](https://github.com/SteveMacenski/slam_toolbox)、[Cartographer](https://github.com/cartographer-project/cartographer)）或 **3D 定位**（[FAST-LIO](https://github.com/hku-mars/FAST_LIO)、[LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) 等，见 [LiDAR/VIO 选型](../comparisons/lidar-slam-lio-vio-selection.md)）；**乘用车级全栈** 走 [Autoware](../entities/autoware.md)；**Jetson + GPU 感知建图** 用 [Isaac ROS Visual SLAM](../entities/isaac-ros-visual-slam.md) + [nvblox](../entities/isaac-ros-nvblox.md)。**人形全身 MPC/WBC** 不属于本栈，见 [OpenLoong-Dyn-Control](https://github.com/loongOpen/OpenLoong-Dyn-Control) 与 [OpenLoong](../entities/openloong.md)。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| ROS 2 | Robot Operating System 2 | 导航与系统集成常用中间件 |
| LiDAR | Light Detection and Ranging | 2D/3D 激光 SLAM 主传感器 |
| MPC | Model Predictive Control | 局部路径/轨迹跟踪优化 |
| VLA | Vision-Language-Action | 语义导航与端到端驾驶新路线 |

## 为什么重要

- 仓库主线以 **腿式/人形运控与模仿学习** 为主，但 **移动底盘、AMR、低速无人车** 仍依赖成熟的 **定位—建图—规划** 栈；与 [VLN](../tasks/vision-language-navigation.md) 共享「感知—决策—执行」分层思想，**执行层接口不同**（`cmd_vel` / Ackermann vs 关节力矩）。
- 21 仓覆盖从 **教学小车 MuSHR** 到 **Autoware Universe** 的完整谱系；常见错误：用 ORB-SLAM3 直接驱动 Nav2 而无坐标系/频率对齐，或把 FAST-LIO 当全局规划器。
- [LeRobot](../entities/lerobot.md) / [OpenVLA](../entities/openvla.md) 解决 **操作与策略**，不替代 Nav2；二者常在「同一机器人」上 **分层共存**（导航栈 + 机械臂 VLA）。
- **[PythonRobotics](../entities/python-robotics.md)** 提供 **算法层入门**（EKF、A*、RRT*、DWA、Stanley/MPC 等可运行 Python 示例），与 Nav2 **互补而非替代**：先建立直觉，再读 ROS 2 工程集成。
- **[DimOS（Dimensional）](../entities/dimensionalos-dimos.md)** 提供 **ROS-optional** 的 Python 导航 + **MCP agent** 蓝图（Unitree Go2/G1 等），默认 LCM 连模块流；适合快速验证「SLAM + 自然语言控车」而不先搭完整 ROS 2 工作空间。

## 流程总览

```mermaid
flowchart TB
  subgraph sense["感知 / 状态估计"]
    LIO["FAST-LIO / LIO-SAM\n3D 里程计"]
    VIO["VINS / ORB-SLAM3 / OpenVINS\n视觉惯性"]
    SLAM2D["slam_toolbox / Cartographer\n2D 地图"]
    ISAAC["Isaac ROS\ncuVSLAM + nvblox TSDF"]
  end
  subgraph map["地图 / 代价"]
    MAP2D["2D occupancy"]
    ESDF["3D ESDF / costmap"]
  end
  subgraph nav["规划与控制"]
    NAV2["Nav2\nBT + planner + controller"]
    AUTO["Autoware\n感知-预测-规划-控制"]
  end
  subgraph plat["平台"]
    AMR["差速 / 全向 AMR"]
    CAR["乘用车 / 园区车"]
    EDU["MuSHR 教学车"]
  end

  LIO --> ESDF
  VIO --> MAP2D
  SLAM2D --> MAP2D
  ISAAC --> ESDF
  MAP2D --> NAV2
  ESDF --> NAV2
  NAV2 --> AMR
  NAV2 --> EDU
  AUTO --> CAR
  sense -.-> AUTO
```

## 分层选型表

| 需求 | 优先选型 | 备选 | 避免 |
|------|----------|------|------|
| ROS 2 标准室内 AMR 导航 | Nav2 + slam_toolbox | Cartographer + AMCL | 无地图直接 DWB 调参 |
| 大规模 / lifelong 2D 地图 | slam_toolbox | Cartographer | 把 3D LIO 当地图服务器却不降维 |
| 3D 激光高频里程计 | FAST-LIO | LIO-SAM | 期望内置全局语义地图 |
| 因子图 + GPS 融合室外 | LIO-SAM | hdl_graph_slam | 室内稀疏特征硬上 GPS |
| 地面车辆 3D 激光 SLAM | LeGO-LOAM | FAST-LIO | 无人机空中场景硬用地面假设 |
| 视觉惯性无人机/手持 | VINS-Fusion | ORB-SLAM3 | 忽略 IMU 标定 |
| 研究向 VIO 滤波器 | OpenVINS | VINS-Fusion | 与 Nav2 无 ROS 封装时裸跑 |
| RGB-D 一体建图导航 | RTAB-Map | — | 与高动态场景无 motion compensation |
| 开源 L4 全栈 | Autoware | — | 用 Nav2 插件冒充感知预测 |
| Jetson 视觉 SLAM + 3D 避障 | Isaac ROS VSLAM + nvblox | — | x86 无 GPU 强上 nvblox |
| 算法预习 / 无 ROS 环境 | PythonRobotics | MuSHR 课程 | 直接上 Autoware 却不理解 DWA/A* |
| 导航课 / 非完整小车 | MuSHR | TurtleBot3（既有资料） | 当人形 WBC 教程 |
| 机械臂抓取策略 | LeRobot / OpenVLA | — | 用 Nav2 输出关节力矩 |
| 开源相控阵雷达 / 主动测距原型 | [AERIS-10](../entities/aeris-10-plfm-radar.md) | 商用毫米波雷达模组 | 无 ROS 驱动时直接接 Nav2；或当稠密 LiDAR SLAM 用 |

## 各层角色摘要

### ROS 2 导航层

- **[Navigation2](../entities/navigation2.md)**：行为树导航器、全局/局部规划器插件（NavFn、Smac、DWB/RPP 等）、代价地图、恢复行为。
- **[MuSHR](../entities/mushr.md)**：低成本阿克曼平台 + 课程式 ROS 导航实验。

### 2D SLAM

- **[SLAM Toolbox](../entities/slam-toolbox.md)**：Karto 后端、序列化大地图、定位/建图模式切换；Nav2 社区常用搭档。
- **[Cartographer](../entities/cartographer.md)**：子图 + scan matching + 位姿图；2D/3D 均可，ROS 封装成熟。

### 3D LiDAR 与图优化

- **[FAST-LIO](../entities/fast-lio.md)**：ikd-Tree + 迭代 ESKF，强调 **速度** 与鲁棒性。
- **[LIO-SAM](../entities/lio-sam.md)**：GTSAM 因子图，易接 **GPS** 与回环。
- **[LeGO-LOAM](../entities/lego-loam.md)**：地面分割与地面优化，适合 **起伏地形**。
- **[hdl_graph_slam](../entities/hdl-graph-slam.md)**：NDT 前端 + g2o，偏 **室外大场景**。
- **[Voxgraph](../entities/voxgraph.md)**：Voxblox TSDF 子图对齐，多会话建图。

### 视觉 / VIO

- **[ORB-SLAM3](../entities/orb-slam3.md)**：多地图、视觉/视觉-惯性；研究基准强，工程需自行对接 ROS 2。
- **[vS-Graphs](../entities/paper-vs-graphs-visual-slam-scene-graph.md)**：ORB-SLAM3 扩展，在线构建可优化 **3D 场景图**（墙/房间/楼层），适合需要 **布局级语义** 的室内 RGB-D SLAM 研究。
- **[VINS-Fusion](../entities/vins-fusion.md)**：优化式多传感器；支持 GPS 全局融合。
- **[OpenVINS](../entities/open-vins.md)**：MSCKF 系，便于 **算法对比实验**。
- **[OpenVSLAM](../entities/openvslam.md)** → 社区多迁移至 **stella_vslam** 分支维护。
- **[Kimera](../entities/kimera.md)**：Kimera-VIO + RPGO + Semantics 模块化 **语义 SLAM**。
- **[RTAB-Map](../entities/rtabmap.md)**：RGB-D/激光多模态建图与记忆管理。
- **[LingBot-Map](../methods/lingbot-map.md)**：流式前馈 3D 几何基础模型（在线位姿/稠密几何）；非经典特征 SLAM，可作视觉几何上游。
- **[Glob3R](../entities/paper-glob3r.md)**：基础模型引导的**离线全局 SfM**（dense warp tracks → 运动平均 + BA）；适合事后高精度建图/渲染，推理代码截至 2026-07-21 仍为占位仓。

### 主动雷达（补充，非 ROS 原生）

- **[AERIS-10（PLFM_RADAR）](../entities/aeris-10-plfm-radar.md)**：开源 **10.5 GHz** 脉冲 LFM **相控阵**（3 km / 20 km 双版本），FPGA 脉冲压缩 + CFAR，Python 地图 GUI；面向 **无人机/野外主动测距** 实验。与 LiDAR SLAM **互补**：擅长远距离动目标与全天候测距，**不提供** 现成 `sensor_msgs`/Nav2 插件，接入本栈需自建驱动与时间同步。

### 自动驾驶与 NVIDIA 栈

- **[Autoware](../entities/autoware.md)**：Autoware Core/Universe 组件化，面向 **道路场景**。
- **车规大算力 SoC（对照）：** 乘用车域控除 x86/Jetson 外，亦有 **>500 TOPS** 级 **ASIL-D** 芯片路线（如 [辉羲智能光至 R1](../entities/rhino-auto.md) 的 **RINA 无图城区方案** 叙事）；与 ROS 2 移动栈接口层不同，但同属 **感知–规划边缘算力** 选型语境。
- **[Isaac ROS Visual SLAM](../entities/isaac-ros-visual-slam.md)** / **[nvblox](../entities/isaac-ros-nvblox.md)**：GPU 加速，与 Nav2 **3D costmap** 衔接。

### 与本仓库其它主线

| 子系统 | 本页栈 | 腿式/人形主线 |
|--------|--------|----------------|
| 状态估计 | 位姿 + 地图 | 浮基 + 接触 + 关节 |
| 规划输出 | `cmd_vel` / 轨迹 | 足端/质心/关节参考 |
| 学习 | VLA/IL 操作 | RL locomotion / tracking |
| 代表仓 | Nav2、FAST-LIO | [OpenLoong-Dyn-Control](../entities/openloong-dyn-control.md)、[legged_gym](https://github.com/leggedrobotics/legged_gym) |

## 常见误区

- **把 SLAM 当规划器**：SLAM 只提供 **位姿与地图**；路径跟踪仍在 Nav2 controller 或 Autoware 控制模块。
- **2D / 3D 混用不投影**：3D LIO 输出需 **投影或切片** 才能喂给经典 2D costmap，或使用 nvblox 等 3D 代价源。
- **OpenVLA 与 Nav2 二选一**：VLA 管 **操作空间**；底盘导航仍常需 Nav2，除非端到端策略直接输出底盘速度且已安全验证。

## 参考来源

- [sources/repos/navigation_slam_autonomy_stack_catalog.md](../../sources/repos/navigation_slam_autonomy_stack_catalog.md)
- 本页所列各 `sources/repos/*.md` 条目

## 关联页面

- [PythonRobotics](../entities/python-robotics.md) — 自主导航算法入门代码与教材
- [LiDAR / LIO / VIO 选型对比](../comparisons/lidar-slam-lio-vio-selection.md)
- [Navigation2](../entities/navigation2.md)
- [Autoware](../entities/autoware.md)
- [ROS 2 基础](../concepts/ros2-basics.md)
- [VLA 方法](../methods/vla.md)
- [LingBot-Map](../methods/lingbot-map.md) — 流式前馈视觉几何
- [Glob3R](../entities/paper-glob3r.md) — 全局 SfM + 3D 基础模型（离线精炼）

## 推荐继续阅读

- [PythonRobotics 在线教材](https://atsushisakai.github.io/PythonRobotics/)（算法入门）
- [Nav2 官方文档](https://docs.nav2.org/)
- [Autoware 文档](https://autowarefoundation.github.io/autoware-documentation/)
- [Isaac ROS](https://nvidia-isaac-ros.github.io/)
