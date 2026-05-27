# AirSim

> 来源归档

- **标题：** Microsoft AirSim
- **类型：** repo
- **链接：** https://github.com/microsoft/AirSim
- **Stars：** ~18.2k（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** 基于 Unreal Engine / Unity 的自动驾驶与无人机 **高保真视觉仿真**：RGB/深度/分割、天气、多机，可与 PX4 SITL 或简单内置动力学耦合。
- **沉淀到 wiki：** [airsim](../../wiki/entities/airsim.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

AirSim 提供 **Unreal/Unity 场景 + 车辆 API**（Python/C++），侧重 **计算机视觉、SLAM、避障、Sim2Real 视觉迁移**。多旋翼模式支持 **SimpleFlight** 内置控制器或 **PX4** 外环。

**注意：** 微软已宣布项目进入维护模式；新研究可评估 [Colosseum](https://github.com/CodexLabsLLC/Colosseum) 等 fork，但 AirSim 仍是文献与课程中最常引用的视觉 UAV 仿真基线之一。

---

## 能力摘要

| 能力 | 说明 |
|------|------|
| 传感器 | 相机、深度、IMU、GPS、LiDAR（配置依赖） |
| 环境 | 程序化天气、时间、可导入自定义 UE 地图 |
| API | `moveOnPath`、速度控制、图像抓取、状态重置 |
| 集成 | ROS wrapper、PX4 HITL/SITL 桥接文档 |

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [px4_autopilot.md](px4_autopilot.md) | 推荐组合做「视觉仿真 + 真飞控栈」 |
| [flightmare.md](flightmare.md) | 同为视觉/RL 向；Flightmare 更偏研究型轻量引擎 |
| [xtdrone.md](xtdrone.md) | 国内教学多用 Gazebo；AirSim 偏 UE 画质 |
