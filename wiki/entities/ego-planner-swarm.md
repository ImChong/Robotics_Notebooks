---

type: entity
tags: [repo, planning, swarm, esdf, uav, ros, zju-fast-lab, zju]
status: complete
updated: 2026-06-14
related:
  - ../overview/multirotor-simulation-planning-control-stack.md
  - ./px4-autopilot.md
  - ./mavsdk.md
  - ./flightmare.md
  - ./airsim.md
  - ./paper-mighty-hermite-spline-trajectory-planning.md
sources:
  - ../../sources/repos/ego_planner_swarm.md
summary: "EGO-Planner Swarm 是 ZJU FAST Lab 的单/多机局部轨迹规划器：ESDF 地图 + B-spline 优化，输出可对接 PX4 Offboard 的位置/速度设定点。"
---

# EGO-Planner Swarm

**EGO-Planner Swarm**（[ZJU-FAST-Lab/ego-planner-swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)）实现 **未知/半已知环境** 下的快速 **局部重规划**，并扩展 **多机 swarm 避碰**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SLAM | Simultaneous Localization and Mapping | 同步定位与建图 |
| LiDAR | Light Detection and Ranging | 激光雷达，地形感知与建图主传感器 |
| VIO | Visual-Inertial Odometry | 视觉-惯性里程计，融合相机与 IMU 估计位姿 |

## 为什么重要

- **规划—控制分界清晰**：输出轨迹/设定点，不替代 [PX4](./px4-autopilot.md) 姿态内环。
- **ESDF + B-spline** 是近年多旋翼自主导航文献与工程中的常见组合，便于与 SLAM/深度相机管线对接。
- 与 [Flightmare](./flightmare.md)、[AirSim](./airsim.md) 等仿真器组合做 **感知—规划—控制** 闭环验证。

## 核心结构/机制

1. **感知/建图**（常外接）：深度或 LiDAR → 占据/ESDF 更新  
2. **前端**：路径搜索或引导路径  
3. **后端**：均匀 B-spline 优化（平滑、动力学、障碍约束）  
4. **Swarm**：机间碰撞代价或协调策略  
5. **执行**：ROS 话题 → MAVROS/[MAVSDK](./mavsdk.md) Offboard → PX4  

**Rebound replanning**：当前轨迹不可行时局部反弹式重优化，适合动态障碍。

## 常见误区或局限

- **误区：有 planner 即可无定位** — 仍需可靠位姿（VIO、动捕、GPS 等）。
- **局限：算力与传感器标定** — 深度噪声会直接反映在 ESDF 与轨迹可行性上。
- **局限：极端动态障碍** — 需额外安全层（限速、禁飞区、人工接管）。
- **对照：[MIGHTY](./paper-mighty-hermite-spline-trajectory-planning.md)** — 同层软约束 UAV 规划器，用 **Hermite 联合时空优化** 在论文 benchmark 中报告更短飞行/求解时间；EGO 优势在 **ROS 1/2 生态与 swarm 扩展**。

## 参考来源

- [sources/repos/ego_planner_swarm.md](../../sources/repos/ego_planner_swarm.md)
- [ZJU-FAST-Lab/ego-planner-swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm)

## 关联页面

- [多旋翼栈总览](../overview/multirotor-simulation-planning-control-stack.md)
- [PX4 Autopilot](./px4-autopilot.md)
- [MAVSDK](./mavsdk.md)
- [MIGHTY（Hermite 样条规划 · RA-L 2026）](./paper-mighty-hermite-spline-trajectory-planning.md)

## 推荐继续阅读

- [EGO-Planner 原版仓库](https://github.com/ZJU-FAST-Lab/ego-planner) — 单机算法与论文入口
