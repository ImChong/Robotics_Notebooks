# ego-planner-swarm

> 来源归档

- **标题：** EGO-Planner Swarm
- **类型：** repo
- **链接：** https://github.com/ZJU-FAST-Lab/ego-planner-swarm
- **Stars：** ~2.0k（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** 浙大 FAST Lab 的 **单/多机局部轨迹规划**：基于 ESDF 地图的 B-spline 优化，支持 swarm 避碰与 ROS 仿真/真机管线。
- **沉淀到 wiki：** [ego-planner-swarm](../../wiki/entities/ego-planner-swarm.md)、[multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**EGO-Planner** 系列面向 **未知/半已知环境** 的快速重规划：前端路径搜索 + 后端 **均匀 B-spline** 优化（平滑性、动力学、避障约束）。**swarm** 分支增加多机间碰撞代价与分布式/集中式协调接口。

典型栈：**深度相机 / LiDAR → 建图（如 FIESTA）→ planner → 位置控制 → PX4 Offboard**。

---

## 关键机制

- **ESDF** 距离场加速碰撞查询
- **Rebound replanning**：轨迹不可行时局部反弹式重优化
- 与 **PX4**、仿真器（Gazebo、自研）通过 ROS 话题衔接

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [px4_autopilot.md](px4_autopilot.md) | 常见执行层飞控 |
| [xtdrone.md](xtdrone.md) | 教学仿真可承载类似规划实验 |
| [flightmare.md](flightmare.md) | 感知/规划算法的高帧率仿真前端 |
