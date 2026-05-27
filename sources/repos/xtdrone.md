# XTDrone

> 来源归档

- **标题：** XTDrone
- **类型：** repo
- **链接：** https://github.com/robin-shaun/XTDrone
- **Stars：** ~1.6k（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** 基于 **PX4 + ROS + Gazebo** 的无人机仿真平台：多机型、视觉 SLAM、目标检测、编队与强化学习实验教程（中文社区友好）。
- **沉淀到 wiki：** [multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**XTDrone** 面向 **教学与算法验证**，在 Gazebo 中集成 iris/typhoon 等模型，提供：

- PX4 SITL 一键启动脚本
- **视觉 + 控制** 实验（跟踪、降落、多机）
- 与 **ROS Melodic/Noetic** 工作空间组织范例

适合作为「**从 Gazebo 到真机 PX4**」的完整链路参考，而非单一算法库。

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [px4_autopilot.md](px4_autopilot.md) | 飞控核心 |
| [ego_planner_swarm.md](ego_planner_swarm.md) | 规划层可替换/叠加 |
| [airsim.md](airsim.md) | 视觉质量更高但栈不同；XTDrone 走 Gazebo 生态 |
