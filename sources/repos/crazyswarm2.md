# Crazyswarm2

> 来源归档

- **标题：** Crazyswarm2
- **类型：** repo
- **链接：** https://github.com/IMRCLab/crazyswarm2
- **Stars：** ~0.2k（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** **大规模 Crazyflie 群体** 起飞与轨迹跟踪框架（ROS2）：动捕/Lighthouse 定位、Python 脚本编排、与 [crazyflie-firmware](crazyflie_firmware.md) 配套。
- **沉淀到 wiki：** [multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**Crazyswarm2** 是 CMU/IMRC Lab 对初代 Crazyswarm 的 **ROS2 重写**，用于：

- 数十至上百架 **微四轴** 同步起飞
- **轨迹集** 上传与碰撞避免（简化模型）
- 灯光秀、编队论文实验

依赖 **动作捕捉或 UWB** 提供全局位姿；不适合 GPS 室外大场景。

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [crazyflie_firmware.md](crazyflie_firmware.md) | 机载固件 |
| [gym_pybullet_drones.md](gym_pybullet_drones.md) | 仿真侧 RL；Crazyswarm 走真机 swarm |
| [ego_planner_swarm.md](ego_planner_swarm.md) | 均为多机协调；尺度与平台不同（微四轴 vs 标准多旋翼） |
