# Flightmare

> 来源归档

- **标题：** Flightmare
- **类型：** repo
- **链接：** https://github.com/uzh-rpg/flightmare
- **Stars：** ~1.4k（2026-05）
- **入库日期：** 2026-05-27
- **一句话说明：** 苏黎世大学 RPG 的 **灵活四旋翼仿真器**：Unity 渲染 + 可配置动力学，面向敏捷飞行、感知与 RL（高吞吐、多环境并行）。
- **沉淀到 wiki：** [multirotor-simulation-planning-control-stack](../../wiki/overview/multirotor-simulation-planning-control-stack.md)

---

## 核心定位

**Flightmare** 强调 **研究迭代速度**：分离 **渲染客户端** 与 **物理/控制后端**，支持多 quad 并行、自定义障碍与传感器配置。常用于 **敏捷机动、避障、端到端策略** 论文基线。

与 AirSim 相比更轻、更易批量化；与 PyBullet Gym 相比视觉与场景更丰富。

---

## 与本批资料关系

| 资料 | 关系 |
|------|------|
| [gym_pybullet_drones.md](gym_pybullet_drones.md) | 同为 RL 友好；Flightmare 偏视觉+Unity |
| [ego_planner_swarm.md](ego_planner_swarm.md) | 规划算法可在不同仿真器间迁移验证 |
| [crazyswarm2.md](crazyswarm2.md) | 真机 swarm 对照 |
