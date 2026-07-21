# isaac_lab

> 来源归档

- **标题：** NVIDIA Isaac Lab
- **类型：** repo
- **来源：** NVIDIA（isaac-sim 组织）
- **链接：** https://github.com/isaac-sim/IsaacLab
- **文档：** https://isaac-sim.github.io/IsaacLab/
- **入库日期：** 2026-07-21（从联合归档拆分为独立 source）
- **一句话说明：** 建立在 Isaac Sim 之上的官方 robot learning 框架；承接 IsaacGymEnvs / Orbit，提供 manager-based 与 direct 两套环境工作流。
- **代码：** https://github.com/isaac-sim/IsaacLab（已开源）
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-lab.md`](../../wiki/entities/isaac-lab.md)

---

## 核心定位

- **仿真底座：** 依赖 [Isaac Sim](../../wiki/entities/isaac-sim.md)（PhysX / USD / 传感器）
- **学习工作流：** `ManagerBasedRLEnv`（模块化 MDP）与 `DirectRLEnv`（单类实现）
- **训练后端：** RSL-RL、rl-games、SKRL 等

---

## 关联档案

- 联合索引：[`isaac_gym_isaac_lab.md`](./isaac_gym_isaac_lab.md)
- Isaac Sim：[`isaac_sim.md`](./isaac_sim.md)
- 前代 Gym：见联合归档与 [`wiki/entities/isaac-gym.md`](../../wiki/entities/isaac-gym.md)
