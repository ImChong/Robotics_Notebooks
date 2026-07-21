# isaac_gym

> 来源归档

- **标题：** NVIDIA Isaac Gym
- **类型：** repo / platform（legacy）
- **来源：** NVIDIA
- **链接：** https://developer.nvidia.com/isaac-gym
- **论文：** Makoviychuk et al., *Isaac Gym: High Performance GPU Based Physics Simulation For Robot Learning* (2021) — https://arxiv.org/abs/2108.10470
- **入库日期：** 2026-07-21（从联合归档拆分为独立 source）
- **一句话说明：** 早期 GPU 并行 RL 仿真框架（PhysX + gymapi/gymtorch tensor API）；官方已 deprecated，新实验应迁 Isaac Lab。
- **代码状态：** Preview / legacy；官方建议迁移至 Isaac Lab（以 NVIDIA 开发者页为准）
- **沉淀到 wiki：** 是 → [`wiki/entities/isaac-gym.md`](../../wiki/entities/isaac-gym.md)

---

## 核心定位

- **物理：** PhysX GPU
- **API：** `gymapi`（仿真对象）+ `gymtorch`（GPU tensor 包装）
- **生态上层：** IsaacGymEnvs、`legged_gym` 等

---

## 关联档案

- 联合索引：[`isaac_gym_isaac_lab.md`](./isaac_gym_isaac_lab.md)
- 后继：[`isaac_lab.md`](./isaac_lab.md)、[`isaac_sim.md`](./isaac_sim.md)
