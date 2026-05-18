# brax

> 来源归档

- **标题：** Brax
- **类型：** repo
- **来源：** Google
- **链接：** https://github.com/google/brax
- **配套论文：** [Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation](https://arxiv.org/abs/2106.13281)（NeurIPS 2021 Datasets and Benchmarks Track）
- **入库日期：** 2026-05-18
- **一句话说明：** 基于 JAX 的可微物理与 RL 训练库；README（0.13.0 起）强调 **`brax/training` 为主维护面**，环境侧推荐转向 **MuJoCo Playground**，纯物理仿真优先 **MJX** 或 **MuJoCo Warp**。
- **沉淀到 wiki：** 是 → [`wiki/entities/brax.md`](../wiki/entities/brax.md)

## 仓库定位（与 README 警告对齐）

- **训练算法**：PPO、SAC、ARS、ES、APG（解析策略梯度）等实现集中在 `brax/training`。
- **环境 `brax/envs`**：官方声明不再作为主推路径；新工作应使用 [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) 等环境与 `brax/training` 组合。
- **物理引擎角色**：若目标是 **MuJoCo 一致动力学 + JAX**，应直接使用 **MJX**（`mujoco-mjx`）或 **MuJoCo Warp**，而不是把 Brax 当作 MuJoCo 的薄封装。

## 为什么值得保留

- 历史上是 **JAX 上大规模刚体仿真 + 可微管线** 的标杆实现之一；论文与后续生态（含与 MJX 的 Colab 教程互链）仍是 **sim2sim / 可微控制** 文献的常见引用栈。
- README 给出的 **维护边界迁移** 对读者做 **技术选型**（「Brax 训练 vs MJX 物理 vs Playground 环境」）有直接价值。

## 对 wiki 的映射

- [Brax（实体页）](../../wiki/entities/brax.md)
