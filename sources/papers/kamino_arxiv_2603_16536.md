# Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies

> 来源归档

- **标题：** Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies
- **类型：** paper
- **作者：** Vassilios Tsounis, Guirec Maloisel, Christian Schumacher, Ruben Grandia, Agon Serifi, David Müller, Chris Amevor, Tobias Widmer, Moritz Bächer
- **机构：** Disney Research, Zurich；NVIDIA, Zurich
- **链接：** https://arxiv.org/abs/2603.16536
- **arXiv：** 2603.16536
- **年份：** 2026
- **入库日期：** 2026-09-05
- **一句话说明：** GPU 原生 PADMM 约束刚体求解器，原生闭链拓扑 + 异构并行世界，在 DR Legs 双足上完成 4096 环境 RL 行走训练。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-kamino.md`](../../wiki/entities/paper-kamino.md)

---

## 关联论文

- **算法基础：** [On Solving the Dynamics of Constrained Rigid Multi-Body Systems with Kinematic Loops](https://arxiv.org/abs/2504.19771)（Tsounis, Grandia, Bächer, 2025）— 极大坐标约束动力学与 PADMM 形式化

## 核心摘录

1. **动机：** 主流 GPU 仿真（Isaac Gym、Brax、MuJoCo/MJX/MJWarp）多用 **树形运动学** 的 $O(n)$ 递推；闭链机构（并联臂、四连杆传动腿、液压挖掘机）常被近似为开链 + 等式约束，带来过约束、大质量比与建模偏置。
2. **方法：** **极大坐标** — 每刚体独立 SE(3) 位姿；关节与 **环闭合** 同为代数约束。前向动力学化为约束反应的对偶问题，用 **Proximal-ADMM** 统一双边关节、关节限位与 De Saxcé 修正的摩擦接触。
3. **计算：** 闭链破坏 Delassus 矩阵的递推结构；小系统用块 Cholesky，大系统用 **warm-started Conjugate Residual** + 块稀疏 matrix-free Delassus 算子。
4. **并行：** **Heterogeneous worlds** — 每个并行环境可有不同刚体/关节/碰撞几何，适合批量多样化机器人。
5. **实证：** **DR Legs** 双足（每腿多四连杆 + 双腿间附加环），单 GPU **4096** 并行环境 RL 训练出可行行走策略 — 首个在 GPU 仿真器上训练的复杂闭链机制 RL 案例。
6. **实现：** NVIDIA Warp + 集成进开源 **Newton** 物理引擎；与 Isaac Lab RL 集成规划中。

## 对 wiki 的映射

| 摘录主题 | 目标 wiki |
|----------|-----------|
| 闭链 vs 开链仿真选型 | [`wiki/concepts/humanoid-parallel-joint-kinematics.md`](../../wiki/concepts/humanoid-parallel-joint-kinematics.md) |
| Newton 求解器后端 | [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md) |
| GPU 批量 RL 仿真 | [`wiki/methods/reinforcement-learning.md`](../../wiki/methods/reinforcement-learning.md) |
| Sim2Real 与机构近似 | [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md) |
