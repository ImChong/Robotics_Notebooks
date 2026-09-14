# Mixed Material Point Methods for Stiff Elastoplasticity

> 来源归档

- **标题：** Mixed Material Point Methods for Stiff Elastoplasticity
- **类型：** paper
- **作者：** Gilles Daviet
- **机构：** NVIDIA Research（PRL）
- **链接：** https://doi.org/10.1145/3811345
- **项目页：** https://research.nvidia.com/labs/prl/mixed_mpm/
- **会议：** SIGGRAPH 2026
- **年份：** 2026
- **入库日期：** 2026-09-13
- **一句话说明：** 将 Daviet & Bertails-Descoubes [2016] 混合离散推广到有限应变粘弹性与更一般流动法则，隐式积分得到对称、良定优化问题与高效 GPU 求解器；Newton 一等 MPM 模块，支持与刚体双向耦合。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-mixed-mpm-stiff-elastoplasticity.md`](../../wiki/entities/paper-mixed-mpm-stiff-elastoplasticity.md)

---

## 核心摘录

1. **动机：** 经典 MPM 在 **刚性弹粘塑性**（沙、雪、混凝土、近不可压流体）上需小步长或昂贵 stencil；混合速度–应力离散可兼顾 CFL 步长与稳定性。
2. **方法：** 扩展 [Daviet & Bertails-Descoubes 2016a] 混合格式至 **有限应变粘弹性** 与更一般 **流动法则**；隐式积分 → **对称、良定** 优化 + **紧凑 stencil** GPU 求解。
3. **材料谱：** 颗粒（沙）、雪（裂缝传播）、弹性固体、**近不可压流体**、刚性弹塑性断裂。
4. **刚体耦合：** 设计为与刚体求解器 **紧耦合** — 颗粒对关节角色/障碍反作用；演示腿式机器人 **双向** 步态调整与交互沙盘。
5. **Newton 集成：** 项目页写明 **first-party module** in Newton physics engine；与 [`SolverImplicitMPM`](../../wiki/entities/newton-physics.md) 后端同栈。
6. **离散选型：** 多种速度–应力对可切换；**trilinear 速度** 常在精度/吞吐上具竞争力。

## 对 wiki 的映射

| 摘录主题 | 目标 wiki |
|----------|-----------|
| Newton MPM 后端 | [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md) |
| 多物理刚–软–颗粒 | [`wiki/entities/paper-dat-divide-and-truncate.md`](../../wiki/entities/paper-dat-divide-and-truncate.md) |
| 仿真物理保真 | [`wiki/queries/simulation-physics-fidelity.md`](../../wiki/queries/simulation-physics-fidelity.md) |
| Genesis 等 MPM 对照 | [`wiki/entities/genesis-world-10.md`](../../wiki/entities/genesis-world-10.md) |
