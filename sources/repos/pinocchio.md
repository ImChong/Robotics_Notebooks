# pinocchio

> 来源归档

- **标题：** pinocchio
- **类型：** repo
- **来源：** Stack of Tasks (LAAS-CNRS)
- **链接：** https://github.com/stack-of-tasks/pinocchio
- **入库日期：** 2026-04-11
- **一句话说明：** 机器人运动学、动力学和导数计算的底层引擎，是 TSID、WBC、trajectory optimization 工具链的关键基础设施。
- **沉淀到 wiki：** 是 → [`wiki/entities/pinocchio.md`](../wiki/entities/pinocchio.md)
- **轻量对照：** [dynibo](./dynibo.md) — Rust 树状 URDF + Workspace 零分配；以 Pinocchio 作 oracle/bench
- **重力 API：** `computeGeneralizedGravity(model, data, q)` → $g(q)$；带外载用 `computeStaticTorque`（$g(q)-J^\top f_{\mathrm{ext}}$）。控制用法见 [重力补偿](../../wiki/concepts/gravity-compensation.md)。
