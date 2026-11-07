# pinocchio

> 来源归档

- **标题：** pinocchio
- **类型：** repo
- **来源：** Stack of Tasks (LAAS-CNRS)
- **链接：** https://github.com/stack-of-tasks/pinocchio
- **入库日期：** 2026-04-11（首次）；2026-11-07（README 与关键头文件核对）
- **核对范围：** 读官方 README（定位、安装方式 conda / pip、Crocoddyl / Aligator 等下游），并核对 `algorithm/compute-all-terms.hpp`（`computeAllTerms` 等价于 FK + CRBA + 非线性项 + 关节雅可比 + 质心等）、`algorithm/crba.hpp`（只算 $M$ 的**上三角**）、`algorithm/frames.hpp`（`getFrameJacobian` 须先调用 `computeJointJacobians`；参考系 LOCAL / LOCAL_WORLD_ALIGNED / WORLD）、`algorithm/rnea.hpp`（偏置项存于 `data.nle`）与 `examples/inverse-kinematics.py`、`examples/meshcat-viewer.py`。未通读完整文档站与论文，未运行代码。
- **通读：** 已完成
- **一句话说明：** 机器人运动学、动力学和导数计算的底层引擎，是 TSID、WBC、trajectory optimization 工具链的关键基础设施。
- **沉淀到 wiki：** 是 → [`wiki/entities/pinocchio.md`](../wiki/entities/pinocchio.md)
- **轻量对照：** [dynibo](./dynibo.md) — Rust 树状 URDF 固定/浮动基计算与多语言绑定；以 Pinocchio 作 oracle/性能对照
- **重力 API：** `computeGeneralizedGravity(model, data, q)` → $g(q)$；带外载用 `computeStaticTorque`（$g(q)-J^\top f_{\mathrm{ext}}$）。控制用法见 [重力补偿](../../wiki/concepts/gravity-compensation.md)。
