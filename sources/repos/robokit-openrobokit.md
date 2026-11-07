# RoboKit 官方源码归档

- **类型：** repo / optimization / robotics simulation
- **链接：** https://github.com/openrobokit/robokit
- **文档：** https://github.com/openrobokit/robokit/blob/main/docs/guide/overview.md
- **Quickstart：** https://github.com/openrobokit/robokit/blob/main/docs/guide/quickstart.md
- **示例源码：** https://github.com/openrobokit/robokit/blob/main/src/robokit/examples/sim_mpc/10_wuji_cube.py
- **关联项目页：** https://openrobokit.com/robokit/
- **关联论文：** https://openreview.net/forum?id=xfZJ2GKrZG
- **许可证：** MIT（官方仓库 LICENSE）
- **核查日期：** 2026-11-07
- **核对范围：** 官方 README、docs/guide/overview.md、docs/guide/quickstart.md 与完整的 src/robokit/examples/sim_mpc/10_wuji_cube.py。
- **通读：** 已完成

## 工程事实

- `Robot` 支持机器人模型与批量可微运动学；`Objectives/Terms` 组合姿态、接触、碰撞及限位代价；`Optimizers` 对代价使用采样或梯度优化。
- `uv run src/robokit/examples/ik/01_basic.py` 是官方快速示例；`uvx --from robokit python -m robokit.gallery.ik` 可启动交互图库。
- Wuji Hand 2.2 示例通过 `mujoco.MjSpec` 构造手与方块场景，`MjWarpSim` 并行仿真，`SimMPC` 优化关节控制，再经 `mpc.execute` 得到 rollout，最后用 Viser 回放。
- 代码示例中的 Wuji Hand 2.2 **不能推导出**与任何第三方科研平台已完成产品级兼容认证。

## 对 wiki 的映射

- [RoboKit](../../wiki/entities/robokit-openrobokit.md)
