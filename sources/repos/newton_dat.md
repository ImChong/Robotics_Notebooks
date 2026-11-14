---
type: source
title: "NVIDIA Newton — SolverVBD 与 Planar-DAT / 刚软截断源码归档"
url: "https://github.com/newton-physics/newton"
date: 2026-11-14
status: ingested
tags: [repo, simulation, contact, multi-physics, nvidia, newton]
---

# NVIDIA Newton — SolverVBD 与 Planar-DAT / 刚软截断源码归档

- **仓库：** <https://github.com/newton-physics/newton>
- **开源状态：** 已开源（Apache-2.0，DAT 逻辑直接集成于 Newton 主线 `SolverVBD` 求解器、碰撞管线与示例脚本中）。
- **核对范围：** 核对官方仓库 `newton-physics/newton` 中与 DAT 相关的核心模块与可运行示例：
  - `newton/examples/multiphysics/example_vbd_dat_rigid_soft.py`（刚–软双层布料蹦床 + 刚体投射物无穿透算例，`python -m newton.examples vbd_dat_rigid_soft`）
  - `newton/_src/solvers/vbd/solver_vbd.py`（`SolverVBD.__init__`、`step`、`_initialize_rigid_bodies`、`_initialize_particles`、`_solve_particle_iteration`、`_solve_rigid_body_iteration`、`_rigid_penetration_free_truncation`、`_penetration_free_truncation`）
  - `newton/_src/solvers/vbd/particle_vbd_kernels.py`（`place_dat_division_plane`、`planar_truncation_t`、`apply_planar_truncation_parallel_by_collision`）
  - `newton/_src/solvers/vbd/rigid_vbd_kernels.py`（`apply_rigid_soft_truncation`、`apply_body_truncation_ts`）
  - `newton/_src/solvers/vbd/interval_arithmetic.py`（实验性 `rigid_soft_dat_use_interval_arithmetic` 旋转轨迹区间验证辅助函数）
- **通读：** 已完成

## 关键实现对齐细节

1. **运行入口：** `newton/examples/multiphysics/example_vbd_dat_rigid_soft.py` 构建两层布料（底层四边固定、顶层自由）与 6 个刚体（球、盒、胶囊），配置 `CollisionPipeline(..., enable_rigid_soft_full_surface_contact=True)` 并交给 `SolverVBD(..., rigid_soft_enable_dat=True, particle_enable_self_contact=True, dat_conservative_bound_relaxation=0.85)` 托管碰撞调度。
2. **求解步与截断插入点（`SolverVBD.step`）：**
   - 初始化阶段：`_initialize_rigid_bodies` 内执行 `_rigid_penetration_free_truncation`（当开启 `rigid_soft_enable_dat` 时）；`_initialize_particles` 内执行 `_penetration_free_truncation`。
   - VBD 内循环（`for iter_num in range(self.iterations)`）：按周期调度执行 `_mid_step_detection`，随后交替调用 `_solve_rigid_body_iteration`（内部结尾调用 `_rigid_penetration_free_truncation`）与 `_solve_particle_iteration`（内部结尾调用 `_penetration_free_truncation`）。
   - 收尾阶段：`_finalize_rigid_bodies` 与 `_finalize_particles` 回写最终位置与速度。
3. **碰撞对并行与运动预算：** `_penetration_free_truncation` 先把 `truncation_ts`（与 `body_truncation_ts`）重置为 `1.0`，再启动 `apply_planar_truncation_parallel_by_collision`（颗粒自接触）与 `apply_rigid_soft_truncation`（刚–软接触），通过 `place_dat_division_plane` 与 `planar_truncation_t` 对双侧累加位移实施平面截断，并以每检测间隔预算 `0.5 * dat_conservative_bound_relaxation * gap` 作为各向同性退化兜底；例程在 CUDA Graph 模式下每帧仅增约数个百分点耗时（23.8 ms → 24.5 ms），未启用 Graph 时因小内核发射较多开销约 40%（35.9 ms → 49.6 ms）。
