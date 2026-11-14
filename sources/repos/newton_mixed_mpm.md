---
type: source
title: "NVIDIA Newton — SolverImplicitMPM 源码与双向耦合归档"
url: "https://github.com/newton-physics/newton"
date: 2026-11-14
status: ingested
tags: [repo, simulation, mpm, elastoplasticity, nvidia, newton]
---

# NVIDIA Newton — SolverImplicitMPM 源码与双向耦合归档

- **仓库：** <https://github.com/newton-physics/newton>
- **开源状态：** 已开源（Apache-2.0，Mixed MPM 核心实现即 Newton 内建 `SolverImplicitMPM` 求解器、`warp.fem` 流变核及 `mpm_*` / `mujoco_mpm_coupled_solver` 系列示例）。
- **核对范围：** 核对官方仓库 `newton-physics/newton` 中核心求解器与代表性示例：
  - `newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py`（`SolverImplicitMPM.step`、`_particles_to_cells`、`_step_impl`、`_rasterize_colliders`、`_compute_unconstrained_velocity`、`_build_collider_rigidity_operator`、`_build_elasticity_system`、`_build_plasticity_system`、`_load_warmstart`、`_solve_rheology`、`_update_particles`、`_save_data`、`collect_collider_impulses`，以及 `CouplingInterface` 的 `coupling_rewind_proxy_body` / `coupling_harvest_proxy_wrenches`）
  - `newton/_src/solvers/implicit_mpm/solve_rheology.py`（`solve_rheology`：支持 `gauss-seidel`/`gs`、`gs-soa`、`gs-batched`、`jacobi`、`cg`/`cr`/`gmres` 链式热启动、Delassus 特征基旋转与条件 CUDA Graph 循环）
  - `newton/_src/solvers/implicit_mpm/rasterized_collisions.py`（`rasterize_collider` 与 SDF/法向/速度场光栅化）
  - `newton/examples/mpm/example_mpm_twoway_coupling.py`（刚体箱落入沙堆，MuJoCo + `SolverImplicitMPM` 双向冲量–速度回退循环）
  - `newton/examples/mpm/example_mpm_anymal.py`（ANYmal C 强化学习策略在隐式 MPM 颗粒地形行走）
  - `newton/examples/multiphysics/example_mujoco_mpm_coupled_solver.py`（`SolverCoupledProxy` 统一调度 MuJoCo/Kamino + `SolverImplicitMPM`）
- **通读：** 已完成

## 关键实现对齐细节

1. **典型调用模式：**
   - **独立/双向示例（`example_mpm_twoway_coupling.py`）：** 用 `SolverImplicitMPM.register_custom_attributes` 注册颗粒属性后实例化 `SolverImplicitMPM(sand_model, config=SolverImplicitMPM.Config(...))` 并调用 `setup_collider(model=rb_model)`。每个外循环周期先由 `compute_body_forces` 将上步颗粒冲量转化为外力注入刚体模型驱动 `SolverMuJoCo.step` 多子步，随后调用 `subtract_body_force` 从当前刚体速度扣除历史 MPM 冲量增量，再进入 `mpm_solver.step(...)` 并通过 `collect_collider_impulses()` 收集新冲量。
   - **标准耦合接口（`SolverCoupledProxy` / `example_mujoco_mpm_coupled_solver.py`）：** 在 `SolverImplicitMPM` 内实现了 `coupling_rewind_proxy_body` 与 `coupling_harvest_proxy_wrenches`，将代理刚体与 MPM 颗粒耦合纳为通用多求解器调度流程。
2. **`SolverImplicitMPM.step` 内部流程（`_step_impl`）：**
   - `_particles_to_cells(state_in.particle_q)` 与 `_rebuild_scratchpad(pic)` 构建网格求积与临时缓冲；
   - `_rasterize_colliders(...)` 调用 `rasterize_collider` 将刚体/代理 SDF、速度与摩擦系数光栅化到碰撞节点（如 `S2`），必要时通过 SDF 梯度插值法向；
   - `_compute_unconstrained_velocity(...)`（含可选 APIC）装配无约束速度与逆质量矩阵；
   - `_build_collider_rigidity_operator(...)` 组装刚体块对角惯量逆 Delassus 算子；
   - `_build_elasticity_system(...)` 与 `_build_plasticity_system(...)` 构建柔度阵、应变投影与四参数屈服面参数；
   - `_load_warmstart(...)` → `solve_rheology(...)`（特征基旋转 + 着色 GS / Jacobi / 链式求解）→ `_save_for_next_warmstart(...)` → `_update_particles(...)` 完成半隐式平流更新。
