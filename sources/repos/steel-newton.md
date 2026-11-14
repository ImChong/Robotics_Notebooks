# Steel — Newton 中的 SolverSteel 实现（源码归档）

- **项目：** Steel / Newton Physics Engine
- **求解器路径：** https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/steel
- **核心文件：** https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/steel/solver_steel.py
- **导出文件：** https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/steel/__init__.py
- **公告仓 README：** https://github.com/skild-ai/steel/blob/main/README.md
- **许可：** Apache-2.0
- **核验日期：** 2026-11-10（源码阅读；非固定 commit），2026-11-14 复核官网评测配置
- **核对范围：** 阅读公告仓完整 README、Newton 的 steel/__init__.py 和 solver_steel.py 中 SolverSteel 类说明、构造参数、step、_stage1_fk_id、_solve_matrix_free 与关联阶段；未运行 Warp/Isaac Lab 基准。
- **通读：** 已完成

## 实现映射

1. `steel/__init__.py` 导出 `SolverSteel`；主类在 `solver_steel.py`，继承 `SolverBase` 与 `CouplingInterface`。
   - **对 wiki 的映射：** [Steel 实体](../../wiki/entities/steel-newton-solver.md) 的时序图。
2. 求解器 `step(state_in, state_out, control, contacts, dt)` 对模型/接触输入执行一步动力学更新；由 `_stage1_fk_id` 等阶段计算正向运动学、逆动力学偏置、质量矩阵和速度预测。
   - **对 wiki 的映射：** 时序图的输入、动力学阶段。
3. 约束行涵盖接触及关节限位等，`_solve_matrix_free` 根据配置组织矩阵自由响应并运行 PGS；最终从约束速度积分得到新状态。源码支持按求解模式、批次和功能选择不同内核。
   - **对 wiki 的映射：** 时序图的约束求解与状态写回。
4. 官网 **20 项单任务** 性能对比使用 opt-in 的 `reuse_kinematics=True`（Newton 1.8）；**异构混合** 评测启用了 joint velocity limits，所以 `reuse_kinematics` 未激活。不可将两组测试参数混淆；还需锁定 Isaac Sim/PhysX 版本与 batch。
   - **对 wiki 的映射：** 评测口径。

**复现边界：** 已确认入口和计算路径；未执行 `SolverSteel` 实例，也未独立测得 3.8×/13×，不能把官网数据表述为复测结论。

## 对 wiki 的映射

- [Steel：Newton 跨本体 GPU 刚体求解器](../../wiki/entities/steel-newton-solver.md)
- [官方博客归档](../sites/steel-skild-ai.md)
