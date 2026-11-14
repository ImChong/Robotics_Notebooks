# acados

> 来源归档

- **标题：** acados
- **类型：** repo / mpc-solver
- **链接：** https://github.com/acados/acados
- **维护：** acados 社区（SYSCOP / 多机构贡献）
- **许可：** BSD-2-Clause
- **入库日期：** 2026-09-19
- **一句话说明：** 面向嵌入式与机器人的快速非线性 MPC / OCP 求解器：基于 RTI、结构 exploiting QP，常用于腿足 / 机械臂 WBC 与 NMPC 栈。
- **开源状态：** **已开源**（C 核心 + Python/MATLAB/Simulink/Octave 接口，BSD-2-Clause）
- **核对范围：** 核查 `acados/acados`（commit `1d682b2`）的 `README.md`、`interfaces/acados_template/acados_template/acados_ocp_solver.py`、`acados/ocp_nlp/ocp_nlp_sqp.c`、`acados/ocp_nlp/ocp_nlp_sqp_rti.c`、`c_templates_tera/acados_solver.in.c`（`acados_solve`）、`examples/acados_python/getting_started/minimal_example_ocp.py`，以及官网 Sphinx 文档 `docs.acados.org`（2026-11-13 构建）。
- **通读：** 已完成
- **沉淀到 wiki：** [acados](../../wiki/entities/acados.md)

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| GitHub README | **已开源**（BSD-2-Clause）；含架构总览、安装、Discourse 论坛与 `examples/` |
| 项目页 | 官方 Sphinx 文档 <https://docs.acados.org/>（覆盖 Python/MATLAB/Simulink 接口、`AcadosOcp` / `AcadosMultiphaseOcp`、`leap-c` 等） |
| 代码 | **可运行** — `acados_template` Python 代码生成与 `ctypes` 绑定 + BLASFEO/HPIPM C 核心；本地安装即编译调用 |

## 源码与文档核心核查（commit `1d682b2`）

1. **求解入口与代码生成链路（`acados_ocp_solver.py`、`c_templates_tera/acados_solver.in.c`）：**
   - `AcadosOcpSolver.__init__` 先执行 `ocp.make_consistent`、检查 `is_code_reuse_possible`，随后通过 `generate` 生成 CasADi C 外部函数、`dump_to_json` 写入问题描述 JSON、Tera `render_templates` 生成 C 求解器源码，再由 `make` 或 `CMakeBuilder` 编译为 `libacados_ocp_solver_<name>.so`。
   - 运行期通过 `ctypes` 加载 `libacados` 与生成的求解器动态库，调用 `<name>_acados_create_capsule`、`<name>_acados_create` 与 `__get_pointers_solver` 完成内存分配；每次调用 `solve()` / `solve_for_x0(x0_bar)` 即更新首阶段 `lbx`/`ubx` 并触发 `<name>_acados_solve` → `ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out)`。
2. **核心 NLP 算法（`ocp_nlp_sqp.c`、`ocp_nlp_sqp_rti.c`）：**
   - 与 2020 年早期论文（曾将线搜索列为未来工作）相比，当前 `ocp_nlp_sqp.c` 主循环已集成完整子模块调度：`ocp_nlp_approximate_qp_matrices` → `ocp_nlp_approximate_qp_vectors_sqp` → `ocp_nlp_add_levenberg_marquardt_term` → `ocp_nlp_res_compute` → `ocp_nlp_qpscaling_scale_qp` → `config->regularize->regularize` → `check_termination` → `ocp_nlp_solve_qp_and_correct_dual` → `config->globalization->find_acceptable_iterate`（例如 `MERIT_BACKTRACKING`）。
   - `ocp_nlp_sqp_rti.c` 支持 `rti_phase`（Preparation / Feedback / `PREPARATION_AND_FEEDBACK`）及多步迭代变体（`as_rti_level`、`as_rti_iter`）。
3. **官方示例与核心特性（`minimal_example_ocp.py`、`docs.acados.org`）：**
   - `minimal_example_ocp.py` 演示倒立摆摆起 OCP（`N=20`, `Tf=1.0`, `NONLINEAR_LS`, `IRK`, `PARTIAL_CONDENSING_HPIPM`, `GAUSS_NEWTON`, `SQP`, `MERIT_BACKTRACKING`）。
   - 官方文档（2026-11-13）明确支持多阶段 `AcadosMultiphaseOcp`、MHE、ODE/DAE（含一阶/二阶敏感度与结构利用 GNSF-IRK）、QP 松弛结构利用、解敏感度（solution sensitivities）及与 `leap-c` 强化学习栈集成。

## 对 wiki 的映射

- [acados](../../wiki/entities/acados.md)
- [model-predictive-control](../../wiki/methods/model-predictive-control.md)
- [pinocchio](../../wiki/entities/pinocchio.md) — 常作动力学前端
