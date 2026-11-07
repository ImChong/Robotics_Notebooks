# TSID（stack-of-tasks）

> 来源归档

- **标题：** TSID — Task Space Inverse Dynamics
- **类型：** repo
- **来源：** Stack of Tasks / LAAS-CNRS / INRIA（Andrea Del Prete, Justin Carpentier 等）
- **链接：** <https://github.com/stack-of-tasks/tsid>
- **许可：** BSD-2-Clause
- **文档：** 仓库 wiki；教学页 <https://andreadelprete.github.io/#teaching>
- **入库日期：** 2026-08-13（首次）；2026-11-07（README 与关键源码通读更新）
- **版本：** 默认分支 `devel`；CHANGELOG 最新 1.10.0（2026-04-14）
- **核对范围：** 通读官方 README（安装 / 文档入口 / 署名与引用）、CHANGELOG 前部；读 `include/tsid/formulations/inverse-dynamics-formulation-acc-force.hpp` 与对应 `.cpp`（`addTask`、`decodeSolution`、`updateTaskWeight`）、`include/tsid/solvers/solver-HQP-base.hpp`、`src/solvers/solver-HQP-eiquadprog.cpp` 的层数检查、`src/contacts/contact-6d.cpp` 的摩擦锥线性化。未逐文件读 tasks / trajectories / python 绑定，未运行 exercises，未读仓库 wiki 页（GitHub 网页不可达）。
- **通读：** 已完成
- **一句话说明：** 基于 Pinocchio 的优化型任务空间逆动力学库；用 **HQP / 加权 QP** 实现任务优先级，是显式零空间投影器在带不等式（接触、限位）时的工程替代。
- **沉淀到 wiki：** [`wiki/concepts/tsid.md`](../../wiki/concepts/tsid.md)、[`wiki/concepts/null-space-control.md`](../../wiki/concepts/null-space-control.md)

## 开源核查（2026-08-13）

| 项 | 结论 |
|----|------|
| GitHub | `stack-of-tasks/tsid`；C++；BSD-2-Clause；默认分支 `devel` |
| 安装 | `conda install tsid -c conda-forge`；robotpkg `robotpkg-py3*-tsid`；源码依赖 Pinocchio + eiquadprog |
| 可运行入口 | `exercises/`（机械臂 / 人形 / 四足 Python 例）；`demo/demo_romeo.py`；`script/test_formulation.py` |
| 无独立项目页 | 以 GitHub + wiki + Del Prete 教学页为准 |

**结论：已开源。** 本库 wiki 此前只外链 GitHub，未建 `sources/repos/` 归档。

## 与零空间投影的关系

TSID **不**在用户代码里手写 $N=I-J^+J$。优先级改由分层 QP 表达：上层最优值成为下层等式约束——这是 Kanoun 2011 / Escande 2014 对 Nakamura 任务优先级的不等式推广。7 轴臂仍可作为 TSID 的「末端任务 + 关节正则」最小例子，但官方 demo 更偏人形 Romeo。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [TSID 概念](../../wiki/concepts/tsid.md) | 方法页 |
| [Pinocchio](./pinocchio.md) | 动力学后端 |
| [HQP](../../wiki/concepts/hqp.md) | 求解结构 |
| [零空间控制](../../wiki/concepts/null-space-control.md) | 连续时间投影 vs QP 分层对照 |


## 源码核对要点（2026-11-07，均读自 devel 分支）

- **决策变量：** `InverseDynamicsFormulationAccForce` 的 QP 变量是 **广义加速度 `dv`（m_v 维）与接触力 `f`（m_k 维）**，不含 $\tau$；`decodeSolution` 用动力学的驱动行事后算出 $\tau = M_a\,dv + h_a - J_a^\top f$。基座动力学（欠驱动行）作为 **优先级 0 的等式约束** 加入。
- **优先级与权重：** `addMotionTask(task, weight, priorityLevel)`；优先级 0 为硬约束层（`updateTaskWeight` 注释写明"第一优先级层权重无意义"），其余层是 **带权重的最小二乘**。
- **层数限制：** 默认的 `SolverHQuadProg`（eiquadprog）在 `retrieveQPData` 里检查 `problemData.size() > 2` 并报 "Solver not implemented for more than 2 hierarchical levels"，即 **同时只支持 2 层**（硬约束层 + 加权任务层）；目录下另有 `solver-HQP-eiquadprog-fast` 等求解器（同样检查层数）与 `solver-proxqp`。
- **接触：** `Contact6d` 用 **线性化摩擦锥（金字塔）** 加法向力上下界 `fMin/fMax`；`HQPBase` 接口提供 `setUseWarmStart`、最大迭代数与最大求解时间设置。
- **与论文的差别：** Del Prete 2014 原论文是伪逆 + 零空间投影的解析解（无 QP、无不等式、全驱动）；本库是其后的 **加速度-力 QP 实现**（浮动基座 + 不等式），工程上以"硬约束层 + 加权层"为主，不是多层严格 HQP。
