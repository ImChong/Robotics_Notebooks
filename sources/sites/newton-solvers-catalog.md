# Newton Physics — `newton/_src/solvers` 求解器目录

> 来源归档

- **标题：** Newton Solver Backends Catalog
- **类型：** site（仓库源码目录再核）
- **来源：** [newton-physics/newton `main`](https://github.com/newton-physics/newton/tree/main/newton/_src/solvers)
- **链接：** https://github.com/newton-physics/newton/tree/main/newton/_src/solvers
- **入库日期：** 2026-09-05
- **再核日期：** 2026-09-05
- **一句话说明：** Newton 公开懒加载八求解器 + 内部 `coupled` 多求解器耦合 API 的源码目录索引与选型要点。
- **沉淀到 wiki：** 是 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)

---

## 目录清单（`main` 分支，2026-09-05）

| 子目录 | 公开 API 类 | 模块文件 | 典型用途 |
|--------|-------------|----------|----------|
| `mujoco/` | `SolverMuJoCo` | `solver_mujoco.py` | **主刚体后端** — MuJoCo Warp（MJCF 资产、接触丰富机器人 RL） |
| `featherstone/` | `SolverFeatherstone` | `solver_featherstone.py` | Featherstone 递推刚体 + 半隐式接触/粒子/肌肉核 |
| `xpbd/` | `SolverXPBD` | `solver_xpbd.py` | 扩展位置基动力学（XPBD）约束与接触 |
| `semi_implicit/` | `SolverSemiImplicit` | `solver_semi_implicit.py` | 半隐式积分通用求解器（体/粒子/肌肉/接触核） |
| `vbd/` | `SolverVBD` | `solver_vbd.py` | Vertex Block Descent — 可变形体、刚–软耦合 |
| `style3d/` | `SolverStyle3D` | `solver_style3d.py` | Style3D 布料管线（PD 矩阵、碰撞、非线性步） |
| `implicit_mpm/` | `SolverImplicitMPM` | `implicit_mpm_model.py` 等 | 隐式 MPM — 颗粒、雪、流体、刚–颗粒双向耦合 |
| `kamino/` | `SolverKamino` | `solver_kamino.py` | **闭链/任意拓扑** 约束多体 — PADMM（BETA） |
| `coupled/` | （内部） | `solver_coupled.py` | **统一多求解器耦合 API**（`CouplingInterface`；非 `__all__` 公开导出） |

`__init__.py` 通过 PEP 562 懒加载导出：`SolverBase`、`SolverFeatherstone`、`SolverImplicitMPM`、`SolverKamino`、`SolverMuJoCo`、`SolverSemiImplicit`、`SolverStyle3D`、`SolverVBD`、`SolverXPBD`、`style3d` 子包。

## 选型速查

| 任务 | 优先后端 |
|------|----------|
| 开链腿/臂 RL、MJCF 生态 | `SolverMuJoCo`（MJWarp） |
| 布料 / 缆索 | `SolverStyle3D` 或 `SolverVBD` |
| 颗粒 / 雪 / 软–刚耦合 | `SolverImplicitMPM` |
| 四连杆、并联机构、原生闭链 | `SolverKamino`（BETA；纯开链仍用 MJWarp/Featherstone 更快） |
| 位置基约束原型 | `SolverXPBD` |
| 多物理同场景耦合 | `coupled/`（开发中统一 API） |

## 示例入口（README / 文档交叉）

| 求解器 | Newton 示例名 |
|--------|---------------|
| MuJoCo | `robot_g1`, `robot_anymal_d`, `robot_policy` |
| Kamino | `kamino_basic_fourbar`, `kamino_robot_anymal_d` |
| Style3D | `cloth_style3d`, `cloth_franka` |
| ImplicitMPM | `mpm_granular`, `mpm_anymal`, `mpm_twoway_coupling` |
| DiffSim | `diffsim_*`（多走非 MJWarp 路径） |

## 对 wiki 的映射

- 引擎实体与 Mermaid 循环 → [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)
- Kamino 论文与闭链 → [`wiki/entities/paper-kamino.md`](../../wiki/entities/paper-kamino.md)
- Isaac Lab preset `newton_mjwarp` / `newton_kamino` → [`wiki/entities/isaac-lab-default-environments.md`](../../wiki/entities/isaac-lab-default-environments.md)
