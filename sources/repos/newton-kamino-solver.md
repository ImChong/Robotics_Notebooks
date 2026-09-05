# Newton — Kamino Solver (`newton/_src/solvers/kamino`)

> 来源归档

- **标题：** SolverKamino（Newton Kamino 后端）
- **类型：** repo（子模块路径）
- **来源：** Disney Research + NVIDIA + Google DeepMind（经 Newton 仓库分发）
- **链接：** https://github.com/newton-physics/newton/tree/main/newton/_src/solvers/kamino
- **父仓库：** https://github.com/newton-physics/newton
- **入库日期：** 2026-09-05
- **许可证：** Apache-2.0（随 Newton）
- **一句话说明：** Newton 内 BETA 求解器，用 PADMM 仿真任意拓扑约束多体系统（含闭链），示例含 fourbar、DR Legs、ANYmal-D。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-kamino.md`](../../wiki/entities/paper-kamino.md)、[`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)

---

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（BETA 1；README 声明暂不建议生产依赖，2026 夏目标 BETA 2） |
| **入口类** | `SolverKamino`（`solver_kamino.py`） |
| **配置** | `config.py` |
| **示例** | `kamino/examples/`（如 `example_sim_dr_legs.py`）；Newton 主示例 `kamino_basic_fourbar`、`kamino_robot_anymal_d` |
| **测试** | `kamino/tests/`（`python -m unittest discover`） |
| **社区贡献** | README：**不接受**外部 PR（开发团队带宽限制） |

## 能力列表（README 摘要，2026-09-05）

- 任意关节拓扑的约束刚体多体系统（**不假设运动学树**）
- 双边关节约束（含高级类型）、关节限位、带空间摩擦与恢复系数的接触
- 关节 Coulomb 摩擦、显式/隐式 PD 驱动的有界关节力矩
- 可按约束子集配置的约束稳定化
- 硬关节限位与接触 — **Proximal-ADMM 前向动力学求解器**

## 开发依赖（README）

- Newton + Warp（nightly 或源码）；MuJoCo pre-release；可选 MJWarp 源码安装
- Linux 需 X11 / GL 开发库（Viewer）

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| 算法与论文 | [`wiki/entities/paper-kamino.md`](../../wiki/entities/paper-kamino.md) |
| 项目页叙事 | [`sources/sites/disney-kamino.md`](../sites/disney-kamino.md) |
| 全求解器目录 | [`sources/sites/newton-solvers-catalog.md`](../sites/newton-solvers-catalog.md) |
