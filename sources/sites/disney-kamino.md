# Disney Research — Kamino Simulator

> 来源归档

- **标题：** Kamino — GPU-Accelerated Physics Solver
- **类型：** site（项目页）
- **来源：** Disney Research, Zurich + NVIDIA, Zurich
- **链接：** https://disneyresearch.github.io/kamino/
- **入库日期：** 2026-09-05
- **一句话说明：** Kamino 是面向闭链/并联机构的 GPU 物理求解器，基于 Warp 并作为 Newton 后端，支持异构并行 RL 训练。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-kamino.md`](../../wiki/entities/paper-kamino.md)、[`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md)

---

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（集成于 [newton-physics/newton](https://github.com/newton-physics/newton) Apache-2.0） |
| **代码** | `newton/_src/solvers/kamino/` → [`sources/repos/newton-kamino-solver.md`](../repos/newton-kamino-solver.md) |
| **论文** | [arXiv:2603.16536](https://arxiv.org/abs/2603.16536) |
| **算法基础** | [arXiv:2504.19771](https://arxiv.org/abs/2504.19771)（约束刚体闭链动力学） |
| **成熟度** | **BETA 1**（README 声明 2026 夏计划 BETA 2；暂不接受社区 PR） |
| **Isaac Lab** | 项目页称 RL 集成 **planned / coming soon**；Newton 示例与 `newton_kamino` preset 已存在 |

---

## 核心定位（项目页摘要，2026-09-05）

- **问题：** 多数仿真器假设运动学树（开链）；闭链机构（四连杆、并联操作臂、多肢耦合关节）常被近似为开链 + 等式约束或 mimic joint，带来 sim-to-real gap 与调参负担。
- **方法：** Kamino **原生支持任意关节拓扑**（含闭链），基于 **极大坐标** + **Proximal-ADMM（PADMM）** 对偶求解器，统一双边关节、关节限位与 Signorini–Coulomb 接触。
- **GPU 批量：** 单 GPU 上数千并行环境；支持 **heterogeneous worlds**（每个并行世界可有不同机器人拓扑）。
- **生态：** Newton 统一 `Model` / `State` / `Solver` 接口；Warp 零拷贝对接 PyTorch / JAX；Isaac Lab 端到端 RL 管线规划中。

## 适用 / 不适用（项目页 When to / When not to）

| 场景 | 建议 |
|------|------|
| 闭链四连杆、并联臂、多环腿机构 | **适用** — 直接仿真真实装配而非近似 |
| 大质量比、冗余约束、强耦合接触 | **适用** — ADMM 约束满足与病态系统鲁棒性 |
| GPU 批量 RL（含异构形态） | **适用** |
| 纯开链无闭环 | **不适用** — 专用 articulated-body 求解器更快 |
| 单环境低延迟实时仿真 | **不适用** — 优先 CPU 专用仿真器 |

## 示例（项目页）

- Basic：`fourbar`、USD 加载
- Robot：`DR Legs`（双足，每腿多嵌套四连杆）、ANYmal-D
- 专有展示：BDX、Olaf、Iron Man（Disney 资产，非公开可复现）

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| 论文贡献、PADMM、DR Legs 4096 环境 RL | [`wiki/entities/paper-kamino.md`](../../wiki/entities/paper-kamino.md) |
| Newton 八求解器谱系与选型 | [`wiki/entities/newton-physics.md`](../../wiki/entities/newton-physics.md) |
| 闭链机构仿真选型 | [`wiki/concepts/humanoid-parallel-joint-kinematics.md`](../../wiki/concepts/humanoid-parallel-joint-kinematics.md) |
| Isaac Lab `newton_kamino` preset | [`wiki/entities/isaac-lab-default-environments.md`](../../wiki/entities/isaac-lab-default-environments.md) |
