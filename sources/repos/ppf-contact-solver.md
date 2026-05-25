# ppf-contact-solver

> 来源归档

- **标题：** ZOZO's Contact Solver (ppf-contact-solver)
- **类型：** repo
- **来源：** ZOZO, Inc. / st-tech（日本最大时尚电商 ZOZOTOWN 技术团队）
- **链接：** https://github.com/st-tech/ppf-contact-solver
- **Stars：** ~2.9k（2026-05，入库时）
- **许可证：** Apache-2.0
- **入库日期：** 2026-05-25
- **一句话说明：** GPU 上大规模并行求解 shell（布料）/ solid（体）/ rod（杆）接触与 FEM 可变形体的离线物理仿真器，配套 JupyterLab、Docker、Windows 独立可执行与 Blender 远程插件。
- **沉淀到 wiki：** 是 → [`wiki/entities/ppf-contact-solver.md`](../../wiki/entities/ppf-contact-solver.md)

---

## 核心定位（README 摘录）

- **对象：** 👚 shell（三角网格壳）、🪵 solid（四面体体）、🪢 rod（杆/索）。
- **方法：** 可变形体用 **FEM** + 符号力 Jacobian；接触侧强调 **无穿透**、无卡顿穿插、三角形应变 **硬上限**（如 5%）。
- **规模：** 宣称极端案例 **>1.8 亿** 接触对；全 GPU **单精度**，无 double。
- **用途：** **离线** 高保真仿真（非实时控制环）；部分示例可交互帧率。
- **前端：** JupyterLab（`from frontend import App`）、Blender 5+ 插件（远程 GPU 求解 + 本地 macOS 可用）、MCP 自然语言驱动仿真。

## 技术材料（关联论文）

| 项 | 链接 |
|----|------|
| 论文 | *A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness*，ACM TOG Vol.43 No.6 |
| DOI | https://dl.acm.org/doi/abs/10.1145/3687908 |
| 论文复现分支 | `sigasia-2024`（维护向，非性能最优；主分支会偏离论文） |
| Docker 论文镜像 | `ghcr.io/st-tech/ppf-contact-solver-compiled-sigasia-2024:latest` |

详见 [`sources/papers/ppf_contact_solver_tog_cubic_barrier.md`](../papers/ppf_contact_solver_tog_cubic_barrier.md)。

## 环境与部署（摘要）

| 项 | 要求 |
|----|------|
| GPU | NVIDIA，CUDA 12.8+；大场景推荐 RTX 4090/5090 |
| 架构 | x86（**不支持 arm64**） |
| 部署 | Docker ~1GB 镜像（`ghcr.io/st-tech/ppf-contact-solver-compiled:latest`）、Windows 10/11 免安装 ~320MB 可执行包 |
| 远程 | vast.ai / AWS / GCE / Scaleway 等模板；Blender 插件经端口连远程求解器 |

> ⚠️ 不要在本地跑 `warmup.py`（README 明确警告难清理）。

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| 引擎定位、与 MuJoCo/Newton 分工 | [`wiki/entities/ppf-contact-solver.md`](../../wiki/entities/ppf-contact-solver.md) |
| TOG 论文方法摘要 | [`wiki/entities/paper-ppf-cubic-barrier-contact-solver.md`](../../wiki/entities/paper-ppf-cubic-barrier-contact-solver.md) |
| 刚体 RL 仿真器对照 | [`wiki/entities/mujoco.md`](../../wiki/entities/mujoco.md)、[`wiki/queries/simulator-selection-guide.md`](../../wiki/queries/simulator-selection-guide.md) |
| 仿真平台索引 | [`references/repos/simulation.md`](../../references/repos/simulation.md) |
