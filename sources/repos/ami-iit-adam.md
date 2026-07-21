# adam（ami-iit/adam）

> 来源归档

- **标题：** adam — Automatic Differentiation for rigid-body-dynamics Algorithms
- **类型：** repo（可微刚体动力学库）
- **机构：** 意大利技术研究院（IIT）AMI
- **链接：** https://github.com/ami-iit/adam
- **PyPI：** `adam-robotics`（extras：`jax` / `casadi` / `pytorch` / `mujoco` / `usd` / `visualization` / `all`）
- **论文（本库被引用场景）：** https://doi.org/10.1038/s42256-026-01272-2
- **Stars：** ~219（2026-07）
- **入库日期：** 2026-07-21
- **许可证：** BSD-3-Clause
- **代码 / 开源状态：** **已开源** — Featherstone 系浮动基动力学；JAX / CasADi / PyTorch / NumPy 统一接口；conda-forge 亦有对应包
- **一句话说明：** ergoCub 硬件优化与可微人–机耦合模型的核心动力学后端；面向优化与批处理，而非单一仿真器绑定。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-ergocub-shared-embodied-intelligence.md`](../../wiki/entities/paper-ergocub-shared-embodied-intelligence.md)
- **交叉归档：** [paper-sartore-2025-ergocub-nmi.md](./paper-sartore-2025-ergocub-nmi.md)、[ergocub_shared_embodied_intelligence_nmi_s42256_026_01272_2.md](../papers/ergocub_shared_embodied_intelligence_nmi_s42256_026_01272_2.md)

---

## 仓内要点（2026-07 快照）

| 项 | 内容 |
|----|------|
| 后端 | JAX（XLA 编译/向量化/求导）、CasADi（符号优化）、PyTorch（GPU/批）、NumPy |
| 安装 | `pip install adam-robotics[jax]` 等；或 `conda-forge` 的 `adam-robotics-*` |
| 角色（对本论文） | `optimal_hardware/` 优化管线的动力学原语 |

## 对 wiki 的映射

- 实体页「源码运行时序图」中的动力学节点
- 可与 [Pinocchio](./pinocchio.md) 对照：adam 强调 **多后端可微**；Pinocchio 是更广的刚体算法生态
