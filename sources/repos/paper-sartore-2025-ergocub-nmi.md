# paper_sartore_2025_ergocub_nature_machine_intelligence

> 来源归档

- **标题：** Towards a Shared Embodied Intelligence … — Nature Machine Intelligence paper code
- **类型：** repo（论文复现脚本 + 实验数据）
- **机构：** 意大利技术研究院（IIT）AMI / GenerativeBionics
- **链接：** https://github.com/ami-iit/paper_sartore_2025_ergocub_nature_machine_intelligence
- **Zenodo：** https://doi.org/10.5281/zenodo.17011716
- **项目页：** https://ergocub.eu/
- **论文：** https://doi.org/10.1038/s42256-026-01272-2（Nature 页：https://www.nature.com/articles/s42256-026-01272-2）
- **Stars：** ~5（2026-07）
- **入库日期：** 2026-07-21
- **许可证：** BSD-3-Clause
- **代码 / 开源状态：** **已开源** — 硬件优化脚本与输出模型、人机交互实验脚本、实验 `data/`；依赖 [adam](https://github.com/ami-iit/adam) 与 [shared-controllers](https://github.com/ami-iit/shared-controllers)
- **一句话说明：** Nature MI ergoCub 论文的窄域复现仓：按「最优硬件 / 物理智能」两实例组织脚本与数据，不替代底层动力学与 WBC 库。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-ergocub-shared-embodied-intelligence.md`](../../wiki/entities/paper-ergocub-shared-embodied-intelligence.md)
- **交叉归档：** [ergocub-eu.md](../sites/ergocub-eu.md)、[ergocub_shared_embodied_intelligence_nmi_s42256_026_01272_2.md](../papers/ergocub_shared_embodied_intelligence_nmi_s42256_026_01272_2.md)、[ami-iit-adam.md](./ami-iit-adam.md)、[gbionics-shared-controllers.md](./gbionics-shared-controllers.md)

---

## 仓内结构（2026-07 快照）

| 路径 | 作用 |
|------|------|
| `optimal_hardware/`（README 写作 OptimalHardware） | 硬件优化脚本、输出模型、Docker recipe |
| `physical_intelligence/`（README 写作 PhysicalIntelligence） | 人机交互实验脚本（依赖 shared-controllers） |
| `data/` | 实验数据与硬件优化结果 |

## 复现入口（README）

1. 安装依赖库 **adam**、**shared-controllers**（各自 README / 集成测试）。
2. **第一实例：** 按 `OptimalHardware` 文档跑连杆长度优化（含 Docker）。
3. **第二实例：** 按 `PhysicalIntelligence` 文档跑协作实验脚本。

---

## 对 wiki 的映射

- 实体页：[ergoCub Shared Embodied Intelligence](../../wiki/entities/paper-ergocub-shared-embodied-intelligence.md)
- 概念交叉：[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Sim2Real](../../wiki/concepts/sim2real.md)
- 任务交叉：[Locomotion](../../wiki/tasks/locomotion.md)
- 平台对照：[iCub3 Avatar 计划页](../../wiki/entities/paper-notebook-icub3-avatar-system-enabling-remote-fully-immers.md)
