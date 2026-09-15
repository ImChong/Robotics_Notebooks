# Whole body dynamic behavior and control of human-like robots（IJHR 2004）

> 来源归档（ingest）

- **标题：** Whole body dynamic behavior and control of human-like robots
- **类型：** paper / journal
- **作者：** Oussama Khatib, Luis Sentis, Jaeheung Park, James Warren
- **期刊：** International Journal of Humanoid Robotics, Vol. 1, No. 1, pp. 29–43 (2004)
- **DOI：** <https://doi.org/10.1142/S0219843610000027>
- **PDF：** <https://khatib.stanford.edu/publications/pdfs/Sentis_2005_IJHR.pdf>（Khatib 实验室镜像；文件名与卷期对应 IJHR 首发）
- **入库日期：** 2026-09-15
- **一句话说明：** Sentis–Khatib 人形 WBC **系统化起点**：在浮基、接触与平衡约束下，用 **任务/姿态分解 + 递归优先级 + 操作空间动力学** 统一表述全身动态行为与控制。

## 摘要级要点

- **问题：** 人形要在动态环境中同时完成操作与运动，必须在 **平衡稳定、接触支撑、关节限位** 等硬约束下协调全身自由度。
- **理论脉络：** 直接扩展 Khatib **操作空间 formulation（1987）** 到 **浮基人形 + 多接触**；不是独立关节 PID，而是在任务空间写动力学再映射关节力矩。
- **三层控制原语（文内层级）：**
  1. **Constraints（约束）** — 最高优先级：平衡、接触、关节限位、自碰等必须始终满足；
  2. **Operational tasks（操作任务）** — 末端/质心等任务，投影到约束零空间；
  3. **Posture（姿态）** — 在剩余冗余里优化次要姿态（能耗、仿人姿态等）。
- **递归优先级：** 多优先级结构保证 **低优先级任务不破坏高优先级约束**；姿态控制在操作任务零空间内完成。
- **动态约束作优先任务：** 平衡、接触等不作为外部后处理，而是 **作为 priority task** 进入同一控制栈。
- **与后续工作关系：** 为 Sentis 2005 ICHR、**ICRA 2006 人形环境 WBC 框架**、2007 斯坦福博士论文与 **ControlIt! / WBOSC** 软件线提供数学与层级语义基础。

## 对 wiki 的映射

- 新建实体：[paper-khatib-sentis-ijhr-2004-whole-body-dynamic-behavior](../../wiki/entities/paper-khatib-sentis-ijhr-2004-whole-body-dynamic-behavior.md)
- 理论源头交叉：[paper-operational-space-formulation](../../wiki/entities/paper-operational-space-formulation.md)（Khatib 1987）
- 代表延伸：[paper-sentis-khatib-icra-2006-whole-body-control-framework](../../wiki/entities/paper-sentis-khatib-icra-2006-whole-body-control-framework.md)
- 概念链：[whole-body-control](../../wiki/concepts/whole-body-control.md)、[null-space-control](../../wiki/concepts/null-space-control.md)、[hqp](../../wiki/concepts/hqp.md)

## 参考来源（原始）

- Khatib, Sentis, Park, Warren (2004), IJHR 1(1):29–43
- Sentis (2007) Stanford PhD thesis bibliography — 确认题名与卷期
