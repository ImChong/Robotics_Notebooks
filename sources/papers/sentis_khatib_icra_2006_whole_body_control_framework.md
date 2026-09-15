# A Whole-Body Control Framework for Humanoids Operating in Human Environments（ICRA 2006）

> 来源归档（ingest）

- **标题：** A Whole-Body Control Framework for Humanoids Operating in Human Environments
- **类型：** paper / conference
- **作者：** Luis Sentis, Oussama Khatib
- **会议：** IEEE International Conference on Robotics and Automation (ICRA) 2006, pp. 2641–2648
- **DOI：** <https://doi.org/10.1109/ROBOT.2006.1642100>
- **PDF：** <https://khatib.stanford.edu/publications/pdfs/Sentis_2006_ICRA.pdf>
- **入库日期：** 2026-09-15
- **一句话说明：** 将 IJHR 2004 的全身动态行为理论 **工程化为可运行框架**：约束–操作任务–姿态三层优先级、浮基动力学与支撑接触纳入同一层级，面向人形在 **人类环境** 中的操作/运动/安全接触。

## 摘要级要点

- **动机：** 未来人形将在人类环境中工作，需要 **高效操作与运动技能** 以及 **安全接触交互**。
- **框架集成：** **任务导向动态控制 + 控制优先级**；在遵守物理与运动相关约束的同时控制多个任务原语。
- **优先级语义（与 IJHR 2004 一致）：**
  - 顶层：**约束处理任务**（constraint-handling）；
  - 中层：**操作任务** 投影到约束零空间；
  - 底层：**姿态** 在剩余冗余内控制。
- **运动学层集成：** 层级直接建在 **运动学层**，程序可在运行时 **监测行为可行性**（feasibility）。
- **浮基模型：** 开发人形 **自由漂浮模型**，将支撑接触动力学效应纳入控制层级。
- **操作空间多级合成：** 可在多个层级合成 **operational space 控制器**；结合阻抗控制实现柔顺交互与软姿态。
- **工程背景：** 文内提及与 **Honda** 长期合作、向 **ASIMO** 实现该框架（代码未随论文公开）。
- **开源状态：** **未开源**（经典会议论文；无可运行官方仓库）。

## 对 wiki 的映射

- 新建实体：[paper-sentis-khatib-icra-2006-whole-body-control-framework](../../wiki/entities/paper-sentis-khatib-icra-2006-whole-body-control-framework.md)
- 系统化起点：[paper-khatib-sentis-ijhr-2004-whole-body-dynamic-behavior](../../wiki/entities/paper-khatib-sentis-ijhr-2004-whole-body-dynamic-behavior.md)
- 理论源头：[paper-operational-space-formulation](../../wiki/entities/paper-operational-space-formulation.md)
- 软件后继：[controlit](../../wiki/entities/controlit.md)（WBOSC 开源实现）
- 概念：[whole-body-control](../../wiki/concepts/whole-body-control.md)、[mpc-wbc-integration](../../wiki/concepts/mpc-wbc-integration.md)

## 参考来源（原始）

- Sentis & Khatib, ICRA 2006 — DOI 10.1109/ROBOT.2006.1642100
- Khatib 实验室 PDF 镜像
