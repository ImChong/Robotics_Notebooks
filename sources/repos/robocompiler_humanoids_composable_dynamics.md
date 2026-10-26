# RoboCompiler-Humanoids 官方仓库

> 来源归档（repo）

- **标题：** Composable Dynamics for Cooperative Humanoids: A RoboCompiler Extension
- **类型：** 官方代码与结果仓库
- **链接：** https://github.com/Mehdi-Heydari-Shahna/RoboCompiler-Humanoids
- **arXiv：** <https://arxiv.org/abs/2610.14237>
- **许可：** Apache-2.0
- **核查日期：** 2026-10-26
- **一句话说明：** RoboCompiler 扩展，提供闭链多人形动力学组合、MuJoCo/Pinocchio 后端、复现实验工作流、结果文件、论文和演示视频。

## 仓库内容与复现

- ipd/：图编译、闭链约束、分层动力学组合、局部惯性更新、约束力映射与仿真后端。
- dancers/：五个人形组成的闭链队形、编舞与侧步执行。
- workflows/：dance、statewise、manipulator 实验及报告生成。
- results/、paper/、videos/：论文报告运行结果、手稿与演示视频。
- README 标注复现环境为 Linux / Python 3.12；安装 requirements.txt 后可运行 smoke 或 full profile，后者重跑论文报告的运行约需双核一小时。
- 仓库描述的对象为合成闭链人形设计（42 个刚体、49 个关节、21 个执行器端口），采用代表性电驱参数；不是五台实体人形的真机实验。
- excavator 状态研究需要用户自备 excavator record，未随仓库分发。

## 与论文的关系

- [论文归档](../papers/robocompiler_humanoids_composable_dynamics_arxiv_2610_14237.md)
- [wiki 实体页](../../wiki/entities/paper-robocompiler-humanoids-composable-dynamics.md)
