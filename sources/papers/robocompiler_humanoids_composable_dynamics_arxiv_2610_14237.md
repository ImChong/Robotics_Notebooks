# RoboCompiler-Humanoids（arXiv:2610.14237）

> 来源归档（paper）

- **标题：** Composable Dynamics for Cooperative Humanoids: A RoboCompiler Extension
- **作者：** Mehdi Heydari Shahna、Jouni Mattila
- **机构：** 坦佩雷大学（Tampere University）
- **类型：** arXiv 预印本（cs.RO）
- **提交日期：** 2026-10-09
- **arXiv：** <https://arxiv.org/abs/2610.14237>
- **PDF：** <https://arxiv.org/pdf/2610.14237>
- **代码与结果：** <https://github.com/Mehdi-Heydari-Shahna/RoboCompiler-Humanoids>
- **入库日期：** 2026-10-26
- **一句话说明：** 扩展 RoboCompiler，以可复用的成员级动力学和两级约简建模手牵手、接触支撑不断变化的多个人形机器人。

## 核心摘录（据 arXiv 摘要）

- 将刚体、运动执行器部件及折算转子惯量组装成约简动力学；编译的刚体雅可比与解析约束曲率给出显式 Coriolis 项，并满足浮动基准速度下的功率恒等式。
- 两级约简保留各机器人可复用的内部模型，再组合地面支撑与手部连接；局部惯性更新秩不超过 6，指定约束加速度所需的约束 wrench 对执行器力是仿射映射。
- 在 8 种人形配置的 264 个可行状态上，归一化加速度差异相对硬约束参考不超过 1.6×10^-10，相对独立编译器参考不超过 1.4×10^-10。
- MuJoCo 仿真中的环形舞蹈骨盆 RMS 误差为 0.73 mm；侧步通过 20 次支撑模式切换。
- 以上是模型数值校验与仿真结果；不能据此推断真机性能。

## 开源状态

- **已开源：** 作者仓库公开，包含代码、论文、报告结果和演示视频；许可为 Apache-2.0。
- **复现入口：** README 提供 python reproduce.py --profile smoke 和 --profile full；完整 profile 重跑论文中报告的所有运行约需双核一小时。详见 [官方仓库归档](../repos/robocompiler_humanoids_composable_dynamics.md)。

## 对 wiki 的映射

- [RoboCompiler-Humanoids 实体页](../../wiki/entities/paper-robocompiler-humanoids-composable-dynamics.md)
- [浮动基动力学](../../wiki/concepts/floating-base-dynamics.md)
- [人形多机协调](../../wiki/concepts/humanoid-multi-robot-coordination.md)
