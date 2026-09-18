# RAI Institute — AthenaZero 博客（动态操作硬件背景）

- **标题：** AthenaZero: A Bimanual Robot for Dynamic Manipulation
- **类型：** site / blog / hardware-platform
- **URL：** <https://rai-inst.com/resources/blog/bimanual-robot-for-dynamic-manipulation/>
- **配套期刊论文：** [`sources/papers/athenazero_scirobotics_aee1868.md`](../papers/athenazero_scirobotics_aee1868.md)（DOI [10.1126/scirobotics.aee1868](https://doi.org/10.1126/scirobotics.aee1868)，*Science Robotics* 11(118)，2026-09-16）
- **配套学习论文：** [`sources/papers/robot_juggling_arxiv_2608_26800.md`](../papers/robot_juggling_arxiv_2608_26800.md)
- **机构：** 机器人与人工智能研究所（RAI Institute）
- **发布日期：** 2026-04-07（博客页）
- **入库日期：** 2026-09-05（2026-09-17 更新：链 SciRob 论文与开源仓）

## 一句话摘要

RAI 首款低惯量双臂原型 **AthenaZero**：准直驱（多数关节 **5:1**）、把电机质量收向躯干、无腕部力矩传感器；演示投掷 **70 mph**、短距接球、挥棒与 **纯 onboard 视觉** 三球抛接（与 arXiv:2608.26800 同平台）。**期刊版** 见 SciRob aee1868。

## 开源状态（步骤 2.5，截至 2026-09-17）

| 资源 | 状态 |
|------|------|
| SciRob 论文 / 博客 / 演示视频 | **已发布** |
| [effective_mass_analysis](../repos/effective_mass_analysis.md) | **已开源**（MIT） |
| [Zenodo 21939225 / 22002793](https://doi.org/10.5281/zenodo.21939225) | **已发布**（冲击/刚度/Fig.5–6） |
| 完整 CAD / 真机棒球控制栈 | **未列 URL** |
| 抛接学习栈（arXiv:2608.26800） | **确认未开源** |

**结论：** **部分开源** — 有效质量分析与论文图表数据可复现；制造与 demo 控制仍闭源。

## 公开信息要点

- **构型：** 1-DoF 躯干 + 双 7-DoF 臂 + 双 6-DoF 欠驱动手（27 关节 / 22 执行器）；身高约 1.6 m，臂展约 1.8 m。
- **设计哲学：** 降低反射惯量与有效质量，使人臂级柔顺接触 + 人类节奏动态操作成为可能。
- **传感：** 仅靠电机电流估力矩，**无** 六维力传感器。
- **棒球任务：** 投 70 mph、24 ft 内接 41 mph、挥棒 31 mph 接触率 82%；人机/机机对传验证。
- **与论文关系：** 博客为 **叙事入口**；[SciRob aee1868](../papers/athenazero_scirobotics_aee1868.md) 为 **peer-reviewed 硬件论文**；[robot_juggling](../papers/robot_juggling_arxiv_2608_26800.md) 为 **同平台学习线**。

## 为何值得保留

- 解释抛接论文的 **低惯量/柔顺接触** 硬件前提，避免读者把方法当成通用工业臂可即插即用。
- 与 [Sumo](../../wiki/methods/sumo.md)、[SMPC-to-RL](../../wiki/entities/paper-smpc2rl-loco-manipulation.md)、[ZEST](../../wiki/entities/paper-zest.md) 并列，构成 RAI 动态操作研究线。

## 关联资料

- 期刊论文：[`sources/papers/athenazero_scirobotics_aee1868.md`](../papers/athenazero_scirobotics_aee1868.md)
- 学习论文：[`sources/papers/robot_juggling_arxiv_2608_26800.md`](../papers/robot_juggling_arxiv_2608_26800.md)
- 分析代码：[`sources/repos/effective_mass_analysis.md`](../repos/effective_mass_analysis.md)
- wiki 实体：[`wiki/entities/paper-athenazero.md`](../../wiki/entities/paper-athenazero.md)、[`wiki/entities/paper-robot-juggling-athenazero.md`](../../wiki/entities/paper-robot-juggling-athenazero.md)
