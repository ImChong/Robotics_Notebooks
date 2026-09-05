# RAI Institute — AthenaZero 博客（动态操作硬件背景）

- **标题：** AthenaZero: A Bimanual Robot for Dynamic Manipulation
- **类型：** site / blog / hardware-platform
- **URL：** <https://rai-inst.com/resources/blog/bimanual-robot-for-dynamic-manipulation/>
- **配套论文归档：** [`sources/papers/robot_juggling_arxiv_2608_26800.md`](../papers/robot_juggling_arxiv_2608_26800.md)
- **机构：** 机器人与人工智能研究所（RAI Institute）
- **发布日期：** 2026-04-07（博客页）
- **入库日期：** 2026-09-05

## 一句话摘要

RAI 首款低惯量双臂原型 **AthenaZero**：准直驱（多数关节 **5:1**）、把电机质量收向躯干、无腕部力矩传感器；演示投掷 **70 mph**、短距接球、挥棒与 **纯 onboard 视觉** 三球抛接（与 arXiv:2608.26800 同平台）。

## 开源状态（步骤 2.5，截至 2026-09-05）

| 资源 | 状态 |
|------|------|
| 博客 / 演示视频 | **已发布** |
| 设计 CAD / 控制代码 / 抛接学习栈 | **未列 URL** |
| GitHub | **无官方仓**（juggling 论文亦未链代码） |

**结论：** 硬件叙事页；**不能**当可复现训练入口。抛接学习细节以 arXiv:2608.26800 为准。

## 公开信息要点

- **构型：** 1-DoF 躯干 + 双 7-DoF 臂 + 双 6-DoF 欠驱动手（27 关节 / 22 执行器）；身高约 1.6 m，臂展约 1.8 m。
- **设计哲学：** 降低反射惯量与有效质量，使人臂级柔顺接触 + 人类节奏动态操作成为可能。
- **传感：** 仅靠电机电流估力矩，**无** 六维力传感器。
- **棒球任务：** 投 70 mph、24 ft 内接 41 mph、挥棒 31 mph 接触率 82%；人机/机机对传验证。
- **与抛接论文关系：** 同一 AthenaZero 平台；博客强调硬件，[robot_juggling 论文](../papers/robot_juggling_arxiv_2608_26800.md) 强调 **正则化记忆学习 + MRS** 软件栈。

## 为何值得保留

- 解释抛接论文的 **低惯量/柔顺接触** 硬件前提，避免读者把方法当成通用工业臂可即插即用。
- 与 [Sumo](../../wiki/methods/sumo.md)、[SMPC-to-RL](../../wiki/entities/paper-smpc2rl-loco-manipulation.md)、[ZEST](../../wiki/entities/paper-zest.md) 并列，构成 RAI 动态操作研究线。

## 关联资料

- 论文归档：[`sources/papers/robot_juggling_arxiv_2608_26800.md`](../papers/robot_juggling_arxiv_2608_26800.md)
- wiki 实体：[`wiki/entities/paper-robot-juggling-athenazero.md`](../../wiki/entities/paper-robot-juggling-athenazero.md)
