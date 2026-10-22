# B-Human '26 – Scalable Team Play up to 11 vs. 11

> 来源归档（paper）

- **标题：** B-Human '26 – Scalable Team Play up to 11 vs. 11
- **作者：** Arne Hasselbring、Tim Laue、Thomas Röfer、Philip Reichenberg
- **机构：** 不来梅大学（Universität Bremen）、德国人工智能研究中心（DFKI）
- **类型：** arXiv 预印本（cs.RO）
- **arXiv：** <https://arxiv.org/abs/2610.14733>
- **PDF：** <https://arxiv.org/pdf/2610.14733>
- **官方代码：** <https://github.com/bhuman/BHumanCodeRelease>
- **官方文档：** <https://docs.b-human.de/coderelease2026/>
- **入库日期：** 2026-10-22
- **一句话说明：** 介绍 B-Human 在转入 Humanoid Soccer League 后的系统调整、不同队伍规模下的行为架构，以及 11v11 展示赛中的策略与位置分配。

## 摘要要点

论文报告 B-Human 在 2026 年 Humanoid Soccer League Middle Division 的比赛成绩，并讨论了从 NAO 标准平台转向体型和重量更大的 Booster 人形平台时所需的系统调整。团队常规参加 3v3 和 5v5 比赛，并用 11v11 展示赛展示行为架构的扩展能力。

官方 2026 代码发布文档补充了可核实的实现边界：常规策略面向最多五名球员，适配更小队伍；11v11 演示使用独立策略，并采用更高效的位置/角色分配算法来实时安排十名场上球员。不要把演示赛描述成该年度常规竞赛规模。

## 开源状态

- **完整比赛代码：** 已发布于 [BHumanCodeRelease](https://github.com/bhuman/BHumanCodeRelease)，为年度代码快照；使用前应阅读仓库许可及 2026 文档。
- **RL 运动策略：** [MachineLearning/IsaacGymRL](https://github.com/bhuman/MachineLearning/tree/master/IsaacGymRL) 提供起立、行走、踢球策略的训练框架和预训练策略，是运动子系统资料，不等同于完整比赛代码。
- **官方说明：** [2026 Code Release 文档](https://docs.b-human.de/coderelease2026/)；[2026 更新记录](https://docs.b-human.de/coderelease2026/changelog/2026/)。

## 对 wiki 的映射

- [B-Human '26：可扩展人形足球团队协作](../../wiki/entities/paper-b-human-26-scalable-team-play.md)
