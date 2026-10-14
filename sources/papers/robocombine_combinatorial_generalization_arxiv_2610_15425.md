# RoboCombine（arXiv:2610.15425）

> 来源归档（paper）

- **标题：** Exploring the Limits of Combinatorial Generalization in Embodied Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.15425>
- **PDF：** <https://arxiv.org/pdf/2610.15425>
- **代码：** <https://github.com/AutoLab-SAI-SJTU/RoboCombine>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人动作为什么总在换块时卡顿？｜附代码与项目入口](../blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)（具身智能小站，2026-10-14）
- **一句话说明：** 把长程操作失败定位到子任务交接状态：分别统计有序成功率与交接成功率，区分理想触发与预测触发。

## 开源状态

- **部分开源**（步骤 2.5 核查，2026-10-14）：仓库可达，但公众号称主要提供评测运行时，需自备策略与仿真资产。

## 核心摘录（公众号转述，未核验原文表格）

- 技能单独成功不意味着串联后成功；没有交接级诊断，组合泛化失败只能被笼统归为「长程难」。
- 评测框架记录每个子任务起止状态；提出扩展原子起始状态的 PRTI 以覆盖交接分布。
- 指标：评测协议为主；数值以原文为准。

## 对 wiki 的映射

- [paper-robocombine-combinatorial-generalization](../../wiki/entities/paper-robocombine-combinatorial-generalization.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
