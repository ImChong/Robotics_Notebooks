# FreeBar（arXiv:2610.14997）

> 来源归档（paper）

- **标题：** Barrier-Constrained Guidance For Real-time Asynchronous VLA Execution
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.14997>
- **PDF：** <https://arxiv.org/pdf/2610.14997>
- **代码：** <https://github.com/NamelessCrew/FreeBar>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人动作为什么总在换块时卡顿？｜附代码与项目入口](../blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)（具身智能小站，2026-10-14）
- **一句话说明：** 推理期用障碍约束引导对齐异步 VLA 的动作块边界：已承诺前缀严格约束、近未来柔性衔接、远未来保留自由度。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-14）：仓库可达；复现需另备策略检查点与场景资产。

## 核心摘录（公众号转述，未核验原文表格）

- 异步执行时新动作块必须接上机器人正在执行的旧动作；硬拼接或全局约束都会牺牲远期调整空间。
- 按时间分区施加一致性约束，用 barrier 形式在采样过程中引导流匹配策略，不改训练；仿真与真机报告改善。
- 指标：作者在仿真与真机报告改善；具体数字以原文为准。

## 对 wiki 的映射

- [paper-freebar-barrier-guidance-async-vla](../../wiki/entities/paper-freebar-barrier-guidance-async-vla.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
