# CS-VGB（arXiv:2610.15281）

> 来源归档（paper）

- **标题：** Imperfect Verifiers, Exact Targets: Inference-Time Scaling for Continuous Generation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.15281>
- **PDF：** <https://arxiv.org/pdf/2610.15281>
- **代码：** <https://github.com/KraitGit/CS_VGB>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人动作为什么总在换块时卡顿？｜附代码与项目入口](../blogs/wechat_embodied_station_41_papers_streamingwam_2026-10-14.md)（具身智能小站，2026-10-14）
- **一句话说明：** 通过配对去噪与再加噪的回溯搜索，在验证器有误差时仍保持目标奖励倾斜分布的精确性。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-14）：仓库可达。

## 核心摘录（公众号转述，未核验原文表格）

- 推理期用验证器引导连续生成时，验证器误差会把采样分布推离目标；需要有保证的搜索规则。
- 搜索在去噪/再加噪配对转移上进行；理论保证依赖其转移与正则条件；除图像外测试多机器人规划。
- 指标：图像生成与多机器人规划实验；数值以原文为准。

## 对 wiki 的映射

- [paper-cs-vgb-inference-time-scaling](../../wiki/entities/paper-cs-vgb-inference-time-scaling.md)
- [流式动作块与世界模型记忆：具身智能小站 10-14 期 41 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md)
