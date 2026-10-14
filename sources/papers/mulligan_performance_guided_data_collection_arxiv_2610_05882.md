# Mulligan（arXiv:2610.05882）

> 来源归档（paper）

- **标题：** Mulligan: Performance-Guided Data Collection for Efficient On-Robot Learning
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.05882>
- **PDF：** <https://arxiv.org/pdf/2610.05882>
- **项目页：** <https://mulligan.page/>
- **代码：** <https://github.com/ankile/mulligan>
- **入库日期：** 2026-10-14
- **线索来源：** [机器人越熟练，采集数据越要盯住失败｜附代码与项目入口](../blogs/wechat_embodied_station_34_papers_mulligan_2026-10-13.md)（具身智能小站，2026-10-13）
- **一句话说明：** 先评测再采集：优先重试失败起点、补足未探索状态，策略用成功人类片段训练、价值函数利用全部交互。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-14）：仓库可达。

## 核心摘录（公众号转述，未核验原文表格）

- 真机策略越熟练，均匀采集越重复已掌握状态；采集起点是部署后学习的可优化变量。
- 采集预算、初始状态范围与人工介入条件需同时控制；与价值重排序分开检验。
- 指标：最终轮插笔任务 HiL-IDQL+Mulligan 38/50 vs 均匀 HG-DAgger 21/50；三项真机任务共 2,550 次留出盲测（公众号转述）。

## 对 wiki 的映射

- [paper-mulligan-performance-guided-data-collection](../../wiki/entities/paper-mulligan-performance-guided-data-collection.md)
- [失败导向采集与可组合世界动作模型：具身智能小站 10-13 期 34 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-13-34-papers-technology-map.md)
