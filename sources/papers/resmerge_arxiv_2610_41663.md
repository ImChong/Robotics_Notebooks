# ResMerge（arXiv:2610.41663）

> 来源归档（paper）

- **标题：** ResMerge: Residual Policy Learning and Merging for Continual Adaptation of Pre-trained Robot Policies
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.41663>
- **PDF：** <https://arxiv.org/pdf/2610.41663>
- **项目页：** <https://resmerge.github.io/>
- **代码：** <https://github.com/zhucoffee/resmerge-official>
- **入库日期：** 2026-11-03
- **线索来源：** [机器人不仅要完成任务，还要知道哪里出了错](../blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)（具身智能小站，2026-11-03）
- **一句话说明：** ResMerge 冻结基础策略，以强化学习训练残差，再依据残差场一致性合并专家。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-03）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 持续适配不必无限增加适配器。适合关注持续适配与参数规模的读者，但训练仍需要环境交互与奖励。
- ResMerge 冻结基础策略，以强化学习训练残差，再依据残差场一致性合并专家。
- 指标：公众号转述：适合关注持续适配与参数规模的读者，但训练仍需要环境交互与奖励。

## 对 wiki 的映射

- [paper-resmerge](../../wiki/entities/paper-resmerge.md)
- [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-03-49-papers-technology-map.md)
