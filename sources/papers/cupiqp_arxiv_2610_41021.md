# CuPIQP（arXiv:2610.41021）

> 来源归档（paper）

- **标题：** CuPIQP: A Batched and Differentiable QP Solver on GPUs for Robot Learning and Control
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.41021>
- **PDF：** <https://arxiv.org/pdf/2610.41021>
- **代码：** <https://github.com/PREDICT-EPFL/cupiqp>
- **入库日期：** 2026-11-03
- **线索来源：** [机器人不仅要完成任务，还要知道哪里出了错](../blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)（具身智能小站，2026-11-03）
- **一句话说明：** CuPIQP 支持稠密、稀疏和多阶段问题，并对各问题独立更新和停止，服务机器人学习中的批量优化。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-03）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 把批量二次规划搬上 GPU。收益需要结合批大小、结构和比较实现判断。
- CuPIQP 支持稠密、稀疏和多阶段问题，并对各问题独立更新和停止，服务机器人学习中的批量优化。
- 指标：公众号转述：收益需要结合批大小、结构和比较实现判断。

## 对 wiki 的映射

- [paper-cupiqp](../../wiki/entities/paper-cupiqp.md)
- [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-03-49-papers-technology-map.md)
